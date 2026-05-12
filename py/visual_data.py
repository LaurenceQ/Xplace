#!/usr/bin/env python3

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from graph_data import (
	DEFAULT_REQUIRED_NODE_TYPES,
	JsonGraphDataset,
	_load_json_graphs,
	_parse_required_types,
	_write_json_graphs,
	filter_graphs,
)


class _OnlineVecStats:
	"""Streaming per-dimension stats for vectors (mean/std/min/max)."""

	def __init__(self) -> None:
		self.n: int = 0
		self.sum: Optional[torch.Tensor] = None
		self.sumsq: Optional[torch.Tensor] = None
		self.vmin: Optional[torch.Tensor] = None
		self.vmax: Optional[torch.Tensor] = None

	def update(self, x: torch.Tensor) -> None:
		if x is None or (not torch.is_tensor(x)):
			return
		if x.numel() == 0:
			return
		if x.dim() == 1:
			x = x.unsqueeze(1)
		if x.dim() != 2:
			raise ValueError(f"Expected 2D tensor [N,D], got shape={tuple(x.shape)}")

		xd = x.detach().to(dtype=torch.float64, device="cpu")
		n_batch = int(xd.size(0))

		bsum = xd.sum(dim=0)
		bsumsq = (xd * xd).sum(dim=0)
		bmin = xd.amin(dim=0)
		bmax = xd.amax(dim=0)

		if self.sum is None:
			self.sum = bsum
			self.sumsq = bsumsq
			self.vmin = bmin
			self.vmax = bmax
		else:
			self.sum += bsum
			self.sumsq += bsumsq
			self.vmin = torch.minimum(self.vmin, bmin)  # type: ignore[arg-type]
			self.vmax = torch.maximum(self.vmax, bmax)  # type: ignore[arg-type]

		self.n += n_batch

	def finalize(self) -> Dict[str, torch.Tensor]:
		if self.sum is None or self.sumsq is None or self.vmin is None or self.vmax is None or self.n <= 0:
			return {}
		mean = self.sum / float(self.n)
		var = self.sumsq / float(self.n) - mean * mean
		var = torch.clamp(var, min=0.0)
		std = torch.sqrt(var)
		return {
			"n": torch.tensor(self.n, dtype=torch.long),
			"mean": mean.to(torch.float32),
			"std": std.to(torch.float32),
			"min": self.vmin.to(torch.float32),
			"max": self.vmax.to(torch.float32),
		}


def _ensure_matplotlib():
	try:
		import matplotlib.pyplot as plt  # type: ignore

		return plt
	except Exception:
		return None


def compute_graph_size_stats(dataset: JsonGraphDataset, *, max_graphs: int = 0) -> Dict[str, object]:
	n = len(dataset) if max_graphs <= 0 else min(len(dataset), max_graphs)
	total_nodes: List[int] = []
	total_edges: List[int] = []
	nodes_by_type: Dict[str, List[int]] = {}

	for i in range(n):
		data = dataset[i]
		nodes = 0
		for ntype in data.node_types:
			# Prefer x if present
			if "x" in data[ntype]:
				ni = int(data[ntype].x.size(0))
			else:
				ni = int(getattr(data[ntype], "num_nodes", 0) or 0)
			nodes += ni
			nodes_by_type.setdefault(ntype, []).append(ni)

		edges = 0
		for etype in data.edge_types:
			if "edge_index" in data[etype]:
				edges += int(data[etype].edge_index.size(1))

		total_nodes.append(nodes)
		total_edges.append(edges)

	def _avg(xs: Sequence[int]) -> float:
		return float(sum(xs)) / max(len(xs), 1)

	return {
		"num_graphs": n,
		"total_nodes": total_nodes,
		"total_edges": total_edges,
		"nodes_by_type": nodes_by_type,
		"avg_nodes": _avg(total_nodes),
		"avg_edges": _avg(total_edges),
	}


def compute_feature_label_dim_stats(
	dataset: JsonGraphDataset,
	*,
	max_graphs: int = 0,
) -> Dict[str, Dict[str, Dict[str, torch.Tensor]]]:
	n = len(dataset) if max_graphs <= 0 else min(len(dataset), max_graphs)

	acc: Dict[str, Dict[str, _OnlineVecStats]] = {}
	for i in range(n):
		data = dataset[i]
		for ntype in data.node_types:
			acc.setdefault(ntype, {})
			if "x" in data[ntype]:
				acc[ntype].setdefault("x", _OnlineVecStats()).update(data[ntype].x)
			if "y" in data[ntype]:
				acc[ntype].setdefault("y", _OnlineVecStats()).update(data[ntype].y)

	out: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
	for ntype, kinds in acc.items():
		out[ntype] = {}
		for kind, stat in kinds.items():
			out[ntype][kind] = stat.finalize()
	return out


def plot_graph_size_stats(stats: Dict[str, object], *, plot_dir: str) -> None:
	plt = _ensure_matplotlib()
	if plt is None:
		print("matplotlib not installed; skip plotting. Install: pip3 install matplotlib")
		return

	os.makedirs(plot_dir, exist_ok=True)
	nodes: List[int] = list(stats["total_nodes"])  # type: ignore[assignment]
	edges: List[int] = list(stats["total_edges"])  # type: ignore[assignment]

	plt.figure(figsize=(7, 4))
	plt.hist(nodes, bins=50)
	plt.title("Graph size: total nodes per graph")
	plt.xlabel("nodes")
	plt.ylabel("count")
	plt.tight_layout()
	plt.savefig(os.path.join(plot_dir, "graph_nodes_hist.png"), dpi=150)
	plt.close()

	plt.figure(figsize=(7, 4))
	plt.hist(edges, bins=50)
	plt.title("Graph size: total edges per graph")
	plt.xlabel("edges")
	plt.ylabel("count")
	plt.tight_layout()
	plt.savefig(os.path.join(plot_dir, "graph_edges_hist.png"), dpi=150)
	plt.close()

	plt.figure(figsize=(6, 5))
	plt.scatter(nodes, edges, s=8, alpha=0.6)
	plt.title("Graph size: nodes vs edges")
	plt.xlabel("nodes")
	plt.ylabel("edges")
	plt.tight_layout()
	plt.savefig(os.path.join(plot_dir, "graph_nodes_vs_edges.png"), dpi=150)
	plt.close()


def plot_feature_label_dim_stats(
	stats: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
	*,
	plot_dir: str,
) -> None:
	plt = _ensure_matplotlib()
	if plt is None:
		print("matplotlib not installed; skip plotting. Install: pip3 install matplotlib")
		return

	os.makedirs(plot_dir, exist_ok=True)
	for ntype, kinds in stats.items():
		for kind, s in kinds.items():
			if not s:
				continue
			mean = s["mean"].cpu()
			std = s["std"].cpu()
			vmin = s["min"].cpu()
			vmax = s["max"].cpu()

			d = int(mean.numel())
			x = torch.arange(d).numpy()

			# Matplotlib line plots can look blank for single-point series.
			# Use markers/errorbars when d==1 so mean/std are visible.
			single_dim = d == 1

			plt.figure(figsize=(10, 4))
			if single_dim:
				m = float(mean.view(-1)[0].item())
				lo = float(vmin.view(-1)[0].item())
				hi = float(vmax.view(-1)[0].item())
				plt.errorbar(
					x,
					[m],
					yerr=[[m - lo], [hi - m]],
					fmt="o",
					capsize=4,
					label="mean (error=min/max)",
				)
			else:
				plt.plot(x, mean.numpy(), linewidth=1.0, label="mean")
				plt.fill_between(x, vmin.numpy(), vmax.numpy(), alpha=0.15, label="min/max")
			plt.title(f"{ntype} {kind}: per-dimension mean (band=min/max)")
			plt.xlabel("dimension")
			plt.ylabel("value")
			plt.legend()
			plt.tight_layout()
			plt.savefig(os.path.join(plot_dir, f"{ntype}__{kind}__mean_minmax.png"), dpi=150)
			plt.close()

			plt.figure(figsize=(10, 4))
			if single_dim:
				plt.plot(x, std.numpy(), linestyle="none", marker="o", markersize=6)
			else:
				plt.plot(x, std.numpy(), linewidth=1.0)
			plt.title(f"{ntype} {kind}: per-dimension std")
			plt.xlabel("dimension")
			plt.ylabel("std")
			plt.tight_layout()
			plt.savefig(os.path.join(plot_dir, f"{ntype}__{kind}__std.png"), dpi=150)
			plt.close()


def write_stats_json(*, out_path: str, graph_size: Dict[str, object], feat_label: Dict[str, Dict[str, Dict[str, torch.Tensor]]]) -> None:
	def _to_py(o):
		if isinstance(o, torch.Tensor):
			return o.detach().cpu().tolist()
		if isinstance(o, dict):
			return {k: _to_py(v) for k, v in o.items()}
		if isinstance(o, list):
			return [_to_py(v) for v in o]
		return o

	payload = {
		"graph_size": _to_py(graph_size),
		"feature_label": _to_py(feat_label),
	}
	os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)
		f.write("\n")


def main() -> None:
	parser = argparse.ArgumentParser(description="Visualize graph sizes and per-dimension feature/label stats")
	parser.add_argument("--data", type=str, default="my_graph.json", help="Path to JSON/JSONL graph file")
	parser.add_argument("--label_norm", action="store_true")
	parser.add_argument("--label_log", action="store_true")
	parser.add_argument(
		"--required_types",
		type=str,
		default=",".join(DEFAULT_REQUIRED_NODE_TYPES),
		help="Comma-separated required node types per graph",
	)
	parser.add_argument("--allow_no_edges", action="store_true")
	parser.add_argument(
		"--filter_invalid",
		action="store_true",
		help="Filter invalid graphs first and analyze the filtered JSON",
	)
	parser.add_argument(
		"--out_json",
		type=str,
		default="",
		help="When --filter_invalid, write filtered JSON here (default: <data>.filtered.json)",
	)
	parser.add_argument("--max_graphs", type=int, default=0, help="Analyze at most N graphs (0=all)")
	parser.add_argument("--plot_dir", type=str, default="", help="Directory to save plots")
	parser.add_argument("--stats_json", type=str, default="", help="Write aggregated stats JSON")
	args = parser.parse_args()

	required = _parse_required_types(args.required_types)
	require_edges = not bool(args.allow_no_edges)

	data_path = args.data
	if args.filter_invalid:
		graphs = _load_json_graphs(args.data)
		if not graphs:
			raise RuntimeError(f"No graphs loaded from {args.data}")
		valid, bad = filter_graphs(graphs, required_node_types=required, require_edges=require_edges)
		print(f"Loaded graphs: {len(graphs)} | valid: {len(valid)} | invalid: {len(bad)}")
		out_json = args.out_json if args.out_json else f"{args.data}.filtered.json"
		_write_json_graphs(out_json, valid)
		print(f"Wrote filtered JSON: {out_json}")
		data_path = out_json

	ds = JsonGraphDataset(
		data_path,
		label_norm=args.label_norm,
		label_log=args.label_log,
		required_node_types=required,
		require_edges=require_edges,
		validate_graphs=False,
	)

	plot_dir = args.plot_dir
	if not plot_dir:
		plot_dir = f"{data_path}.plots"
	stats_json = args.stats_json
	if not stats_json:
		stats_json = f"{data_path}.stats.json"

	print(f"Analyzing graphs={len(ds)} (max_graphs={args.max_graphs or 'all'})")
	graph_size = compute_graph_size_stats(ds, max_graphs=args.max_graphs)
	feat_label = compute_feature_label_dim_stats(ds, max_graphs=args.max_graphs)

	print(
		f"Graph size summary: num_graphs={graph_size['num_graphs']} avg_nodes={graph_size['avg_nodes']:.2f} avg_edges={graph_size['avg_edges']:.2f}"
	)
	write_stats_json(out_path=stats_json, graph_size=graph_size, feat_label=feat_label)
	print(f"Wrote stats JSON: {stats_json}")

	plot_graph_size_stats(graph_size, plot_dir=plot_dir)
	plot_feature_label_dim_stats(feat_label, plot_dir=plot_dir)
	print(f"Wrote plots to: {plot_dir}")


if __name__ == "__main__":
	main()
