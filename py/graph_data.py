#!/usr/bin/env python3

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData


# DEFAULT_REQUIRED_NODE_TYPES: Tuple[str, ...] = ("cell_input", "net_input", "net_output")
DEFAULT_REQUIRED_NODE_TYPES: Tuple[str, ...] = ( "net_input", "net_output")


def _label_should_transform(ntype: str) -> bool:
	return ntype in ("net_input", "net_output")


def compute_feature_stats(graphs: List[dict]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
	stats: Dict[str, List[torch.Tensor]] = {}
	for g in graphs:
		for n in g.get("nodes", []):
			stats.setdefault(n["type"], []).append(torch.tensor(n["features"], dtype=torch.float32))

	norm_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
	for ntype, feats in stats.items():
		x = torch.stack(feats, dim=0)
		mean = x.mean(dim=0)
		std = x.std(dim=0)
		std = torch.where(std < 1e-12, torch.ones_like(std), std)
		norm_stats[ntype] = (mean, std)
	return norm_stats


def compute_label_stats(graphs: List[dict], *, label_log: bool) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
	stats: Dict[str, List[torch.Tensor]] = {}
	for g in graphs:
		for n in g.get("nodes", []):
			if "labels" not in n:
				continue
			ntype = n["type"]
			t = torch.tensor(n["labels"], dtype=torch.float32)
			if label_log and _label_should_transform(ntype):
				t = torch.log1p(t)
			stats.setdefault(ntype, []).append(t)

	label_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
	for ntype, labs in stats.items():
		y = torch.stack(labs, dim=0)
		mean = y.mean(dim=0)
		std = y.std(dim=0)
		std = torch.where(std < 1e-12, torch.ones_like(std), std)
		if ntype == "net_input":
			# Only first dim is meaningful for net_input
			mean = mean[:1]
			std = std[:1]
		elif ntype == "net_output":
			# Only first two dims are meaningful for net_output
			mean = mean[:2]
			std = std[:2]
		label_stats[ntype] = (mean, std)
	return label_stats


def _is_outlier(vec: torch.Tensor, *, mean: torch.Tensor, std: torch.Tensor, sigma: float) -> bool:
	# vec/mean/std are 1D; returns True if any dim exceeds sigma std.
	if sigma <= 0:
		return False
	z = (vec - mean) / std
	return bool(torch.any(torch.abs(z) > float(sigma)).item())


def filter_outlier_graphs(
	graphs: List[dict],
	*,
	feature_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
	label_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
	label_log: bool,
	sigma: float = 3.0,
	check_features: bool = True,
	check_labels: bool = True,
) -> Tuple[List[dict], List[Tuple[int, str]]]:
	"""Remove graphs containing any node whose feature/label deviates from mean by > sigma*std.

	This operates on raw JSON graphs (before normalization) and uses the provided stats.
	"""
	if sigma <= 0 or (not check_features and not check_labels):
		return list(graphs), []

	valid: List[dict] = []
	bad: List[Tuple[int, str]] = []
	for gi, g in enumerate(graphs):
		nodes = g.get("nodes", [])
		reason: Optional[str] = None
		for ni, n in enumerate(nodes):
			ntype = n.get("type")
			if not isinstance(ntype, str):
				continue
			if check_features and ntype in feature_stats and "features" in n:
				mean, std = feature_stats[ntype]
				x = torch.tensor(n["features"], dtype=torch.float32)
				if _is_outlier(x, mean=mean, std=std, sigma=sigma) and ntype != "cell_input":
					reason = f"feature outlier (> {sigma} std) at node {ni} type {ntype}"
					break
			if check_labels and "labels" in n and ntype in label_stats:
				mean, std = label_stats[ntype]
				y = torch.tensor(n["labels"], dtype=torch.float32)
				if label_log and _label_should_transform(ntype):
					y = torch.log1p(y)
				if _is_outlier(y, mean=mean, std=std, sigma=sigma):
					reason = f"label outlier (> {sigma} std) at node {ni} type {ntype}"
					break
		if reason is None:
			valid.append(g)
		else:
			bad.append((gi, reason))
	return valid, bad


def validate_raw_graph(
	graph: dict,
	*,
	graph_index: int,
	required_node_types: Tuple[str, ...] = DEFAULT_REQUIRED_NODE_TYPES,
	require_edges: bool = True,
) -> None:
	nodes = graph.get("nodes", [])
	edges = graph.get("edges", [])

	if not isinstance(nodes, list) or len(nodes) == 0:
		raise ValueError(f"Graph {graph_index} has no nodes")
	if len(nodes) > 12:
		raise ValueError(f"Graph {graph_index} has too many nodes: {len(nodes)} > 12")

	types_present = set()
	for n in nodes:
		if isinstance(n, dict) and "type" in n:
			types_present.add(n["type"])

	missing = [t for t in required_node_types if t not in types_present]
	if missing:
		raise ValueError(f"Graph {graph_index} missing node types: {missing}")

	if require_edges:
		if not isinstance(edges, list) or len(edges) == 0:
			raise ValueError(f"Graph {graph_index} has no edges")


def validate_hetero_data_list(
	data_list: Sequence[HeteroData],
	*,
	required_node_types: Tuple[str, ...] = DEFAULT_REQUIRED_NODE_TYPES,
	require_edges: bool = True,
) -> None:
	for gi, data in enumerate(data_list):
		missing = [t for t in required_node_types if t not in data.node_types]
		if missing:
			raise ValueError(f"Graph {gi} missing node types after processing: {missing}")
		if require_edges:
			total_edges = 0
			for etype in data.edge_types:
				total_edges += int(data[etype].edge_index.size(1))
			if total_edges <= 0:
				raise ValueError(f"Graph {gi} has no edges after processing")


def _load_json_graphs(path: str) -> List[dict]:
	with open(path, "r", encoding="utf-8") as f:
		raw = f.read().lstrip("\ufeff").strip()
	if not raw:
		return []

	def _coerce_to_graph_list(obj: object) -> List[dict]:
		if isinstance(obj, list):
			# Expect list[dict]
			return [g for g in obj if isinstance(g, dict)]
		if isinstance(obj, dict):
			# Support wrappers like {"graphs": [...]} as well as a single graph dict.
			if "graphs" in obj and isinstance(obj.get("graphs"), list):
				return [g for g in obj["graphs"] if isinstance(g, dict)]
			return [obj]
		return []

	# Try full JSON first
	try:
		obj = json.loads(raw)
		graphs = _coerce_to_graph_list(obj)
		if graphs:
			return graphs
	except json.JSONDecodeError:
		pass

	# Fallback: stream of multiple JSON objects (supports pretty-printed multi-line objects)
	decoder = json.JSONDecoder()
	graphs_stream: List[dict] = []
	idx = 0
	last_err: Optional[json.JSONDecodeError] = None
	while idx < len(raw):
		# Skip whitespace between objects
		while idx < len(raw) and raw[idx].isspace():
			idx += 1
		if idx >= len(raw):
			break
		try:
			obj, end = decoder.raw_decode(raw, idx=idx)
		except json.JSONDecodeError as e:
			last_err = e
			break
		graphs_stream.extend(_coerce_to_graph_list(obj))
		idx = end

	# If we managed to parse at least one object and consumed all remaining non-space, accept.
	if graphs_stream:
		j = idx
		while j < len(raw) and raw[j].isspace():
			j += 1
		if j >= len(raw):
			return graphs_stream

	# Fallback: JSONL
	graphs: List[dict] = []
	for line in raw.splitlines():
		line = line.strip()
		if not line:
			continue
		graphs.extend(_coerce_to_graph_list(json.loads(line)))
	return graphs




class JsonGraphDataset(Dataset):
	def __init__(
		self,
		data_path: str,
		*,
		label_norm: bool = False,
		label_log: bool = False,
		feat_norm: bool = False,
		outlier_sigma: float = 3.0,
		filter_outlier_features: bool = False,
		filter_outlier_labels: bool = False,
		required_node_types: Tuple[str, ...] = DEFAULT_REQUIRED_NODE_TYPES,
		require_edges: bool = True,
		validate_graphs: bool = False,
	) -> None:
		super().__init__()
		self.data_path = data_path
		self.label_norm = label_norm
		self.label_log = label_log
		self.feat_norm = feat_norm
		self.outlier_sigma = float(outlier_sigma)
		self.filter_outlier_features = bool(filter_outlier_features)
		self.filter_outlier_labels = bool(filter_outlier_labels)
		self.required_node_types = tuple(required_node_types)
		self.require_edges = bool(require_edges)
		self.validate_graphs = bool(validate_graphs)

		self._norm_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
		self._label_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
		self._data_list: List[HeteroData] = []

		graphs = _load_json_graphs(self.data_path)
		if not graphs:
			self._data_list = []
			return

		# Optional: remove feature/label outlier graphs (> sigma*std) before building data.
		self._norm_stats = self._compute_feature_stats(graphs)
		self._label_stats = self._compute_label_stats(graphs)
		if self.outlier_sigma > 0 and (self.filter_outlier_features or self.filter_outlier_labels):
			graphs_filtered, outlier_bad = filter_outlier_graphs(
				graphs,
				feature_stats=self._norm_stats,
				label_stats=self._label_stats,
				label_log=bool(self.label_log),
				sigma=float(self.outlier_sigma),
				check_features=bool(self.filter_outlier_features),
				check_labels=bool(self.filter_outlier_labels),
			)
			if outlier_bad:
				graphs = graphs_filtered
				# Recompute stats after filtering so normalization matches the filtered population.
				self._norm_stats = self._compute_feature_stats(graphs)
				self._label_stats = self._compute_label_stats(graphs)

		self._data_list = self._build_data_list(graphs)
		if self.validate_graphs:
			validate_hetero_data_list(
				self._data_list,
				required_node_types=self.required_node_types,
				require_edges=self.require_edges,
			)

	def _compute_feature_stats(self, graphs: List[dict]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
		return compute_feature_stats(graphs)

	def _normalize_features(self, ntype: str, x: torch.Tensor) -> torch.Tensor:
		if ntype not in self._norm_stats:
			return x
		mean, std = self._norm_stats[ntype]
		mean = mean.to(x.device)
		std = std.to(x.device)
		if self.feat_norm:
			return (x - mean) / std
		return x

	def _compute_label_stats(self, graphs: List[dict]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
		return compute_label_stats(graphs, label_log=self.label_log)

	def normalize_labels(self, ntype: str, y: torch.Tensor) -> torch.Tensor:
		if self.label_log and _label_should_transform(ntype):
			y = torch.log1p(y)
		if ntype not in self._label_stats:
			return y
		mean, std = self._label_stats[ntype]
		mean = mean.to(y.device)
		std = std.to(y.device)
		if self.label_norm and _label_should_transform(ntype):
			return (y - mean) / std
		return y

	def denormalize_labels(self, ntype: str, y: torch.Tensor) -> torch.Tensor:
		if ntype in self._label_stats and self.label_norm and _label_should_transform(ntype):
			mean, std = self._label_stats[ntype]
			mean = mean.to(y.device)
			std = std.to(y.device)
			y = y * std + mean
		if self.label_log and _label_should_transform(ntype):
			y = torch.expm1(y)
		return y

	def get_label_stats(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
		return dict(self._label_stats)

	def get_feature_stats(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
		return dict(self._norm_stats)

	def _build_data_list(self, graphs: List[dict]) -> List[HeteroData]:
		out: List[HeteroData] = []
		for gi, g in enumerate(graphs):
			if self.validate_graphs:
				validate_raw_graph(
					g,
					graph_index=gi,
					required_node_types=self.required_node_types,
					require_edges=self.require_edges,
				)
			data = HeteroData()
			nodes = g.get("nodes", [])
			edges = g.get("edges", [])

			# Map global node id -> (type, local_index)
			type_lists: Dict[str, List[dict]] = {}
			for n in nodes:
				type_lists.setdefault(n["type"], []).append(n)

			id_to_type_index: Dict[int, Tuple[str, int]] = {}
			for ntype, nlist in type_lists.items():
				for local_idx, n in enumerate(nlist):
					id_to_type_index[n["id"]] = (ntype, local_idx)

			# Build node features and labels per type
			for ntype, nlist in type_lists.items():
				x = torch.tensor([n["features"] for n in nlist], dtype=torch.float32)
				data[ntype].x = self._normalize_features(ntype, x)

				# More robust than checking nlist[0]
				if all("labels" in n for n in nlist):
					y = torch.tensor([n["labels"] for n in nlist], dtype=torch.float32)
					if ntype == "net_output":
						y = y[...,:2]
					elif ntype == "net_input":
						y = y[...,:1]
					data[ntype].y = self.normalize_labels(ntype, y)
			
			# Build edges by node type pairs
			edge_dict: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = {}
			dropped = 0
			for src_id, dst_id in edges:
				if src_id not in id_to_type_index or dst_id not in id_to_type_index:
					dropped += 1
					continue
				src_type, src_local = id_to_type_index[src_id]
				dst_type, dst_local = id_to_type_index[dst_id]
				etype = (src_type, "to", dst_type)
				edge_dict.setdefault(etype, []).append((src_local, dst_local))

			for etype, pairs in edge_dict.items():
				if not pairs:
					continue
				src_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long)
				dst_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long)
				data[etype].edge_index = torch.stack([src_idx, dst_idx], dim=0)

			if self.validate_graphs and self.require_edges:
				total_edges = 0
				for etype in data.edge_types:
					total_edges += int(data[etype].edge_index.size(1))
				if total_edges <= 0:
					raise ValueError(f"Graph {gi} has no valid edges after processing (dropped={dropped})")

			out.append(data)
		return out

	def __len__(self) -> int:
		return len(self._data_list)

	def __getitem__(self, idx: int) -> HeteroData:
		return self._data_list[idx]


def infer_out_dims(dataset: Dataset) -> Dict[str, int]:
	sample = dataset[0]
	out_dims: Dict[str, int] = {}
	for ntype in sample.node_types:
		if "y" in sample[ntype]:
			out_dims[ntype] = sample[ntype].y.size(-1)
	return out_dims


def check_out_dims(dataset: Dataset, out_dims: Dict[str, int]) -> None:
	for i in range(len(dataset)):
		data = dataset[i]
		for ntype, dim in out_dims.items():
			if "y" not in data[ntype]:
				continue
			if data[ntype].y.size(-1) != dim:
				raise ValueError(
					f"Label dim mismatch on graph {i}, type {ntype}: {data[ntype].y.size(-1)} != {dim}"
				)


def inspect_net_output_labels(dataset: JsonGraphDataset, num_samples: int) -> None:
	num_graphs = min(num_samples, len(dataset))
	found = False
	for gi in range(num_graphs):
		data = dataset[gi]
		if "net_output" not in data.node_types or "y" not in data["net_output"]:
			continue
		y = data["net_output"].y
		rows = min(5, y.size(0))
		print(f"graph {gi} net_output y shape: {tuple(y.shape)}")
		for r in range(rows):
			print(f"  row {r}: {y[r].tolist()}")
		found = True

	if not found:
		print("No net_output labels found in inspected graphs.")


def report_label_stats(dataset: JsonGraphDataset, types: Sequence[str]) -> None:
	label_stats = dataset.get_label_stats()
	for ntype in types:
		if ntype not in label_stats:
			continue
		mean, std = label_stats[ntype]
		print(f"Label stats {ntype}: dims={mean.numel()}, mean_abs={mean}, std_mean={std}")


def summarize_graphs(dataset: JsonGraphDataset) -> Dict[str, float]:
	num_graphs = len(dataset)
	total_nodes = 0
	for i in range(num_graphs):
		data = dataset[i]
		total_nodes += int(data.num_nodes)
	avg_nodes = float(total_nodes) / max(num_graphs, 1)
	return {"num_graphs": float(num_graphs), "avg_nodes": float(avg_nodes)}


def split_indices(
	num_graphs: int,
	*,
	train_ratio: float,
	val_ratio: float,
	seed: int,
) -> Tuple[List[int], List[int], List[int]]:
	gen = torch.Generator()
	gen.manual_seed(seed)
	perm = torch.randperm(num_graphs, generator=gen).tolist()
	n_train = int(num_graphs * train_ratio)
	n_val = int(num_graphs * val_ratio)
	train_ids = perm[:n_train]
	val_ids = perm[n_train : n_train + n_val]
	test_ids = perm[n_train + n_val :]
	return train_ids, val_ids, test_ids


def _parse_required_types(raw: str) -> Tuple[str, ...]:
	parts = [p.strip() for p in raw.split(",") if p.strip()]
	return tuple(parts) if parts else DEFAULT_REQUIRED_NODE_TYPES


def _write_json_graphs(path: str, graphs: List[dict]) -> None:
	os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(graphs, f)
		f.write("\n")


def filter_graphs(
	graphs: List[dict],
	*,
	required_node_types: Tuple[str, ...] = DEFAULT_REQUIRED_NODE_TYPES,
	require_edges: bool = True,
) -> Tuple[List[dict], List[Tuple[int, str]]]:
	valid: List[dict] = []
	bad: List[Tuple[int, str]] = []
	for gi, g in enumerate(graphs):
		try:
			validate_raw_graph(
				g,
				graph_index=gi,
				required_node_types=required_node_types,
				require_edges=require_edges,
			)
			valid.append(g)
		except Exception as e:  # noqa: BLE001
			bad.append((gi, str(e)))
	return valid, bad


def main() -> None:
	parser = argparse.ArgumentParser(description="Filter invalid JSON graphs")
	parser.add_argument("--data", type=str, default="my_graph.json", help="Path to JSON/JSONL graph file")
	parser.add_argument(
		"--out_json",
		type=str,
		default="",
		help="Write filtered graphs to this JSON file (default: <data>.filtered.json)",
	)
	parser.add_argument("--label_norm", action="store_true")
	parser.add_argument("--label_log", action="store_true")
	parser.add_argument(
		"--outlier_sigma",
		type=float,
		default=3.0,
		help="Filter out graphs containing any feature/label dim farther than N std from mean (0 disables)",
	)
	parser.add_argument(
		"--no_outlier_feature_filter",
		action="store_true",
		help="Disable feature-based outlier filtering",
	)
	parser.add_argument(
		"--no_outlier_label_filter",
		action="store_true",
		help="Disable label-based outlier filtering",
	)
	parser.add_argument(
		"--required_types",
		type=str,
		default=",".join(DEFAULT_REQUIRED_NODE_TYPES),
		help="Comma-separated required node types per graph",
	)
	parser.add_argument(
		"--allow_no_edges",
		action="store_true",
		help="Allow graphs with zero edges (default requires edges)",
	)
	parser.add_argument(
		"--max_bad_print",
		type=int,
		default=20,
		help="Max number of invalid graphs to print",
	)
	args = parser.parse_args()

	required = _parse_required_types(args.required_types)
	require_edges = not bool(args.allow_no_edges)

	graphs = _load_json_graphs(args.data)
	if not graphs:
		raise RuntimeError(f"No graphs loaded from {args.data}")

	valid, bad = filter_graphs(graphs, required_node_types=required, require_edges=require_edges)
	print(f"Loaded graphs: {len(graphs)} | valid: {len(valid)} | invalid: {len(bad)}")
	if bad:
		for i, (gi, msg) in enumerate(bad[: max(args.max_bad_print, 0)]):
			print(f"  bad[{i}] graph={gi}: {msg}")

	if not valid:
		raise RuntimeError("All graphs are invalid after filtering; no cache written")

	# Outlier filtering based on 3-sigma (configurable)
	outlier_sigma = float(args.outlier_sigma)
	filter_outlier_features = not bool(args.no_outlier_feature_filter)
	filter_outlier_labels = not bool(args.no_outlier_label_filter)
	if outlier_sigma > 0 and (filter_outlier_features or filter_outlier_labels):
		feat_stats = compute_feature_stats(valid)
		lab_stats = compute_label_stats(valid, label_log=bool(args.label_log))
		valid2, out_bad = filter_outlier_graphs(
			valid,
			feature_stats=feat_stats,
			label_stats=lab_stats,
			label_log=bool(args.label_log),
			sigma=outlier_sigma,
			check_features=filter_outlier_features,
			check_labels=filter_outlier_labels,
		)
		print(f"Outlier filter: sigma={outlier_sigma} | kept: {len(valid2)} | dropped: {len(out_bad)}")
		if out_bad:
			for i, (gi, msg) in enumerate(out_bad[: max(args.max_bad_print, 0)]):
				print(f"  outlier_bad[{i}] graph={gi}: {msg}")
		valid = valid2
		if not valid:
			raise RuntimeError("All graphs are outliers after 3-sigma filtering; no cache written")

	out_json = args.out_json if args.out_json else f"{args.data}.filtered.json"
	_write_json_graphs(out_json, valid)
	print(f"Wrote filtered JSON: {out_json}")

	# Optional: run a full in-memory build to ensure the filtered data is parseable.
	_ = JsonGraphDataset(
		out_json,
		label_norm=args.label_norm,
		label_log=args.label_log,
		outlier_sigma=outlier_sigma,
		filter_outlier_features=filter_outlier_features,
		filter_outlier_labels=filter_outlier_labels,
		required_node_types=required,
		require_edges=require_edges,
		validate_graphs=True,
	)
	print("Validated filtered graphs in-memory (no cache written)")


if __name__ == "__main__":
	main()

