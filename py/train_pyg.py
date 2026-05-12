#!/usr/bin/env python3
import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, HeteroConv, MessagePassing, SAGEConv

from graph_data import (
    JsonGraphDataset,
    check_out_dims,
    infer_out_dims,
    inspect_net_output_labels,
    report_label_stats,
    split_indices,
    summarize_graphs,
)


def _try_import_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:  # noqa: BLE001
        return None


def _aggregate_metric(metric_by_type: Dict[str, float]) -> float:
    if not metric_by_type:
        return float("nan")
    return float(sum(metric_by_type.values()) / max(len(metric_by_type), 1))


def _nanmean(values: List[float]) -> float:
    xs = [float(v) for v in values if v == v]
    if not xs:
        return float("nan")
    return float(sum(xs) / len(xs))


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    metric_name: str
    train_metric_mean: float
    val_metric_mean: float
    train_loss_by_type: Dict[str, float]
    train_metric_by_type: Dict[str, float]
    val_metric_by_type: Dict[str, float]
    train_metric_per_dim_by_type: Dict[str, List[float]]
    val_metric_per_dim_by_type: Dict[str, List[float]]


def _write_metrics_json(path: str, metrics: List[EpochMetrics], args_dict: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    metric_name = metrics[0].metric_name if metrics else ""
    payload = {
        "args": args_dict,
        "metric": metric_name,
        "epochs": [m.epoch for m in metrics],
        "train_loss": [m.train_loss for m in metrics],
        "train_metric_mean": [m.train_metric_mean for m in metrics],
        "val_metric_mean": [m.val_metric_mean for m in metrics],
        "train_loss_by_type": [m.train_loss_by_type for m in metrics],
        "train_metric_by_type": [m.train_metric_by_type for m in metrics],
        "val_metric_by_type": [m.val_metric_by_type for m in metrics],
        "train_metric_per_dim_by_type": [m.train_metric_per_dim_by_type for m in metrics],
        "val_metric_per_dim_by_type": [m.val_metric_per_dim_by_type for m in metrics],
    }
    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _plot_metrics(path: str, metrics: List[EpochMetrics]) -> bool:
    plt = _try_import_matplotlib_pyplot()
    if plt is None:
        return False

    metric_name = metrics[0].metric_name if metrics else "metric"
    metric_label = "R2" if metric_name == "r2" else ("RMSE" if metric_name == "rmse" else metric_name)

    epochs = [m.epoch for m in metrics]
    train_loss = [m.train_loss for m in metrics]
    train_metric = [m.train_metric_mean for m in metrics]
    val_metric = [m.val_metric_mean for m in metrics]

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(epochs, train_loss, label="train_loss", color="tab:blue")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ax2.plot(epochs, train_metric, label=f"train_{metric_name}(mean)", color="tab:green")
    ax2.plot(epochs, val_metric, label=f"val_{metric_name}(mean)", color="tab:red")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(metric_label)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_metrics_by_type(path: str, metrics: List[EpochMetrics]) -> bool:
    plt = _try_import_matplotlib_pyplot()
    if plt is None:
        return False

    metric_name = metrics[0].metric_name if metrics else "metric"
    metric_label = "R2" if metric_name == "r2" else ("RMSE" if metric_name == "rmse" else metric_name)

    all_types = set()
    for m in metrics:
        all_types.update(m.train_loss_by_type.keys())
        all_types.update(m.train_metric_by_type.keys())
        all_types.update(m.val_metric_by_type.keys())
    types_sorted = sorted(all_types)

    epochs = [m.epoch for m in metrics]

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for t in types_sorted:
        ys = [m.train_loss_by_type.get(t, float("nan")) for m in metrics]
        ax1.plot(epochs, ys, label=f"train_loss[{t}]")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    for t in types_sorted:
        ytr = [m.train_metric_by_type.get(t, float("nan")) for m in metrics]
        yva = [m.val_metric_by_type.get(t, float("nan")) for m in metrics]
        ax2.plot(epochs, ytr, label=f"train_{metric_name}[{t}]", linestyle="-")
        ax2.plot(epochs, yva, label=f"val_{metric_name}[{t}]", linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(metric_label)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_r2_per_dim_trainval(
    path: str,
    metrics: List[EpochMetrics],
    *,
    node_type: str,
    max_dims: int,
) -> bool:
    plt = _try_import_matplotlib_pyplot()
    if plt is None:
        return False
    if not metrics:
        return False
    if metrics[0].metric_name != "r2":
        return False

    epochs = [m.epoch for m in metrics]
    train_vecs = [m.train_metric_per_dim_by_type.get(node_type, []) for m in metrics]
    val_vecs = [m.val_metric_per_dim_by_type.get(node_type, []) for m in metrics]

    dim = max((len(v) for v in train_vecs + val_vecs), default=0)
    if dim <= 0:
        return False

    dims_to_plot = dim if int(max_dims) < 0 else min(dim, int(max_dims))
    if dims_to_plot <= 0:
        return False

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for i in range(dims_to_plot):
        y_tr = [float(v[i]) if i < len(v) else float("nan") for v in train_vecs]
        y_va = [float(v[i]) if i < len(v) else float("nan") for v in val_vecs]
        ax.plot(epochs, y_tr, linewidth=1.2, alpha=0.85, label=f"train:d{i}")
        ax.plot(epochs, y_va, linewidth=1.2, alpha=0.85, linestyle="--", label=f"val:d{i}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("R2")
    ax.set_title(f"R2 per-dim (train/val) for {node_type}  (showing {dims_to_plot}/{dim} dims)")
    ax.grid(True, alpha=0.3)

    # Legend can explode; keep it only when manageable.
    if dims_to_plot <= 10:
        ax.legend(loc="best", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_regression_scatter(
    path: str,
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    title: str,
    max_points: int,
) -> bool:
    plt = _try_import_matplotlib_pyplot()
    if plt is None:
        return False

    if pred.numel() == 0 or target.numel() == 0:
        return False
    if pred.numel() != target.numel():
        raise ValueError(f"pred/target size mismatch: {int(pred.numel())} vs {int(target.numel())}")

    x = pred.flatten()
    y = target.flatten()

    # Optional downsampling for speed/overplot.
    if max_points > 0 and int(x.numel()) > int(max_points):
        idx = torch.randperm(int(x.numel()))[: int(max_points)]
        x = x[idx]
        y = y[idx]

    # Metrics on the plotted subset (still meaningful for sanity checks).
    diff = (x - y)
    rmse = float(torch.sqrt(torch.mean(diff * diff)).item())

    eps = 1e-12
    y_mean = float(y.mean().item())
    ss_res = float(torch.sum((y - x) ** 2).item())
    ss_tot = float(torch.sum((y - y_mean) ** 2).item())
    r2 = float("nan") if ss_tot <= eps else (1.0 - ss_res / ss_tot)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 6.2))
    ax.scatter(x.numpy(), y.numpy(), s=6, alpha=0.35, edgecolors="none")

    # y=x reference
    vmin = float(torch.min(torch.min(x), torch.min(y)).item())
    vmax = float(torch.max(torch.max(x), torch.max(y)).item())
    if math.isfinite(vmin) and math.isfinite(vmax):
        ax.plot([vmin, vmax], [vmin, vmax], color="black", linewidth=1.0, alpha=0.8)

    ax.set_xlabel("pred")
    ax.set_ylabel("target")
    ax.set_title(f"{title}\nN={int(x.numel())}  RMSE={rmse:.4g}  R2={r2:.4g}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _save_checkpoint(
    path: str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
    args_dict: dict,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": args_dict,
        "rng": {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(payload, path)


def _load_checkpoint(
    path: str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    map_location: str,
) -> int:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    rng = ckpt.get("rng") or {}
    try:
        if "python" in rng and rng["python"] is not None:
            random.setstate(rng["python"])
        if "torch_cpu" in rng and rng["torch_cpu"] is not None:
            torch.set_rng_state(rng["torch_cpu"])
        if torch.cuda.is_available() and rng.get("torch_cuda_all") is not None:
            torch.cuda.set_rng_state_all(rng["torch_cuda_all"])
    except Exception:
        # RNG restore is best-effort; training state restore is the priority.
        pass

    return int(ckpt.get("epoch", 0))


def _parse_float_list(raw: str, *, expected_len: int, name: str) -> List[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != expected_len:
        raise ValueError(f"{name} must have {expected_len} comma-separated floats (got {len(parts)})")
    out: List[float] = []
    for p in parts:
        out.append(float(p))
    return out


class CustomEdgeConv(MessagePassing):
    def __init__(self, hidden_channels: int, aggr: str = "add") -> None:
        super().__init__(aggr=aggr)
        self.edge_mlp = nn.Sequential(
            nn.LazyLinear(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.msg_mlp = nn.Sequential(
            nn.LazyLinear(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.update_mlp = nn.Sequential(
            nn.LazyLinear(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def edge_update(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        edge_in = torch.cat([x_i, x_j], dim=-1)
        return self.edge_mlp(edge_in)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        msg_in = torch.cat([x_j, edge_attr], dim=-1)
        return self.msg_mlp(msg_in)

    def update(self, aggr_out: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        out_in = torch.cat([aggr_out, x_i], dim=-1)
        return self.update_mlp(out_in)

    def forward(self, x, edge_index):
        if isinstance(x, tuple):
            x_src, x_dst = x
        else:
            x_src = x_dst = x

        src, dst = edge_index
        edge_attr = self.edge_update(x_dst[dst], x_src[src])
        return self.propagate(edge_index, x=(x_src, x_dst), edge_attr=edge_attr)


class HeteroGNN(nn.Module):
    def __init__(
        self,
        metadata,
        hidden_channels: int,
        num_layers: int,
        out_dims: Dict[str, int],
        conv_type: str = "sage",
        gat_heads: int = 4,
    ):
        super().__init__()
        self.metadata = metadata
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.conv_type = conv_type
        # print(self.edge_types)
        self.lin_in = nn.ModuleDict()
        for ntype in self.node_types:
            self.lin_in[ntype] = nn.Sequential(nn.LazyLinear(hidden_channels, bias=True), nn.LeakyReLU(0.2))

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            if conv_type == "sage":
                conv_dict = {etype: SAGEConv((hidden_channels, hidden_channels), hidden_channels, project=True) for etype in self.edge_types}
            elif conv_type == "gatv2":
                conv_dict = {
                    etype: GATv2Conv(
                        (-1, -1),
                        hidden_channels,
                        heads=gat_heads,
                        concat=False,
                        add_self_loops=False,
                        residual=True,
                    )
                    for etype in self.edge_types
                }
            elif conv_type == "custom":
                conv_dict = {etype: CustomEdgeConv(hidden_channels) for etype in self.edge_types}
            else:
                raise ValueError(f"Unsupported conv_type: {conv_type}")
            conv = HeteroConv(conv_dict, aggr="max")
            self.convs.append(conv)
        self.heads = nn.ModuleDict()
        # Special case: cell_input has 20 labels, grouped by 4 -> 5 independent heads.
        # self.cell_input_heads = nn.ModuleList()
        # self.MLP_msg = nn.Sequential(nn.Linear(2 * hidden_channels, 2 * hidden_channels, bias=True), nn.LeakyReLU(0.2),
        #                              nn.Linear(2 * hidden_channels, 2 * hidden_channels, bias=True), nn.LeakyReLU(0.2),
        #                              nn.Linear(2 * hidden_channels, 2 * hidden_channels, bias=True), nn.LeakyReLU(0.2),
        #                              nn.Linear(2 * hidden_channels, hidden_channels, bias=True), nn.LeakyReLU(0.2)
        #                              )
        for ntype, dim in out_dims.items():
        #     if ntype == "cell_input":
        #         if dim != 20 or (dim % 4) != 0:
        #             raise ValueError(f"cell_input label dim must be 20 (got {dim})")
        #         num_groups = dim // 4
        #         self.cell_input_heads = nn.ModuleList([nn.Linear(hidden_channels, 4) for _ in range(num_groups)])
        #     else:
            self.heads[ntype] = nn.Linear(hidden_channels, dim)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {ntype: self.lin_in[ntype](x) for ntype, x in x_dict.items()}
        # x_list = [self.lin_in[ntype](x) for ntype, x in x_dict.items()]
        # x2 = x_list[0].repeat(x_list[1].size(0), 1)
        # x2_cat = torch.cat([x_list[1], x2], dim=1)
        # print(x2_cat.shape)
        # exit(0)
        
        # x2_msg = self.MLP_msg(x2_cat)
        
        # x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.leaky_relu(v, 0.2) for k, v in x_dict.items()}
            # x_dict = {k: self.MLP[k](v) for k, v in x_dict.items()}
        out: Dict[str, torch.Tensor] = {}
        for ntype, x in x_dict.items():
        #     # if ntype == "cell_input" and len(self.cell_input_heads) > 0:
        #         # parts = [head(x) for head in self.cell_input_heads]
        #         # out[ntype] = torch.cat(parts, dim=-1)
            if ntype in self.heads:
                out[ntype] = self.heads[ntype](x)
                # if ntype == "net_output":
                #     out[ntype] = self.heads[ntype](x2_msg)
                # else :
                #     out[ntype] = self.heads[ntype](x_list[0])
        return out


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    log_iter: int,
    epoch: int,
    loss_type: str,
    clip_grad: float,
    *,
    cell_input_group_weights: List[float],
    net_input_weight: float,
    net_output_weight: float,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_count = 0
    log_loss = 0.0
    per_type_sum: Dict[str, float] = {}
    per_type_count: Dict[str, int] = {}
    # if len(cell_input_group_weights) != 5:
    #     raise ValueError(f"cell_input_group_weights must have length 5 (got {len(cell_input_group_weights)})")
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data.x_dict, data.edge_index_dict)

        loss = None
        for ntype, out in pred.items():
            if "y" not in data[ntype]:
                continue

            target = data[ntype].y

            # # Special: cell_input has 20 labels grouped into 5 independent 4-dim heads.
            # if ntype == "cell_input" and out.size(-1) == 20 and target.size(-1) == 20:
            #     group_weighted_sum = None
            #     for gi in range(5):
            #         sl = slice(gi * 4, (gi + 1) * 4)
            #         if loss_type == "huber":
            #             group_loss = F.smooth_l1_loss(out[:, sl], target[:, sl])
            #         else:
            #             group_loss = F.mse_loss(out[:, sl], target[:, sl])

            #         per_type_sum[f"cell_input_g{gi}"] = per_type_sum.get(f"cell_input_g{gi}", 0.0) + float(
            #             group_loss.detach()
            #         )
            #         per_type_count[f"cell_input_g{gi}"] = per_type_count.get(f"cell_input_g{gi}", 0) + 1

            #         w = float(cell_input_group_weights[gi])
            #         term_w = group_loss * w
            #         print(f"Epoch {epoch:03d} | cell_input group {gi} loss: {group_loss:.6f} weighted by {w}")
            #         group_weighted_sum = term_w if group_weighted_sum is None else (group_weighted_sum + term_w)

            #     # Also log total (weighted) cell_input contribution
            #     if group_weighted_sum is not None:
            #         per_type_sum["cell_input_total_w"] = per_type_sum.get("cell_input_total_w", 0.0) + float(
            #             group_weighted_sum.detach()
            #         )
            #         per_type_count["cell_input_total_w"] = per_type_count.get("cell_input_total_w", 0) + 1
            #         loss = group_weighted_sum if loss is None else (loss + group_weighted_sum)
            #     continue

            if loss_type == "huber":
                term = F.smooth_l1_loss(out, target)
            else:
                term = F.mse_loss(out, target)

            # Optional weights for other types
            if ntype == "net_input":
                term = term * float(net_input_weight)
            elif ntype == "net_output":
                term = term * float(net_output_weight)

            loss = term if loss is None else (loss + term)
            per_type_sum[ntype] = per_type_sum.get(ntype, 0.0) + float(term.detach())
            per_type_count[ntype] = per_type_count.get(ntype, 0) + 1

        if loss is None:
            continue

        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        total_loss += float(loss)
        log_loss += float(loss)
        total_count += 1
        if log_iter > 0 and (total_count % log_iter  == 0):
            print(f"Epoch {epoch:03d} | iter {total_count:06d} | loss {log_loss / max(log_iter, 1):.6f}")
            log_loss = 0.0
    per_type_avg: Dict[str, float] = {}
    for ntype, s in per_type_sum.items():
        per_type_avg[ntype] = float(s / max(per_type_count.get(ntype, 0), 1))
    return total_loss / max(total_count, 1), per_type_avg


   
@torch.no_grad()
def eval_epoch(
    model,
    loader,
    device,
    split: str,
    dataset: JsonGraphDataset,
    *,
    metric: str,
    epoch: Optional[int] = None,
    regression_plot_dir: str = "",

) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    model.eval()
    if metric not in {"rmse", "r2"}:
        raise ValueError(f"Unsupported metric: {metric}")
    plot_regr = bool(regression_plot_dir) and epoch is not None
    # Optional: regression scatter plot collection for net_output(elmore_delay)
    regr_pred: Dict[str, List[torch.Tensor]] = {}
    regr_target: Dict[str, List[torch.Tensor]] = {}


    # For RMSE (scalar)
    sum_sq: Dict[str, float] = {}
    count: Dict[str, int] = {}

    # For R2 (vector over label dims)
    ss_res: Dict[str, torch.Tensor] = {}
    sum_y: Dict[str, torch.Tensor] = {}
    sum_y2: Dict[str, torch.Tensor] = {}
    
    n: Dict[str, int] = {}
    # iter = 0
    for data in loader:
        data = data.to(device)
        out_dict = model(data.x_dict, data.edge_index_dict)

        for ntype, out in out_dict.items():
            if "y" not in data[ntype]:
                continue

            if dataset.label_norm or dataset.label_log:
                pred_t = dataset.denormalize_labels(ntype, out)
                target_t = dataset.denormalize_labels(ntype, data[ntype].y)
            else:
                pred_t = out
                target_t = data[ntype].y

            # Ensure 2D (N, D) for unified metric computation.
            if pred_t.dim() == 1:
                pred_t = pred_t.unsqueeze(-1)
            if target_t.dim() == 1:
                target_t = target_t.unsqueeze(-1)
            diff = pred_t - target_t
            # iter += 1
            # if iter == 1:
                # print(f"normed_pred_t first 5: {pred_t[:5,...].squeeze().tolist()}")
                # print(f"normed_target_t first 5: {target_t[:5,...].squeeze().tolist()}")
                # print(f"origin_pred_t first 5: {out[:5,...].squeeze().tolist()}")
                # print(f"origin_target_t first 5: {data[ntype].y[:5,...].squeeze().tolist()}")
                # print(f"normed_feature first 5: {data[ntype].x[:5,...].squeeze().tolist()}")

            # print(diff.shape)
            if plot_regr:
                if pred_t.dim() == 2 and target_t.dim() == 2 and int(pred_t.size(-1)) == int(target_t.size(-1)):
                    regr_pred.setdefault(ntype, []).append(pred_t.detach().float().cpu())
                    regr_target.setdefault(ntype, []).append(target_t.detach().float().cpu())
                else:
                    raise ValueError(
                        f"Expected target label dim = predict label dim for regression plot, got pred={tuple(pred_t.shape)}, target={tuple(target_t.shape)}"
                    )

            if metric == "rmse":
                sum_sq[ntype] = sum_sq.get(ntype, 0.0) + float((diff * diff).sum())
                count[ntype] = count.get(ntype, 0) + int(diff.numel())
            else:
                # R2 = 1 - SS_res / SS_tot
                # SS_tot = sum((y - mean(y))^2) = sum(y^2) - sum(y)^2 / n
                if ntype not in ss_res:
                    d = int(pred_t.size(1))
                    ss_res[ntype] = torch.zeros(d, dtype=torch.float32, device=pred_t.device)
                    sum_y[ntype] = torch.zeros(d, dtype=torch.float32, device=pred_t.device)
                    sum_y2[ntype] = torch.zeros(d, dtype=torch.float32, device=pred_t.device)
                    n[ntype] = 0

                ss_res[ntype] += (diff * diff).sum(dim=0).float()
                sum_y[ntype] += target_t.sum(dim=0).float()
                sum_y2[ntype] += (target_t * target_t).sum(dim=0).float()
                n[ntype] = n.get(ntype, 0) + int(target_t.size(0))

    if plot_regr:
        for ntype in regr_pred.keys():
            if not regr_pred[ntype] or not regr_target[ntype]:
                continue
            pred_alld = torch.cat(regr_pred[ntype], dim=0) if regr_pred[ntype] else torch.empty(0)
            target_alld = torch.cat(regr_target[ntype], dim=0) if regr_target[ntype] else torch.empty(0)
            for i in range(pred_alld.size(-1)):
                pred_1d = pred_alld[:, i]
                target_1d = target_alld[:, i]
                reg_path = os.path.join(
                    regression_plot_dir,
                    f"{ntype}_epoch_{int(epoch):04d}_{split}_dim_{i}.png",
                )
                ok = _plot_regression_scatter(
                    reg_path,
                    pred=pred_1d,
                    target=target_1d,
                    title=f"{ntype} ({split}) epoch {int(epoch)} dim {i}",
                    max_points=-1,
                )
                if ok:
                    print(f"Wrote regression plot: {reg_path}")
    out_mean: Dict[str, float] = {}
    out_per_dim: Dict[str, List[float]] = {}
    if metric == "rmse":
        for ntype in sum_sq:
            out_mean[ntype] = math.sqrt(sum_sq[ntype] / max(count[ntype], 1))
        return out_mean, out_per_dim

    eps = 1e-12
    for ntype in ss_res:
        nn = max(n.get(ntype, 0), 1)
        ss_res_nt = ss_res[ntype]
        sum_y_nt = sum_y[ntype]
        sum_y2_nt = sum_y2[ntype]

        ss_tot_nt = sum_y2_nt - (sum_y_nt * sum_y_nt) / float(nn)

        # 防止 0 或负数导致的除零/无效
        valid = ss_tot_nt > eps
        r2_nt = torch.empty_like(ss_tot_nt)

        # 对有效维度计算 R^2
        r2_nt[valid] = 1.0 - ss_res_nt[valid] / ss_tot_nt[valid]

        # 对无效维度填 NaN
        r2_nt[~valid] = float("nan")

        r2_list = r2_nt.detach().float().cpu().tolist()
        out_per_dim[ntype] = r2_list
        out_mean[ntype] = _nanmean(r2_list)
    return out_mean, out_per_dim


def format_per_dim_metrics(metrics, float_fmt="{:.4f}", *, max_dims: int = 0):
    """
    metrics: dict[str, 1D tensor/list], 比如:
        {"ntypeA": tensor([0.9, 0.8, ...]), "ntypeB": ...}
    返回: 'ntypeA:d0=0.9000,d1=0.8000 | ntypeB:d0=...'
    """
    parts = []
    for ntype, vec in metrics.items():
        # 把 tensor 转成 list，避免 .item() 一维一维写
        if hasattr(vec, "detach"):
            vec = vec.detach().cpu().tolist()
        vec = list(vec)
        if int(max_dims) > 0:
            vec = vec[: int(max_dims)]
        # 逐维度格式化
        dim_strs = []
        for i, v in enumerate(vec):
            # 处理 NaN：打印成 'nan'
            if v != v:  # NaN 判断
                dim_strs.append(f"d{i}=nan")
            else:
                dim_strs.append(f"d{i}=" + float_fmt.format(v))
        parts.append(f"{ntype}:" + ",".join(dim_strs))
    return " | ".join(parts)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="my_graph.json", help="Path to JSON/JSONL graph file")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--lr", type=float, default=4e-3)
    parser.add_argument("--eta_min", type=float, default=1e-6, help="Minimum LR for cosine scheduler")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conv", type=str, default="sage", choices=["gatv2", "sage", "custom"])
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--log_iter", type=int, default=200)
    parser.add_argument("--plot_epoch", type=int, default=10)
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs",
        help="Directory for metrics outputs (no checkpoints are written)",
    )
    parser.add_argument(
        "--ckpt_every",
        type=int,
        default=2,
        help="Save a training checkpoint every N epochs (0 disables)",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="",
        help="Directory to write checkpoints (default: <out_dir>/checkpoints)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to a checkpoint .pt to resume from (loads model/optimizer/scheduler)",
    )
    parser.add_argument("--label_norm", action="store_true")
    parser.add_argument("--label_log", action="store_true")
    parser.add_argument("--feat_norm", action="store_true")
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
    parser.add_argument("--loss", type=str, default="huber", choices=["huber", "mse"])
    parser.add_argument(
        "--cell_input_group_weights",
        type=str,
        default="1,1,1,1,1",
        help="Comma-separated 5 floats for cell_input group (4-dim) loss weights",
    )
    parser.add_argument(
        "--loss_weight_net_input",
        type=float,
        default=1.0,
        help="Multiply net_input loss by this weight (default 0)",
    )
    parser.add_argument(
        "--loss_weight_net_output",
        type=float,
        default=1.0,
        help="Multiply net_output loss by this weight (default 0)",
    )
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--inspect_labels", action="store_true")
    parser.add_argument("--inspect_n", type=int, default=5)
    parser.add_argument("--report_label_stats", action="store_true")
    parser.add_argument(
        "--metrics_json",
        type=str,
        default="",
        help="Write per-epoch loss/metric history to this JSON file (default: <out_dir>/metrics.json)",
    )
    parser.add_argument(
        "--metrics_plot",
        type=str,
        default="",
        help="Write loss/metric curves PNG to this path (default: <out_dir>/metrics.png)",
    )
    parser.add_argument(
        "--metrics_plot_by_type",
        type=str,
        default="",
        help="Write per-node-type loss/metric curves PNG (default: <out_dir>/metrics_by_type.png)",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Disable matplotlib plotting even if available",
    )

    parser.add_argument(
        "--log_metric_max_dims",
        type=int,
        default=16,
        help="When printing per-dim metrics, show only first K dims (0 shows all)",
    )

    parser.add_argument(
        "--plot_r2_per_dim",
        action="store_true",
        help="If set, write a per-dimension R2 curve plot for every node type",
    )
    parser.add_argument(
        "--plot_r2_per_dim_max_dims",
        type=int,
        default=16,
        help="Max dims to plot in per-dimension R2 curve (-1 for all)",
    )

    parser.add_argument(
        "--eval_metric",
        type=str,
        default="r2",
        choices=["rmse", "r2"],
        help="Evaluation metric to report for train/val/test (default: r2)",
    )
    args = parser.parse_args()

    cell_input_group_weights = _parse_float_list(
        args.cell_input_group_weights, expected_len=5, name="--cell_input_group_weights"
    )

    set_seed(args.seed)

    dataset = JsonGraphDataset(
        args.data,
        label_norm=args.label_norm,
        label_log=args.label_log,
        feat_norm=args.feat_norm,
        outlier_sigma=0,
        filter_outlier_features=False,
        filter_outlier_labels=False,
        validate_graphs=False,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No graphs loaded from {args.data}")

    if args.inspect_labels:
        inspect_net_output_labels(dataset, args.inspect_n)

    summary = summarize_graphs(dataset)
    print(f"Loaded graphs: {int(summary['num_graphs'])}, avg nodes per graph: {summary['avg_nodes']:.2f}")

    out_dims = infer_out_dims(dataset)
    check_out_dims(dataset, out_dims)
    # print(out_dims)
    train_ids, val_ids, test_ids = split_indices(
        len(dataset),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_loader = DataLoader(Subset(dataset, train_ids), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_ids), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_ids), batch_size=args.batch_size, shuffle=False)

    sample = dataset[0]
    if args.report_label_stats:
        report_label_stats(dataset, tuple(sample.node_types))
    model = HeteroGNN(
        sample.metadata(),
        args.hidden,
        args.layers,
        out_dims,
        conv_type=args.conv,
        gat_heads=args.gat_heads,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.eta_min)
    os.makedirs(args.out_dir, exist_ok=True)

    metrics: List[EpochMetrics] = []

    start_epoch = 1

    ckpt_dir = args.ckpt_dir if args.ckpt_dir else os.path.join(args.out_dir, "checkpoints")
    if args.resume:
        resumed_epoch = _load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location="cpu" if args.device.startswith("cuda") else args.device,
        )
        start_epoch = int(resumed_epoch) + 1
        print(f"Resumed from checkpoint: {args.resume} (epoch {resumed_epoch})")
    
    # Optional baseline evaluation at epoch 0 (kept as print only).
    regression_plot_dir = os.path.join(args.out_dir, "regression") if not args.no_plot else ""

    val_m0, val_m0_per_dim = eval_epoch(
        model,
        val_loader,
        args.device,
        "val",
        dataset,
        metric=args.eval_metric,
        epoch=0,
        regression_plot_dir=regression_plot_dir,
    )
    train_m0, train_m0_per_dim = eval_epoch(
        model,
        train_loader,
        args.device,
        "train",
        dataset,
        metric=args.eval_metric,
        epoch=0,
        regression_plot_dir=regression_plot_dir,
    )
    test_m0, test_m0_per_dim = eval_epoch(
        model,
        test_loader,
        args.device,
        "test",
        dataset,
        metric=args.eval_metric,
        epoch=0,
        regression_plot_dir=regression_plot_dir,
    )
    msg0 = "Epoch 0"
    if args.eval_metric == "r2" and val_m0_per_dim:
        msg0 += f" | val_{args.eval_metric} " + format_per_dim_metrics(
            val_m0_per_dim,
            max_dims=int(args.log_metric_max_dims),
        )
    elif val_m0:
        msg0 += f" | val_{args.eval_metric} " + ", ".join([f"{k}:{v:.4f}" for k, v in val_m0.items()])

    if args.eval_metric == "r2" and train_m0_per_dim:
        msg0 += f" | train_{args.eval_metric} " + format_per_dim_metrics(
            train_m0_per_dim,
            max_dims=int(args.log_metric_max_dims),
        )
    elif train_m0:
        msg0 += f" | train_{args.eval_metric} " + ", ".join([f"{k}:{v:.4f}" for k, v in train_m0.items()])

    if args.eval_metric == "r2" and test_m0_per_dim:
        print(
            f"Test {args.eval_metric.upper()}: "
            + format_per_dim_metrics(test_m0_per_dim, max_dims=int(args.log_metric_max_dims))
        )
    elif test_m0:
        print(
            f"Test {args.eval_metric.upper()}: "
            + ", ".join([f"{k}:{v:.4f}" for k, v in test_m0.items()])
        )

    print(msg0)

    for epoch in range(start_epoch, args.epochs + 1):
        loss, train_loss_by_type = train_one_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            args.log_iter,
            epoch,
            args.loss,
            args.clip_grad,
            cell_input_group_weights=cell_input_group_weights,
            net_input_weight=float(args.loss_weight_net_input),
            net_output_weight=float(args.loss_weight_net_output),
        )
        val_metric, val_metric_per_dim = eval_epoch(
            model,
            val_loader,
            args.device,
            "val",
            dataset,
            metric=args.eval_metric,
            epoch=int(epoch),
            regression_plot_dir= regression_plot_dir if epoch % int(args.plot_epoch) == 0 else "",
        )
        train_metric, train_metric_per_dim = eval_epoch(
            model,
            train_loader,
            args.device,
            "train",
            dataset,
            metric=args.eval_metric,
            epoch=int(epoch),
            regression_plot_dir= regression_plot_dir if epoch % int(args.plot_epoch) == 0 else "",
        )

        metrics.append(
            EpochMetrics(
                epoch=int(epoch),
                train_loss=float(loss),
                metric_name=str(args.eval_metric),
                train_metric_mean=_aggregate_metric(train_metric),
                val_metric_mean=_aggregate_metric(val_metric),
                train_loss_by_type=dict(train_loss_by_type),
                train_metric_by_type=dict(train_metric),
                val_metric_by_type=dict(val_metric),
                train_metric_per_dim_by_type=dict(train_metric_per_dim),
                val_metric_per_dim_by_type=dict(val_metric_per_dim),
            )
        )
        if epoch % max(args.log_every, 1) == 0:
            msg = f"Epoch {epoch:03d} | loss {loss:.6f}"
            if args.eval_metric == "r2" and val_metric_per_dim:
                msg += f" | val_{args.eval_metric} " + format_per_dim_metrics(
                    val_metric_per_dim,
                    max_dims=int(args.log_metric_max_dims),
                )
            elif val_metric:
                m_str = ", ".join([f"{k}:{v:.4f}" for k, v in val_metric.items()])
                msg += f" | val_{args.eval_metric} {m_str}"

            if args.eval_metric == "r2" and train_metric_per_dim:
                msg += f" | train_{args.eval_metric} " + format_per_dim_metrics(
                    train_metric_per_dim,
                    max_dims=int(args.log_metric_max_dims),
                )
            elif train_metric:
                m_str = ", ".join([f"{k}:{v:.4f}" for k, v in train_metric.items()])
                msg += f" | train_{args.eval_metric} {m_str}"
            print(msg)

        # Step scheduler once per epoch (cosine: high -> low)
        scheduler.step()

        if int(args.ckpt_every) > 0 and (epoch % int(args.ckpt_every) == 0):
            ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")
            _save_checkpoint(
                ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                args_dict=vars(args),
            )
            # print(f"Wrote checkpoint: {ckpt_path}")



    test_metric, test_metric_per_dim = eval_epoch(
        model,
        test_loader,
        args.device,
        "test",
        dataset,
        metric=args.eval_metric,
        epoch=int(args.epochs),
        regression_plot_dir=regression_plot_dir,
    )
    if args.eval_metric == "r2" and test_metric_per_dim:
        print(
            f"Test {args.eval_metric.upper()}: "
            + format_per_dim_metrics(test_metric_per_dim, max_dims=int(args.log_metric_max_dims))
        )
    elif test_metric:
        print(f"Test {args.eval_metric.upper()}:", ", ".join([f"{k}:{v:.4f}" for k, v in test_metric.items()]))

    # Write metrics and plot curves.
    metrics_json = args.metrics_json if args.metrics_json else os.path.join(args.out_dir, "metrics.json")
    _write_metrics_json(metrics_json, metrics, vars(args))
    print(f"Wrote metrics JSON: {metrics_json}")

    if not args.no_plot:
        metrics_plot = args.metrics_plot if args.metrics_plot else os.path.join(args.out_dir, "metrics.png")
        ok = _plot_metrics(metrics_plot, metrics)
        if ok:
            print(f"Wrote metrics plot: {metrics_plot}")
        else:
            print("Matplotlib not available; skipped metrics plot (install matplotlib or pass --no_plot)")

        metrics_plot_by_type = (
            args.metrics_plot_by_type
            if args.metrics_plot_by_type
            else os.path.join(args.out_dir, "metrics_by_type.png")
        )
        ok2 = _plot_metrics_by_type(metrics_plot_by_type, metrics)
        if ok2:
            print(f"Wrote per-type metrics plot: {metrics_plot_by_type}")
        else:
            print("Matplotlib not available; skipped per-type plot (install matplotlib or pass --no_plot)")

        if args.eval_metric == "r2" and args.plot_r2_per_dim:
            for ntype in test_metric.keys():
                per_dim_plot = os.path.join(
                    args.out_dir,
                    f"r2_per_dim_{ntype}.png",
                )
                ok3 = _plot_r2_per_dim_trainval(
                    per_dim_plot,
                    metrics,
                    node_type=ntype,
                    max_dims=int(args.plot_r2_per_dim_max_dims),
                )
                if ok3:
                    print(f"Wrote per-dim R2 plot: {per_dim_plot}")
                else:
                    print("Skipped per-dim R2 plot (no data or matplotlib missing)")


if __name__ == "__main__":
    main()
