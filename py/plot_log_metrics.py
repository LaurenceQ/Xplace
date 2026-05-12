#!/usr/bin/env python3

import argparse
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class EpochRow:
    epoch: int
    loss: float  # may be NaN for epoch 0 if not present
    metric_name: str
    train_metric: Dict[str, float]
    val_metric: Dict[str, float]


_EPOCH_WITH_LOSS_RE = re.compile(
    r"^Epoch\s+(?P<epoch>\d+)\s*\|\s*loss\s+(?P<loss>[+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?)\s*\|\s*(?P<val_name>val_(?:rmse|r2))\s+(?P<val>.+?)\s*\|\s*(?P<train_name>train_(?:rmse|r2))\s+(?P<train>.+?)\s*$"
)
_EPOCH_HEADER_RE = re.compile(r"^Epoch\s+(?P<epoch>\d+)\s*$")


def _try_import_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:  # noqa: BLE001
        return None


def _parse_kv_list(s: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        v = v.strip()
        try:
            out[k] = float(v)
        except ValueError:
            continue
    return out


def parse_log(path: str) -> List[EpochRow]:
    rows: List[EpochRow] = []

    pending_epoch: Optional[int] = None
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip("\n")
            stripped = line.strip()
            if not stripped:
                continue

            # Handle the special 2-line "Epoch 0" block:
            #   Epoch 0
            #    | val_rmse ... | train_rmse ...
            # or | val_r2 ... | train_r2 ...
            if pending_epoch is not None and stripped.startswith("|"):
                # Normalize "| val_rmse ... | train_rmse ..."
                block = stripped.lstrip("|").strip()
                # Expect "val_<metric> ... | train_<metric> ..."
                if "|" in block and "val_" in block and "train_" in block:
                    left, right = [p.strip() for p in block.split("|", 1)]
                    if left.startswith("val_") and right.startswith("train_"):
                        val_name = left.split(None, 1)[0]
                        train_name = right.split(None, 1)[0]
                        val_metric_name = val_name.split("_", 1)[1]
                        train_metric_name = train_name.split("_", 1)[1]
                        if val_metric_name == train_metric_name:
                            val = _parse_kv_list(left[len(val_name) :].strip())
                            train = _parse_kv_list(right[len(train_name) :].strip())
                            rows.append(
                                EpochRow(
                                    epoch=int(pending_epoch),
                                    loss=float("nan"),
                                    metric_name=val_metric_name,
                                    train_metric=train,
                                    val_metric=val,
                                )
                            )
                pending_epoch = None
                continue

            m = _EPOCH_WITH_LOSS_RE.match(stripped)
            if m:
                epoch = int(m.group("epoch"))
                loss = float(m.group("loss"))
                val_name = m.group("val_name")
                train_name = m.group("train_name")
                metric_name = val_name.split("_", 1)[1]
                if train_name.split("_", 1)[1] != metric_name:
                    continue
                val = _parse_kv_list(m.group("val"))
                train = _parse_kv_list(m.group("train"))
                rows.append(
                    EpochRow(epoch=epoch, loss=loss, metric_name=metric_name, train_metric=train, val_metric=val)
                )
                pending_epoch = None
                continue

            # Epoch header without loss (epoch 0)
            mh = _EPOCH_HEADER_RE.match(stripped)
            if mh:
                pending_epoch = int(mh.group("epoch"))
                continue

            # Ignore iter-level lines like: Epoch 001 | iter 000100 | loss 4.522986
            # Ignore Test RMSE lines

    # Ensure sorted and unique by epoch (keep last occurrence)
    by_epoch: Dict[int, EpochRow] = {}
    for r in rows:
        by_epoch[r.epoch] = r
    return [by_epoch[k] for k in sorted(by_epoch.keys())]


def plot_rows(rows: List[EpochRow], *, out_png: str, title: str = "") -> None:
    plt = _try_import_matplotlib_pyplot()
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting (pip install matplotlib)")

    epochs = [r.epoch for r in rows]
    losses = [r.loss for r in rows]

    metric_name = rows[0].metric_name if rows else "metric"
    metric_label = "R2" if metric_name == "r2" else ("RMSE" if metric_name == "rmse" else metric_name)

    # Collect rmse types
    types = set()
    for r in rows:
        types.update(r.train_metric.keys())
        types.update(r.val_metric.keys())
    types_sorted = sorted(types)

    os.makedirs(os.path.dirname(os.path.abspath(out_png)) or ".", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # Loss curve (skip NaNs by plotting as gaps)
    ax1.plot(epochs, losses, label="train_loss", color="tab:blue")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")
    if title:
        ax1.set_title(title)

    # Per-type metric (train solid, val dashed)
    for t in types_sorted:
        ytr = [r.train_metric.get(t, float("nan")) for r in rows]
        yva = [r.val_metric.get(t, float("nan")) for r in rows]
        ax2.plot(epochs, ytr, label=f"train_{metric_name}[{t}]", linestyle="-")
        ax2.plot(epochs, yva, label=f"val_{metric_name}[{t}]", linestyle="--")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(metric_label)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse training log and plot epoch loss + train/val metric by type (RMSE or R2)")
    parser.add_argument("--log", type=str, required=True, help="Path to log file (e.g. checkpoints/aes.log)")
    parser.add_argument("--out", type=str, default="", help="Output PNG path (default: <log>.png)")
    parser.add_argument("--title", type=str, default="", help="Optional plot title")
    args = parser.parse_args()

    out_png = args.out if args.out else f"{args.log}.png"

    rows = parse_log(args.log)
    if not rows:
        raise RuntimeError(f"No epoch summaries parsed from {args.log}")

    plot_rows(rows, out_png=out_png, title=args.title)
    print(f"Parsed epochs: {len(rows)} (min={rows[0].epoch}, max={rows[-1].epoch})")
    print(f"Wrote plot: {out_png}")


if __name__ == "__main__":
    main()
