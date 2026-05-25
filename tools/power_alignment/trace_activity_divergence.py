#!/usr/bin/env python3
"""Trace activity divergence upstream through gate-level Verilog connectivity.

The script is intentionally evidence-only: it reads an existing
activity_snapshot_compare.csv and gate-level Verilog, then walks from a target
pin to its driver cone at one snapshot point. It does not run OpenROAD/Xplace
and does not change propagation behavior.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


OUTPUT_PORT_HINTS = {
    "Z",
    "ZN",
    "Q",
    "QN",
    "CO",
    "S",
    "SO",
    "Y",
}


@dataclass
class Instance:
    cell_type: str
    name: str
    conns: dict[str, str]


@dataclass
class Module:
    name: str
    ports: dict[str, str] = field(default_factory=dict)
    instances: list[Instance] = field(default_factory=list)
    inst_by_name: dict[str, Instance] = field(default_factory=dict)

    def add_instance(self, inst: Instance) -> None:
        self.instances.append(inst)
        self.inst_by_name[inst.name] = inst


@dataclass
class PinRow:
    pin: str
    status: str
    debug_status: str
    or_density: float
    x_density: float
    density_abs: float
    density_rel: float
    or_duty: float
    x_duty: float
    duty_abs: float
    or_origin: str
    x_origin: str
    or_is_driver: bool
    x_is_driver: bool
    x_is_load: bool

    @property
    def mismatch(self) -> bool:
        return self.status != "MATCH"

    @property
    def follow_mismatch(self) -> bool:
        return self.debug_status == "DEBUG_MISMATCH"

    @property
    def origin_diff(self) -> bool:
        return self.or_origin != self.x_origin

    @property
    def propagated_origin_diff(self) -> bool:
        return self.origin_diff and self.or_origin != "8"


@dataclass
class Context:
    prefix: tuple[str, ...]
    module: Module


@dataclass
class PinLocation:
    context: Context
    inst: Instance
    port: str
    net: str
    pin_path: str


@dataclass
class DriverResult:
    pin_path: str
    inst: Instance | None
    context: Context
    port: str
    net: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verilog", type=Path, required=True)
    parser.add_argument("--compare", type=Path, required=True)
    parser.add_argument("--first-divergence", type=Path)
    parser.add_argument("--pin", required=True, help="Target pin, e.g. a/b/c/A")
    parser.add_argument("--pass-id", type=int, default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--use-first-divergence", action="store_true")
    parser.add_argument("--top", default=None)
    parser.add_argument("--max-depth", type=int, default=40)
    parser.add_argument("--max-branches", type=int, default=3)
    parser.add_argument(
        "--follow-origin-diff",
        action="store_true",
        help="Also follow origin-only differences when density/duty match.",
    )
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def norm_name(value: str) -> str:
    text = (value or "").strip().strip('"')
    if text.startswith("\\"):
        text = text[1:]
    text = text.replace(r"\[", "[").replace(r"\]", "]")
    text = text.replace("\\", "")
    return text.replace(":", "/").strip()


def norm_net(value: str) -> str:
    text = (value or "").strip()
    if text.startswith("{") or "," in text or text.startswith("1'b"):
        return text
    return norm_name(text)


def to_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def to_bool(value: str) -> bool:
    return str(value).strip() in {"1", "true", "True"}


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return path.open("r", errors="replace")


def strip_line_comment(line: str) -> str:
    return line.split("//", 1)[0]


def split_decl_names(rest: str) -> Iterable[str]:
    rest = rest.replace(";", " ")
    rest = re.sub(r"\[[^]]+\]", " ", rest)
    rest = re.sub(r"\b(?:wire|reg|logic|signed)\b", " ", rest)
    for item in rest.split(","):
        name = norm_name(item)
        if name:
            yield name


def parse_named_conns(text: str) -> dict[str, str]:
    conns: dict[str, str] = {}
    i = 0
    while i < len(text):
        dot = text.find(".", i)
        if dot < 0:
            break
        j = dot + 1
        while j < len(text) and (text[j].isalnum() or text[j] in "_$\\[]"):
            j += 1
        port = norm_name(text[dot + 1 : j])
        while j < len(text) and text[j].isspace():
            j += 1
        if j >= len(text) or text[j] != "(":
            i = j + 1
            continue
        depth = 1
        k = j + 1
        while k < len(text) and depth:
            if text[k] == "(":
                depth += 1
            elif text[k] == ")":
                depth -= 1
            k += 1
        if port and depth == 0:
            conns[port] = norm_net(text[j + 1 : k - 1])
        i = k
    return conns


def parse_instance(stmt: str) -> Instance | None:
    stmt = " ".join(stmt.split())
    if not stmt or "." not in stmt:
        return None
    if re.match(r"^(?:input|output|inout|wire|assign|module)\b", stmt):
        return None
    match = re.match(r"^(\S+)\s+(\S+)\s*\((.*)\)\s*;$", stmt)
    if not match:
        return None
    cell_type, name, body = match.groups()
    if cell_type in {"if", "for", "always", "assign"}:
        return None
    conns = parse_named_conns(body)
    if not conns:
        return None
    return Instance(norm_name(cell_type), norm_name(name), conns)


def parse_verilog(path: Path) -> dict[str, Module]:
    modules: dict[str, Module] = {}
    current: Module | None = None
    in_header = False
    stmt_parts: list[str] = []

    def flush_statement(stmt: str) -> None:
        nonlocal current
        if current is None:
            return
        stripped = stmt.strip()
        decl = re.match(r"^(input|output|inout)\b(.*);$", stripped)
        if decl:
            direction = decl.group(1)
            for name in split_decl_names(decl.group(2)):
                current.ports[name] = direction
            return
        inst = parse_instance(stripped)
        if inst:
            current.add_instance(inst)

    with open_text(path) as f:
        for raw_line in f:
            line = strip_line_comment(raw_line).strip()
            if not line:
                continue
            mod_match = re.match(r"^module\s+(\S+)\b", line)
            if mod_match:
                current = Module(norm_name(mod_match.group(1)))
                modules[current.name] = current
                in_header = True
                if ";" in line:
                    in_header = False
                continue
            if current is None:
                continue
            if line.startswith("endmodule"):
                current = None
                in_header = False
                stmt_parts = []
                continue
            if in_header:
                if ";" in line:
                    in_header = False
                continue
            stmt_parts.append(line)
            if ";" in line:
                flush_statement(" ".join(stmt_parts))
                stmt_parts = []
    return modules


def infer_top(modules: dict[str, Module], requested: str | None) -> Module:
    if requested:
        if requested not in modules:
            raise SystemExit(f"top module not found: {requested}")
        return modules[requested]
    instantiated = {inst.cell_type for module in modules.values() for inst in module.instances}
    candidates = [name for name in modules if name not in instantiated]
    if len(candidates) == 1:
        return modules[candidates[0]]
    if "ariane" in modules:
        return modules["ariane"]
    raise SystemExit(f"cannot infer top module; candidates={candidates[:10]}")


def load_snapshot_rows(compare_csv: Path, pass_id: int, tag: str) -> dict[str, PinRow]:
    rows: dict[str, PinRow] = {}
    with compare_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["pass"]) != pass_id or row["tag"] != tag:
                continue
            pin = norm_name(row["pin_name_norm"] or row["pin_name"])
            rows[pin] = PinRow(
                pin=pin,
                status=row["status"],
                debug_status=row["debug_status"],
                or_density=to_float(row["or_density"]),
                x_density=to_float(row["x_density"]),
                density_abs=to_float(row["density_abs"]),
                density_rel=to_float(row["density_rel"]),
                or_duty=to_float(row["or_duty"]),
                x_duty=to_float(row["x_duty"]),
                duty_abs=to_float(row["duty_abs"]),
                or_origin=row["or_origin"],
                x_origin=row["x_origin"],
                or_is_driver=to_bool(row.get("or_is_driver", "0")),
                x_is_driver=to_bool(row.get("x_is_driver", "0")),
                x_is_load=to_bool(row.get("x_is_load", "0")),
            )
    return rows


def read_first_divergence(path: Path, pin: str) -> tuple[int, str] | None:
    target = norm_name(pin)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if norm_name(row["pin_name_norm"]) == target:
                return int(row["pass"]), row["tag"]
    return None


def pin_path(prefix: tuple[str, ...], inst: str, port: str) -> str:
    parts = [*prefix, inst, port]
    return "/".join(part for part in parts if part)


def resolve_context(
    modules: dict[str, Module], top: Module, prefix: tuple[str, ...]
) -> Context:
    module = top
    for name in prefix:
        inst = module.inst_by_name.get(name)
        if inst is None:
            raise KeyError(f"instance not found in {'/'.join(prefix)}: {name}")
        child = modules.get(inst.cell_type)
        if child is None:
            raise KeyError(f"instance is leaf before path ends: {name}:{inst.cell_type}")
        module = child
    return Context(prefix, module)


def resolve_pin(
    modules: dict[str, Module], top: Module, pin: str
) -> PinLocation:
    parts = tuple(norm_name(pin).split("/"))
    if len(parts) < 2:
        raise KeyError(f"pin path must include instance and port: {pin}")
    prefix = parts[:-2]
    inst_name = parts[-2]
    port = parts[-1]
    context = resolve_context(modules, top, prefix)
    inst = context.module.inst_by_name.get(inst_name)
    if inst is None:
        raise KeyError(f"instance not found: {'/'.join(parts[:-1])}")
    net = inst.conns.get(port)
    if net is None:
        raise KeyError(f"port not connected: {pin}")
    return PinLocation(context, inst, port, net, "/".join(parts))


def port_direction(
    modules: dict[str, Module],
    rows: dict[str, PinRow],
    context: Context,
    inst: Instance,
    port: str,
) -> str:
    child = modules.get(inst.cell_type)
    if child and port in child.ports:
        return child.ports[port]
    row = rows.get(pin_path(context.prefix, inst.name, port))
    if row and (row.or_is_driver or row.x_is_driver):
        return "output"
    if row and row.x_is_load:
        return "input"
    return "output" if port in OUTPUT_PORT_HINTS else "input"


def parent_context(
    modules: dict[str, Module], top: Module, context: Context
) -> tuple[Context, Instance] | None:
    if not context.prefix:
        return None
    parent_prefix = context.prefix[:-1]
    inst_name = context.prefix[-1]
    parent = resolve_context(modules, top, parent_prefix)
    inst = parent.module.inst_by_name.get(inst_name)
    if inst is None:
        return None
    return parent, inst


def find_driver(
    modules: dict[str, Module],
    top: Module,
    rows: dict[str, PinRow],
    context: Context,
    net: str,
    seen: set[tuple[tuple[str, ...], str]] | None = None,
) -> DriverResult | None:
    if seen is None:
        seen = set()
    key = (context.prefix, net)
    if key in seen:
        return None
    seen.add(key)

    for inst in context.module.instances:
        for port, conn in inst.conns.items():
            if conn != net:
                continue
            direction = port_direction(modules, rows, context, inst, port)
            if direction != "output":
                continue
            child = modules.get(inst.cell_type)
            if child:
                child_context = Context((*context.prefix, inst.name), child)
                internal = find_driver(modules, top, rows, child_context, port, seen)
                if internal:
                    return internal
            return DriverResult(
                pin_path=pin_path(context.prefix, inst.name, port),
                inst=inst,
                context=context,
                port=port,
                net=net,
                reason="local_instance_output",
            )

    direction = context.module.ports.get(net)
    if direction == "input":
        parent = parent_context(modules, top, context)
        if parent is None:
            return None
        parent_ctx, parent_inst = parent
        parent_net = parent_inst.conns.get(net)
        if not parent_net:
            return None
        return find_driver(modules, top, rows, parent_ctx, parent_net, seen)
    return None


def format_row(row: PinRow | None) -> str:
    if row is None:
        return "NO_ROW"
    return (
        f"{row.status}/{row.debug_status} OR(d={row.or_duty:g},rho={row.or_density:g},o={row.or_origin}) "
        f"X(d={row.x_duty:g},rho={row.x_density:g},o={row.x_origin}) "
        f"diff(d={row.duty_abs:g},rho={row.density_abs:g})"
    )


def instance_input_pins(
    modules: dict[str, Module],
    rows: dict[str, PinRow],
    driver: DriverResult,
) -> list[str]:
    inst = driver.inst
    if inst is None:
        return []
    pins = []
    for port in inst.conns:
        if port == driver.port:
            continue
        if port_direction(modules, rows, driver.context, inst, port) == "input":
            pins.append(pin_path(driver.context.prefix, inst.name, port))
    return pins


def trace(
    modules: dict[str, Module],
    top: Module,
    rows: dict[str, PinRow],
    pin: str,
    depth: int,
    max_depth: int,
    max_branches: int,
    follow_origin_diff: bool,
    visited: set[str],
    out_lines: list[str],
) -> None:
    pin = norm_name(pin)
    indent = "  " * depth
    row = rows.get(pin)
    out_lines.append(f"{indent}PIN {pin}: {format_row(row)}")
    if depth >= max_depth:
        out_lines.append(f"{indent}STOP max_depth")
        return
    if pin in visited:
        out_lines.append(f"{indent}STOP loop")
        return
    visited.add(pin)

    try:
        loc = resolve_pin(modules, top, pin)
    except KeyError as exc:
        out_lines.append(f"{indent}STOP unresolved_pin: {exc}")
        return
    driver = find_driver(modules, top, rows, loc.context, loc.net)
    if driver is None:
        out_lines.append(f"{indent}STOP no_driver net={loc.net} context={'/'.join(loc.context.prefix)}")
        return

    driver_row = rows.get(driver.pin_path)
    out_lines.append(
        f"{indent}DRIVER {driver.pin_path} via net={loc.net} reason={driver.reason}: "
        f"{format_row(driver_row)}"
    )

    inputs = instance_input_pins(modules, rows, driver)
    mismatched_inputs: list[tuple[float, str]] = []
    if inputs:
        out_lines.append(f"{indent}INPUTS of {'/'.join((*driver.context.prefix, driver.inst.name if driver.inst else ''))}:")
    for input_pin in inputs:
        input_row = rows.get(input_pin)
        out_lines.append(f"{indent}  {input_pin}: {format_row(input_row)}")
        if input_row and (input_row.follow_mismatch or (follow_origin_diff and input_row.propagated_origin_diff)):
            score = input_row.duty_abs + input_row.density_abs
            if input_row.propagated_origin_diff:
                score += 1.0e-6
            mismatched_inputs.append((score, input_pin))

    driver_diff = bool(driver_row and (driver_row.follow_mismatch or (follow_origin_diff and driver_row.propagated_origin_diff)))
    if driver_diff and inputs and not mismatched_inputs:
        out_lines.append(f"{indent}FOUND cell_output_divergence_with_inputs_aligned")
        return
    if not mismatched_inputs:
        if driver_row and not driver_row.mismatch:
            out_lines.append(f"{indent}STOP driver_aligned")
        else:
            out_lines.append(f"{indent}STOP no_mismatched_input_to_continue")
        return

    mismatched_inputs.sort(reverse=True)
    for _, next_pin in mismatched_inputs[:max_branches]:
        out_lines.append(f"{indent}FOLLOW {next_pin}")
        trace(
            modules,
            top,
            rows,
            next_pin,
            depth + 1,
            max_depth,
            max_branches,
            follow_origin_diff,
            visited,
            out_lines,
        )
    if len(mismatched_inputs) > max_branches:
        out_lines.append(f"{indent}SKIP {len(mismatched_inputs) - max_branches} extra mismatched branches")


def main() -> None:
    args = parse_args()
    pin = norm_name(args.pin)
    pass_id = args.pass_id
    tag = args.tag
    if args.use_first_divergence:
        if not args.first_divergence:
            raise SystemExit("--use-first-divergence requires --first-divergence")
        first = read_first_divergence(args.first_divergence, pin)
        if first is None:
            raise SystemExit(f"pin not found in first divergence CSV: {pin}")
        pass_id, tag = first
    if pass_id is None or tag is None:
        raise SystemExit("provide --pass-id/--tag or --use-first-divergence")

    modules = parse_verilog(args.verilog)
    top = infer_top(modules, args.top)
    rows = load_snapshot_rows(args.compare, pass_id, tag)
    out_lines = [
        f"TRACE pin={pin} snapshot=pass{pass_id}/{tag} top={top.name}",
        f"loaded_modules={len(modules)} loaded_snapshot_rows={len(rows)}",
    ]
    trace(
        modules=modules,
        top=top,
        rows=rows,
        pin=pin,
        depth=0,
        max_depth=args.max_depth,
        max_branches=args.max_branches,
        follow_origin_diff=args.follow_origin_diff,
        visited=set(),
        out_lines=out_lines,
    )
    text = "\n".join(out_lines) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
