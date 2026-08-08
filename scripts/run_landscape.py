"""scripts/run_landscape.py -- feature-importance landscape entrypoint.

Runs the exact, cascade-aware landscape analysis on one or more instance sizes and writes a
structured JSON artifact per size::

    experiments/landscape/landscape_n{N}.json

The analysis is computed on the exact Max-Plus evaluator (the structural causal model),
not a fitted surrogate:

- grouped Sobol' indices (first + total order) over the four decision families,
- exact critical-path cascade attribution (AGV/QC binding, marginal effects, cascade size),
- Pareto-vs-dominated contrast (Cliff's delta of decision descriptors),
- optional TreeSHAP-vs-Sobol failure-boundary comparison (--shap).

Usage::

    python scripts/run_landscape.py                          # quick (N=6)
    python scripts/run_landscape.py --tasks 10 20 50 \\
        --sobol-base 2048 --cascade-samples 512 --contrast-samples 1024 --shap
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.table import Table

from ehgat.benchmark.landscape import (
    FEATURE_FAMILIES,
    OBJECTIVES,
    run_landscape,
    tabular_failure_boundary,
    to_json_dict,
)
from ehgat.environment.instance import build_toy_instance

console = Console()
OUT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "landscape"


def _print_summary(payload: dict[str, object]) -> None:
    inst = payload["instance"]
    console.rule(f"[bold white]Landscape -- N={inst['num_tasks']} A={inst['num_agvs']} QC={inst['num_qcs']}")  # type: ignore[index]

    sobol = payload["sobol"]
    tbl = Table(title="Grouped Sobol' indices (exact evaluator)", header_style="bold cyan")
    tbl.add_column("Objective", style="bold")
    tbl.add_column("Family")
    tbl.add_column("S_i (first)", justify="right")
    tbl.add_column("ST_i (total)", justify="right")
    for obj in OBJECTIVES:
        for fam in FEATURE_FAMILIES:
            idx = sobol["indices"][obj][fam]  # type: ignore[index]
            tbl.add_row(obj, fam, f"{idx['first_order']:.3f}", f"{idx['total_order']:.3f}")
    console.print(tbl)

    casc = payload["cascade"]
    console.print(
        f"[bold]Critical path:[/bold] AGV-bound share = "
        f"[bold]{casc['agv_share']:.1%}[/bold] of critical mass "  # type: ignore[index]
        f"({casc['num_samples']} samples)"  # type: ignore[index]
    )

    contrast = payload["pareto_contrast"]
    ct = Table(title="Pareto vs dominated (Cliff's delta)", header_style="bold magenta")
    ct.add_column("Descriptor", style="bold")
    ct.add_column("Pareto mean", justify="right")
    ct.add_column("Dominated mean", justify="right")
    ct.add_column("Cliff's d", justify="right")
    for name in contrast["descriptors"]:  # type: ignore[index]
        ct.add_row(
            name,
            f"{contrast['pareto_mean'][name]:.3f}",  # type: ignore[index]
            f"{contrast['dominated_mean'][name]:.3f}",  # type: ignore[index]
            f"{contrast['cliffs_delta'][name]:+.3f}",  # type: ignore[index]
        )
    console.print(
        f"  front size = {contrast['num_pareto']}/{contrast['num_samples']}"  # type: ignore[index]
    )
    console.print(ct)

    if "tabular_boundary" in payload:
        boundary = payload["tabular_boundary"]["boundary"]  # type: ignore[index]
        console.print(
            "[bold yellow]Exact Sobol' vs TreeSHAP topological importance:[/bold yellow]"
        )
        for obj in OBJECTIVES:
            b = boundary[obj]
            console.print(
                f"  {obj}: structural mass  Sobol=[bold]{b['sobol_structural_mass']:.2f}[/bold] "
                f"vs TreeSHAP={b['shap_structural_mass']:.2f}  "
                f"(underweight [bold]{b['structural_underweight']:+.2f}[/bold], "
                f"TV={b['total_variation']:.2f})"
            )


def _run_one(tasks: int, ns: argparse.Namespace, out: Path) -> None:
    instance = build_toy_instance(num_tasks=tasks)
    result = run_landscape(
        instance,
        sobol_base_samples=ns.sobol_base,
        cascade_samples=ns.cascade_samples,
        contrast_samples=ns.contrast_samples,
        seed=ns.seed,
    )
    payload = to_json_dict(result)
    if ns.shap:
        console.print(f"  [dim]training TreeSHAP foil for N={tasks}...[/dim]")
        payload["tabular_boundary"] = tabular_failure_boundary(
            instance, result.sobol, num_samples=ns.shap_samples, seed=ns.seed
        )

    _print_summary(payload)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"landscape_n{tasks}.json"
    path.write_text(json.dumps(payload, indent=2))
    console.print(f"\n[bold]Saved:[/bold] {path}\n")


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="E-HGATv2 landscape analysis")
    parser.add_argument("--tasks", type=int, nargs="+", default=[6], help="instance size(s) N")
    parser.add_argument("--sobol-base", type=int, default=1024, help="Sobol' base samples")
    parser.add_argument("--cascade-samples", type=int, default=256, help="critical-path samples")
    parser.add_argument("--contrast-samples", type=int, default=512, help="Pareto contrast samples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shap", action="store_true", help="add the TreeSHAP failure-boundary foil")
    parser.add_argument("--shap-samples", type=int, default=2000, help="XGB/SHAP training samples")
    parser.add_argument("--out", type=str, default=str(OUT_DIR))
    ns = parser.parse_args(args)

    out = Path(ns.out)
    for tasks in ns.tasks:
        _run_one(tasks, ns, out)


if __name__ == "__main__":
    main()
