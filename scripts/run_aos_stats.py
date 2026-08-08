"""scripts/run_aos_stats.py -- operator-selection ablation statistics.

Loads an aos_ablation.json artifact and produces the significance + effect-size
analysis: Friedman omnibus, Holm-corrected pairwise Wilcoxon, matched-pairs rank-biserial
effect size, and bootstrap CIs of the paired median difference. Writes aos_stats.json
next to the input and prints a publication-style table.

Usage::

    python scripts/run_aos_stats.py                                   # default input
    python scripts/run_aos_stats.py --input experiments/aos_n10/aos_ablation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.table import Table

from ehgat.benchmark.stats import AblationStats, analyse_ablation, to_json_dict

console = Console()
_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = _ROOT / "experiments" / "aos_n10" / "aos_ablation.json"


def _sig(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _print(stats: AblationStats) -> None:
    console.rule("[bold white]Operator-selection ablation statistics")
    console.print(f"  Arms: {', '.join(stats.arms)}   Seeds (paired): {stats.n_seeds}")

    tbl = Table(title="Friedman omnibus + arm means", header_style="bold cyan")
    tbl.add_column("Metric", style="bold")
    tbl.add_column("dir")
    for arm in stats.arms:
        tbl.add_column(arm, justify="right")
    tbl.add_column("Friedman p", justify="right")
    for name, m in stats.metrics.items():
        row = [name, "up" if m.higher_is_better else "down"]
        row += [f"{m.arm_means.get(a, float('nan')):.4f}" for a in stats.arms]
        row.append(f"{m.friedman.pvalue:.2e} {_sig(m.friedman.pvalue)}")
        tbl.add_row(*row)
    console.print(tbl)

    ptbl = Table(
        title="Post-hoc pairwise Wilcoxon (Holm-corrected) + rank-biserial effect size",
        header_style="bold cyan",
    )
    ptbl.add_column("Metric", style="bold")
    ptbl.add_column("Pair (a - b)")
    ptbl.add_column("median diff [95% CI]", justify="right")
    ptbl.add_column("rank-biserial", justify="right")
    ptbl.add_column("Holm p", justify="right")
    for name, m in stats.metrics.items():
        for w in m.pairwise:
            ptbl.add_row(
                name,
                f"{w.pair[0]} - {w.pair[1]}",
                f"{w.median_diff:+.4f} [{w.ci_lo:+.4f}, {w.ci_hi:+.4f}]",
                f"{w.rank_biserial:+.3f}",
                f"{w.pvalue_holm:.2e} {_sig(w.pvalue_holm)}",
            )
    console.print(ptbl)

    # Headline read-out for HV.
    hv = stats.metrics.get("final_hv")
    if hv is not None:
        att = next((w for w in hv.pairwise if w.pair == ("attention", "random")), None)
        orc = next((w for w in hv.pairwise if w.pair == ("attention", "oracle")), None)
        if att is not None:
            verdict = "PASS" if (att.median_diff > 0 and att.pvalue_holm < 0.05) else "INCONCLUSIVE"
            msg = (
                f"\n[bold green]Ablation (HV):[/bold green] attention > random "
                f"median +{att.median_diff:.4f} (Holm p={att.pvalue_holm:.2e}, "
                f"r={att.rank_biserial:+.2f}) -> [bold]{verdict}[/bold]"
            )
            if orc is not None:
                msg += (
                    f"; attention vs oracle median {orc.median_diff:+.4f} "
                    f"(Holm p={orc.pvalue_holm:.2e})"
                )
            console.print(msg)


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Operator-selection ablation statistics")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument(
        "--out", type=str, default=None, help="default: aos_stats.json next to input"
    )
    parser.add_argument("--resamples", type=int, default=10000, help="bootstrap resamples")
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    ns = parser.parse_args(args)

    in_path = Path(ns.input)
    if not in_path.exists():
        console.print(f"[bold red]Input not found:[/bold red] {in_path}")
        raise SystemExit(1)

    stats = analyse_ablation(in_path, resamples=ns.resamples, ci=ns.ci, seed=ns.seed)
    _print(stats)

    out_path = Path(ns.out) if ns.out else in_path.with_name("aos_stats.json")
    out_path.write_text(json.dumps(to_json_dict(stats), indent=2))
    console.print(f"\n[bold]Saved:[/bold] {out_path}")


if __name__ == "__main__":
    main()
