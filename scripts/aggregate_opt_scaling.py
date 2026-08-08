"""Aggregate per-N tape_bench_* JSONs into an optimization-scaling table.

Reads every tape_bench_*.json in a directory (each produced by
run_tape_guided_bench.py at one N) and emits, per method, HV/HV* and IGD+ as a
function of N -- the evidence that guided search holds its advantage while mp-BRKGA /
random NSGA-II stall as N grows.

    python scripts/aggregate_opt_scaling.py --dir experiments/fused_tape_guided/scaling_opt_unc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = Path(args.dir)
    files = sorted(d.glob("tape_bench_*.json"))
    if not files:
        raise SystemExit(f"no tape_bench_*.json in {d}")

    per_n: dict[int, dict] = {}
    methods: list[str] = []
    for f in files:
        obj = json.loads(f.read_text())
        n = int(obj["n"])
        agg = obj["aggregate"]
        if not methods:
            methods = list(agg.keys())
        per_n[n] = {
            "coupled": obj["coupled"],
            "seeds": obj["seeds"],
            "gens": obj["gens"],
            "p_mult": obj.get("p_mult"),
            "hv_ratio": {m: agg[m]["hv_ratio"] for m in agg},
            "igd_plus": {m: agg[m]["igd_plus"] for m in agg},
        }

    ns = sorted(per_n)
    lines = ["# Optimization scaling -- HV/HV* by N (mean +- 95% CI)\n"]
    lines.append("| Method | " + " | ".join(f"N={n}" for n in ns) + " |")
    lines.append("|---|" + "|".join(["---"] * len(ns)) + "|")
    for m in methods:
        cells = []
        for n in ns:
            mean, half = per_n[n]["hv_ratio"][m]
            cells.append(f"{mean:.3f}±{half:.3f}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    # Gap of each guided arm over mp-BRKGA (the "stall" signal).
    lines.append("\n## HV/HV* gap: guided minus mp-BRKGA (positive = guided ahead)\n")
    lines.append("| Guided arm | " + " | ".join(f"N={n}" for n in ns) + " |")
    lines.append("|---|" + "|".join(["---"] * len(ns)) + "|")
    for m in [x for x in methods if x.startswith("E-HGATv2")]:
        cells = []
        for n in ns:
            g = per_n[n]["hv_ratio"][m][0] - per_n[n]["hv_ratio"]["mp-BRKGA"][0]
            cells.append(f"{g:+.3f}")
        lines.append(f"| {m} - mp-BRKGA | " + " | ".join(cells) + " |")

    text = "\n".join(lines) + "\n"
    out = Path(args.out) if args.out else d / "opt_scaling_summary.md"
    out.write_text(text)
    (d / "opt_scaling_summary.json").write_text(json.dumps(
        {"ns": ns, "methods": methods, "per_n": per_n}, indent=2))
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
