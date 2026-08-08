"""Merge seed-sharded tape_bench shards into one canonical tape_bench_<tag>.json.

run_tape_guided_bench.py can shard an instance's seeds across parallel processes
(--seed-start + --out-tag + --out-dir shards/). Each shard writes a
tape_bench json holding its seeds' raw per-seed metric arrays. This script, per
instance, concatenates the raw arrays across shards, recomputes the 95% CIs, and
writes the merged file into experiments/fused_tape_guided/ where compute_paper_stats
globs it. Faithfulness and reference fields are identical across shards (fixed seeds),
so they are taken from the first shard.

Usage:
    python scripts/merge_tape_shards.py --shards-dir experiments/fused_tape_guided/shards
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.stats as sps

OUT = Path(__file__).resolve().parents[1] / "experiments" / "fused_tape_guided"
METRICS = ["gd_plus", "igd_plus", "spread", "hv", "hv_ratio", "evals"]


def _ci(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, float)
    if a.size < 2:
        return (float(a.mean()) if a.size else float("nan")), 0.0
    half = float(sps.t.ppf(0.975, a.size - 1) * a.std(ddof=1) / np.sqrt(a.size))
    return float(a.mean()), half


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir", default=str(OUT / "shards"))
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()
    sdir = Path(args.shards_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Group shard files by instance base tag: tape_bench_<base>_sh<k>.json
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(sdir.glob("tape_bench_*_sh*.json")):
        base = re.sub(r"_sh\d+$", "", f.stem.replace("tape_bench_", ""))
        groups[base].append(f)

    if not groups:
        print(f"No shard files in {sdir}")
        return

    for base, files in sorted(groups.items()):
        recs = [json.loads(f.read_text()) for f in sorted(files)]
        methods = list(recs[0]["raw"].keys())
        merged_raw: dict[str, dict[str, list[float]]] = {
            m: {k: [] for k in METRICS} for m in methods}
        total_seeds = 0
        for r in recs:
            for m in methods:
                for k in METRICS:
                    merged_raw[m][k].extend(r["raw"][m].get(k, []))
        total_seeds = max(len(merged_raw[m]["hv_ratio"]) for m in methods)

        agg = {m: {k: list(_ci(merged_raw[m][k])) for k in
                   ["hv_ratio", "gd_plus", "igd_plus", "spread", "evals"]}
               for m in methods}

        base_rec = recs[0]
        out_rec = dict(base_rec)
        out_rec["seeds"] = total_seeds
        out_rec["n_shards"] = len(files)
        out_rec["raw"] = merged_raw
        out_rec["aggregate"] = agg
        out_path = out / f"tape_bench_{base}.json"
        out_path.write_text(json.dumps(out_rec, indent=2))
        hv = agg[next(iter(methods))]["hv_ratio"][0]
        print(f"  {base}: {len(files)} shards -> {total_seeds} seeds  "
              f"({', '.join(methods)})  -> {out_path.name}")

    print(f"Merged {len(groups)} instances into {out}")


if __name__ == "__main__":
    main()