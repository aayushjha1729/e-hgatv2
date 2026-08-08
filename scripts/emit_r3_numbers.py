"""Emit the per-instance tab:main numbers (HV per method + faithfulness) from the
merged tape_bench_*.json files, in the paper's SD-N row order."""
import json
from pathlib import Path

OUT = Path("experiments/fused_tape_guided")
ORDER = ["toy5", "toy8", "toy10", "toy15", "toy20", "L07", "L15", "L21", "L35",
         "toy10_pp30", "toy20_pp30"]
NAME = {"toy5": "SD-5", "toy8": "SD-8", "toy10": "SD-10", "toy15": "SD-15",
        "toy20": "SD-20", "L07": "L07", "L15": "L15", "L21": "L21", "L35": "L35",
        "toy10_pp30": "SD-10-C", "toy20_pp30": "SD-20-C"}
M = ["E-HGATv2-TAPE", "E-HGATv2-attn", "NSGA-II (random)", "mp-BRKGA", "single-pop BRKGA"]

for b in ORDER:
    f = OUT / f"tape_bench_{b}.json"
    if not f.exists():
        print(f"{NAME[b]:8} MISSING")
        continue
    d = json.loads(f.read_text())
    ag, fa, sd = d["aggregate"], d["faithfulness"], d["seeds"]
    hv = [ag[m]["hv_ratio"][0] for m in M]
    best = max(hv)
    cells = " ".join(f"{'*' if v == best else ' '}{v:.3f}" for v in hv)
    print(f"{NAME[b]:8} N={d['n']:2} seeds={sd:2} | {cells} | "
          f"Jac={fa['tape_leg_critical_jaccard']:.3f} rho={fa['attention_spearman_rho']:+.3f}")