"""scripts/run_gnn_landscape.py -- the scalable, model-native landscape analysis.

For each instance size N: train the frozen E-HGATv2 core + fused TAPE head, sample random
schedules, and compute the GNN/TAPE-derived decision-family importance landscape
(ehgat.explain.gnn_landscape.gnn_landscape). At the small sizes where the exact
oracle is affordable the exact landscape is computed as well and the rank agreement
reported, which is the validation supporting the GNN landscape at the large sizes
(DL: 40-160 tasks) where exact variance-based methods (Sobol) are intractable.

Usage::

    python scripts/run_gnn_landscape.py --tasks 10 20 50 100 --validate-upto 20 \\
        --schedules 256 --core-samples 3000 --core-epochs 80 --fused-samples 1500 \\
        --fused-epochs 40 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console

console = Console()
OUT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "gnn_landscape"


def main() -> None:
    p = argparse.ArgumentParser(description="Scalable GNN/TAPE landscape analysis.")
    p.add_argument("--tasks", type=int, nargs="+", default=[10, 20, 50, 100])
    p.add_argument("--validate-upto", type=int, default=20,
                   help="run the exact-oracle landscape + rank agreement for N <= this")
    p.add_argument("--schedules", type=int, default=256, help="sampled schedules per instance")
    p.add_argument("--core-samples", type=int, default=3000)
    p.add_argument("--core-epochs", type=int, default=80)
    p.add_argument("--fused-samples", type=int, default=1500)
    p.add_argument("--fused-epochs", type=int, default=40)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", type=str, default=None)
    args = p.parse_args()

    from ehgat.environment.decoder import NUM_BLOCKS, decode
    from ehgat.environment.instance import build_toy_instance
    from ehgat.explain.gnn_landscape import (
        exact_landscape,
        gnn_landscape,
        landscape_rank_agreement,
    )
    from ehgat.explain.train_fused import FusedTrainConfig, build_core
    from ehgat.explain.train_fused_batched import train_fused_batched
    from ehgat.utils.seeding import make_rng

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for n in args.tasks:
        t0 = time.perf_counter()
        instance = build_toy_instance(num_tasks=n)  # uncoupled regime
        core = build_core(instance, seed=args.seed,
                          num_samples=args.core_samples, epochs=args.core_epochs)
        fused = train_fused_batched(
            instance, core,
            FusedTrainConfig(num_samples=args.fused_samples, epochs=args.fused_epochs,
                             seed=args.seed),
            device=args.device,
        )
        rng = make_rng(args.seed + 7)
        schedules = [decode(rng.random(NUM_BLOCKS * n), instance) for _ in range(args.schedules)]

        gnn = gnn_landscape(fused.model, instance, schedules)
        row: dict = {"num_tasks": n, "r2_makespan": fused.metrics.get("r2_makespan"),
                     "gnn_landscape": gnn.to_dict()}
        if n <= args.validate_upto:
            exact = exact_landscape(instance, schedules)
            row["exact_landscape"] = exact.to_dict()
            row["rank_agreement"] = landscape_rank_agreement(gnn, exact)
            console.print(f"[green]N={n}[/green] rank_agreement(gnn,exact)="
                          f"{row['rank_agreement']:.3f} agv_share={gnn.agv_share:.3f}")
        else:
            console.print(f"[green]N={n}[/green] (exact skipped) agv_share={gnn.agv_share:.3f} "
                          f"importance={ {k: round(v,3) for k,v in gnn.makespan_importance.items()} }")
        row["wall_s"] = time.perf_counter() - t0
        results.append(row)

    tag = args.tag or "uncoupled"
    out_path = OUT_DIR / f"gnn_landscape_{tag}.json"
    out_path.write_text(json.dumps({"config": vars(args), "results": results}, indent=2))
    console.print(f"\n[bold green]wrote[/bold green] {out_path}")


if __name__ == "__main__":
    main()
