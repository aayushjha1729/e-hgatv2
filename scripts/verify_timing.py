"""Independent timing verifier: a literal, line-by-line transcription of the FSMJ 2023
MILP timing constraints (typeset_equations/2023_FSMJ_AGVinCT_equations.md, Eqs 2-4, 10-18),
solved by fixed-point longest-path iteration with the schedule's binaries fixed.

It shares no code with evaluator.py's composition: it tracks neither AGV-free times nor
handoffs, and enforces each MILP inequality var >= rhs repeatedly until nothing changes
(Bellman-Ford on the DAG, converging in at most N sweeps). The leg time and energy
kinematics are taken from physics.py, itself validated against the published curves, which
confines the object under test to the timing composition (the max-plus structure).

Symbol map (Eqs as written):
  c[j] = handling completion time of task j        (variable)
  r[j] = delivery time of task j                   (variable)
  tau[j] = QC handling time                        (Task.handling_time)
  theta[j] = empty travel time onto j              (origin -> j.pickup, j's empty speed)
  vartheta[j] = loaded travel time of j            (j.pickup -> j.dropoff, j's loaded speed)
"""

from __future__ import annotations

import numpy as np

from ehgat.environment.decoder import decode
from ehgat.environment.evaluator import build_precedence, evaluate
from ehgat.environment.instance import TaskKind, build_toy_instance
from ehgat.environment.physics import leg_energy, travel_time


def milp_timing(schedule, instance) -> tuple[float, float]:
    """(makespan, energy) for a fixed schedule by literal MILP-constraint iteration."""
    n = instance.num_tasks
    agv_prev, qc_prev, _ = build_precedence(schedule.agv_sequences, schedule.qc_sequences, n)
    tasks = instance.tasks

    # Leg kinematics (the inputs the evaluator sees; only the composition differs).
    theta = [0.0] * n       # empty travel time onto j
    vartheta = [0.0] * n    # loaded travel time of j
    energy = 0.0
    for j in range(n):
        i = agv_prev[j]
        origin = instance.agv_start if i < 0 else tasks[i].dropoff
        empty_dist = instance.distance.distance(origin, tasks[j].pickup)
        loaded_dist = instance.loaded_distance(tasks[j])
        theta[j] = travel_time(empty_dist, schedule.empty_speed[j], loaded=False)
        vartheta[j] = travel_time(loaded_dist, schedule.loaded_speed[j], loaded=True)
        energy += leg_energy(empty_dist, schedule.empty_speed[j], loaded=False)
        energy += leg_energy(loaded_dist, schedule.loaded_speed[j], loaded=True)

    tau = [float(t.handling_time) for t in tasks]
    is_load = [t.kind is TaskKind.LOAD for t in tasks]

    c = [0.0] * n
    r = [0.0] * n
    for _ in range(n + 2):  # Bellman-Ford: converges in <= N sweeps on a DAG
        changed = False
        for j in range(n):
            i = agv_prev[j]
            q = qc_prev[j]
            new_c = c[j]
            new_r = r[j]

            # (10) QC precedence: c_j >= c_{prevQC} + tau_j
            if q >= 0:
                new_c = max(new_c, c[q] + tau[j])

            if is_load[j]:
                # AGV release after predecessor i:
                #   first task        -> 0           (Eq 13: r_j >= theta_aj + vartheta_j)
                #   load-pred i        -> c_i - tau_i (Eq 12)
                #   unload-pred i      -> r_i         (Eq 18)
                if i < 0:
                    release = 0.0
                elif is_load[i]:
                    release = c[i] - tau[i]
                else:
                    release = r[i]
                new_r = max(new_r, release + theta[j] + vartheta[j])
                # (11) c_j >= r_j + tau_j
                new_c = max(new_c, new_r + tau[j])
            else:
                # Unloading completion (Eqs 14/15/17): c_j >= arrival, NO +tau here.
                #   first task    -> theta_aj                 (Eq 15)
                #   unload-pred i -> r_i + theta_ij           (Eq 14)
                #   load-pred i   -> (c_i - tau_i) + theta_ij (Eq 17)
                if i < 0:
                    arrival = theta[j]
                elif is_load[i]:
                    arrival = (c[i] - tau[i]) + theta[j]
                else:
                    arrival = r[i] + theta[j]
                new_c = max(new_c, arrival)
                # (16) r_j >= c_j + vartheta_j
                new_r = max(new_r, new_c + vartheta[j])

            if new_c != c[j] or new_r != r[j]:
                c[j], r[j] = new_c, new_r
                changed = True
        if not changed:
            break
    else:
        raise RuntimeError("fixed-point did not converge (cycle in schedule?)")

    # (2) C_max >= c_j for loads; (3) C_max >= r_j for unloads.
    makespan = max(c[j] if is_load[j] else r[j] for j in range(n))
    return makespan, energy


def milp_timing_lp(schedule, instance) -> float:
    """Makespan for a fixed schedule via an OR-Tools GLOP LP (independent of milp_timing).

    Encodes Eqs (10)-(18),(2),(3) verbatim as a continuous linear program over c_j, r_j,
    C_max >= 0 and minimizes C_max. GLOP is an independent simplex solver, making this a
    second, fully independent oracle for the timing composition.
    """
    from ortools.linear_solver import pywraplp

    n = instance.num_tasks
    agv_prev, qc_prev, _ = build_precedence(schedule.agv_sequences, schedule.qc_sequences, n)
    tasks = instance.tasks
    theta = [0.0] * n
    vartheta = [0.0] * n
    for j in range(n):
        i = agv_prev[j]
        origin = instance.agv_start if i < 0 else tasks[i].dropoff
        theta[j] = travel_time(
            instance.distance.distance(origin, tasks[j].pickup),
            schedule.empty_speed[j], loaded=False,
        )
        vartheta[j] = travel_time(
            instance.loaded_distance(tasks[j]), schedule.loaded_speed[j], loaded=True
        )
    tau = [float(t.handling_time) for t in tasks]
    is_load = [t.kind is TaskKind.LOAD for t in tasks]

    s = pywraplp.Solver.CreateSolver("GLOP")
    inf = s.infinity()
    c = [s.NumVar(0, inf, f"c{j}") for j in range(n)]
    r = [s.NumVar(0, inf, f"r{j}") for j in range(n)]
    cmax = s.NumVar(0, inf, "Cmax")
    for j in range(n):
        i, q = agv_prev[j], qc_prev[j]
        if q >= 0:
            s.Add(c[j] >= c[q] + tau[j])                                  # (10)
        if is_load[j]:
            if i < 0:
                s.Add(r[j] >= theta[j] + vartheta[j])                     # (13)
            elif is_load[i]:
                s.Add(r[j] >= c[i] - tau[i] + theta[j] + vartheta[j])     # (12)
            else:
                s.Add(r[j] >= r[i] + theta[j] + vartheta[j])              # (18)
            s.Add(c[j] >= r[j] + tau[j])                                  # (11)
            s.Add(cmax >= c[j])                                           # (2)
        else:
            if i < 0:
                s.Add(c[j] >= theta[j])                                   # (15)
            elif is_load[i]:
                s.Add(c[j] >= c[i] - tau[i] + theta[j])                  # (17)
            else:
                s.Add(c[j] >= r[i] + theta[j])                           # (14)
            s.Add(r[j] >= c[j] + vartheta[j])                            # (16)
            s.Add(cmax >= r[j])                                           # (3)
    s.Minimize(cmax)
    assert s.Solve() == pywraplp.Solver.OPTIMAL
    return cmax.solution_value()


def main() -> None:
    from ehgat.environment.oracle import Structure, evaluate_speeds

    rng = np.random.default_rng(0)
    n_make = n_energy = trials = 0
    fp_lp_disagree = orc_make = 0
    max_make = max_make_rel = max_fp_lp = max_orc = 0.0
    for n_tasks in (5, 8, 10, 12):
        for n_agvs in (1, 2, 3):
            inst = build_toy_instance(seed=n_tasks, num_tasks=n_tasks, num_agvs=n_agvs)
            for _ in range(500):
                sched = decode(rng.random(4 * n_tasks), inst)
                ev = evaluate(sched, inst)
                mk, en = milp_timing(sched, inst)       # fixed-point oracle
                mk_lp = milp_timing_lp(sched, inst)      # GLOP LP oracle
                trials += 1
                # Cross-check the two independent oracles agree with each other.
                d_fp_lp = abs(mk - mk_lp)
                if d_fp_lp > 1e-4:
                    fp_lp_disagree += 1
                    max_fp_lp = max(max_fp_lp, d_fp_lp)
                # Evaluator vs oracle.
                dm = abs(ev.makespan - mk)
                if dm > 1e-6:
                    n_make += 1
                    max_make = max(max_make, dm)
                    max_make_rel = max(max_make_rel, dm / mk)
                if abs(ev.energy - en) > 1e-6:
                    n_energy += 1
                # Oracle (exact scaled-int recurrence) vs fixed-point oracle.
                struct = Structure(sched.assignment, sched.agv_sequences, sched.qc_sequences)
                omk, _ = evaluate_speeds(struct, sched.empty_speed, sched.loaded_speed, inst)
                d_orc = abs(float(omk) - mk)
                if d_orc > 1e-6:
                    orc_make += 1
                    max_orc = max(max_orc, d_orc)
    print(f"trials={trials}")
    print(f"[oracle cross-check] fixed-point vs GLOP disagree: {fp_lp_disagree} "
          f"({100*fp_lp_disagree/trials:.2f}%)  max_abs={max_fp_lp:.4f}s")
    print(f"[evaluator vs oracle] makespan mismatch: {n_make} ({100*n_make/trials:.1f}%)  "
          f"max_abs={max_make:.3f}s  max_rel={100*max_make_rel:.2f}%")
    print(f"[evaluator vs oracle] energy mismatch: {n_energy} ({100*n_energy/trials:.1f}%)")
    print(f"[oracle.evaluate_speeds vs oracle] makespan mismatch: {orc_make} "
          f"({100*orc_make/trials:.1f}%)  max_abs={max_orc:.3f}s")


if __name__ == "__main__":
    main()
