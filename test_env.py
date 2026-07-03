"""
Test_Env.py
───────────
Runs the full BTSP experiment suite across instance sizes and compatibility
structures (BTSP Full / Single / Cluster — Cases 1/2/3). This is the main
entry point used to reproduce Tables 1-4 in the paper (Section 7).
Edit only the CONFIGURATION section below. Test_main.py is never modified.

Requires CPLEX_PATH to be set (see README.md) if 'exact_cplex' is included
in METHODS_PER_CASE.
"""

from Test_main import run_tsp
from instance_generators import (generate_case1, generate_case2,
                                  generate_case3, is_feasible)
import json
import os
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# ── CONFIGURATION — edit only this section ────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# ── Cases to run ──────────────────────────────────────────────────────────────
# 1 = BTSP Full (complete bipartite), 2 = BTSP Single (one-to-one),
# 3 = BTSP Cluster (family-based, sparse)
CASES_TO_RUN = [1, 2, 3]

# ── Methods per case ──────────────────────────────────────────────────────────
# Controls which methods are run AND logged per case, matching the method
# availability described in Section 5.2 / Section 7 of the paper:
#   - 2-opt is applicable only to BTSP Full (Case 1)
#   - Q-learning is excluded from BTSP Cluster (Case 3) — see Section 7.3
METHODS_PER_CASE = {
    1: ['nearest', '2-opt', 'q-learning', 'exact_cbc', 'exact_cplex'],
    2: ['nearest', '2-opt', 'q-learning', 'exact_cbc', 'exact_cplex'],
    3: ['nearest', 'exact_cbc', 'exact_cplex'],
}

# ── Case 3 (BTSP Cluster) component-family distribution ───────────────────────
# None = equal split across all detected families (default used in the paper)
COMPONENT_DISTRIBUTION = None

# ── Number of runs per configuration ──────────────────────────────────────────
N_RUNS = 10

# ── Instance sizes ─────────────────────────────────────────────────────────────
# Each entry: ((rows, cols, components), (holders, blocks))
# Full 18-size range used to produce Tables 1-4 (10x10 through 300x300 equal
# sets). For unequal-set instances (Table 2), see instance_generators.py's
# generate_case1() called directly with asymmetric GR/KH position lists.
INSTANCE_CONFIGS = [
    ((1,  2,  5), ( 2,  5)),  # 10x10,   2 fam, 5/fam
    ((1,  2, 10), ( 2, 10)),  # 20x20,   2 fam, 10/fam
    ((1,  5,  6), ( 5,  6)),  # 30x30,   5 fam, 6/fam
    ((1,  5,  8), ( 5,  8)),  # 40x40,   5 fam, 8/fam
    ((1,  5, 10), ( 5, 10)),  # 50x50,   5 fam, 10/fam
    ((1,  5, 12), ( 5, 12)),  # 60x60,   5 fam, 12/fam
    ((1,  5, 14), ( 5, 14)),  # 70x70,   5 fam, 14/fam
    ((1,  5, 16), ( 5, 16)),  # 80x80,   5 fam, 16/fam
    ((1,  5, 18), ( 5, 18)),  # 90x90,   5 fam, 18/fam
    ((1, 10, 10), (10, 10)),  # 100x100, 10 fam, 10/fam
    ((1, 10, 12), (10, 12)),  # 120x120, 10 fam, 12/fam
    ((1, 10, 14), (10, 14)),  # 140x140, 10 fam, 14/fam
    ((1, 10, 16), (10, 16)),  # 160x160, 10 fam, 16/fam
    ((1, 10, 18), (10, 18)),  # 180x180, 10 fam, 18/fam
    ((1, 10, 20), (10, 20)),  # 200x200, 10 fam, 20/fam
    ((1, 15, 20), (15, 20)),  # 300x300, 15 fam, 20/fam
    ((1, 20, 20), (20, 20)),  # 400x400, 20 fam, 20/fam
    ((1, 20, 25), (20, 25)),  # 500x500, 20 fam, 25/fam
]

# ── Output base directory ──────────────────────────────────────────────────────
# Must match BASE_DIR used by aggregate_results.py and convert_to_tsplib.py.
BASE_DIR = "Experiment Results All Cases v2"

# ══════════════════════════════════════════════════════════════════════════════
# ── END OF CONFIGURATION ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════


def get_instance(case, gr_pos, kh_pos):
    if case == 1:
        return generate_case1(gr_pos, kh_pos)
    elif case == 2:
        return generate_case2(gr_pos, kh_pos)
    elif case == 3:
        for attempt in range(20):
            candidate = generate_case3(gr_pos, kh_pos, COMPONENT_DISTRIBUTION)
            if is_feasible(candidate):
                return candidate
            print(f"  Infeasible Case 3 instance (attempt {attempt+1}), retrying...")
        return None
    raise ValueError(f"Unknown case: {case}")


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    all_summary = []

    for case in CASES_TO_RUN:
        print(f"\n{'='*60}")
        print(f"  STARTING CASE {case}")
        print(f"{'='*60}")

        methods  = METHODS_PER_CASE[case]
        case_dir = os.path.join(BASE_DIR, f"Case{case}")
        os.makedirs(case_dir, exist_ok=True)

        for (gr_cfg, kh_cfg) in INSTANCE_CONFIGS:
            rows, cols, components = gr_cfg
            kh, blocks = kh_cfg

            gr_positions = [
                f"{row}.{col}.{comp}"
                for row  in range(1, rows + 1)
                for col  in range(1, cols + 1)
                for comp in range(1, components + 1)
            ]
            kh_positions = [
                f"{holder}.{block}"
                for holder in range(1, kh + 1)
                for block  in range(1, blocks + 1)
            ]

            gr_n = len(gr_positions)
            kh_n = len(kh_positions)

            if case == 2 and gr_n != kh_n:
                print(f"[Case 2] Skipping {gr_n}x{kh_n} — equal sets required.")
                continue

            config_label = f"GR_{rows}x{cols}x{components}_KH_{kh}x{blocks}"
            config_dir   = os.path.join(case_dir, config_label)
            os.makedirs(config_dir, exist_ok=True)

            for run_id in range(1, N_RUNS + 1):
                print(f"\n[Case {case}] GR={gr_n} KH={kh_n} | Run {run_id}/{N_RUNS}")

                run_dir = os.path.join(config_dir, f"Run_{run_id}")
                os.makedirs(run_dir, exist_ok=True)
                prefix  = f"GR_{rows}x{cols}x{components}_KH_{kh}x{blocks}_Run_{run_id}"

                input_json = get_instance(case, gr_positions, kh_positions)
                if input_json is None:
                    print(f"  Skipping run {run_id} — no feasible instance after 20 attempts.")
                    continue

                input_file = os.path.join(run_dir, f"input_{prefix}.json")
                with open(input_file, 'w') as f:
                    json.dump(input_json, f, indent=4)

                output_file = os.path.join(run_dir, f"output_{prefix}.json")
                results = run_tsp(
                    input_file, output_file,
                    gr_positions, kh_positions,
                    experiment_case=case,
                    methods_to_run=methods
                )

                exact_cost = (results['exact_cbc']['cost']
                              if 'exact_cbc' in results else None)

                result_rows = []
                for method in methods:
                    if method not in results:
                        continue
                    if exact_cost and not method.startswith('exact'):
                        gap = round(
                            ((results[method]['cost'] - exact_cost) / exact_cost) * 100, 2)
                    else:
                        gap = None

                    row = [
                        case,
                        f"{rows}x{cols}x{components}",
                        f"{kh}x{blocks}",
                        run_id,
                        method.capitalize(),
                        results[method]['cost'],
                        round(results[method]['exec_time'], 4),
                        gap
                    ]
                    all_summary.append(row)
                    result_rows.append(row[1:])

                df_run = pd.DataFrame(result_rows, columns=[
                    'Gravity Rack', 'Kit Holder', 'Run', 'Method',
                    'Cost', 'Exec time', 'Optimality Gap (%)'
                ])
                csv_file = os.path.join(run_dir, f"results_{prefix}.csv")
                df_run.to_csv(csv_file, index=False)
                print(f"  Saved: {csv_file}")

    df_all = pd.DataFrame(all_summary, columns=[
        'Case', 'Gravity Rack', 'Kit Holder', 'Run', 'Method',
        'Cost', 'Exec time', 'Optimality Gap (%)'
    ])
    return df_all


if __name__ == "__main__":
    main()