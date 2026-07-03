# REPOSITORY NAME (c) by the University of Piraeus, Greece.
#
# REPOSITORY NAME is licensed under a
# Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License.
#
# You should have received a copy of the license along with this
# work.  If not, see <http://creativecommons.org/licenses/by-nc-nd/3.0/>.
#
# ---------------------------------------------------------------------------
# exact_lp_relaxation_test.py
#
# Standalone diagnostic script to answer Reviewer 1's comment (i-b):
# "show that the LP relaxation works quite well (due to sparsity of the
#  graph and several constraints)."
#
# This script solves the ROOT LP relaxation of the BTSP IP model
# (constraints (2)-(5) in Section 5.1 of the paper, WITHOUT the subtour-
# elimination constraints (6)) using CPLEX, and compares it against the
# proven IP optimum obtained via the existing exact_method.py pipeline
# (which does include iterative subtour elimination).
#
# It deliberately does NOT modify exact_method.py: it reuses the data-
# preparation helpers from that module (classify_nodes, create_distances_dict,
# extract_possible_edges, run_exact_tsp, etc.) and builds its own continuous-
# relaxation model here, mirroring the exact model's constraints exactly.
#
# Only CPLEX is used for the relaxation (and recommended for the exact run
# too, for a like-for-like comparison), since CPLEX is already the reference
# solver used throughout the paper's computational study (Section 7).
# ---------------------------------------------------------------------------

import pulp
import statistics

from exact_method import (
    classify_nodes,
    create_distances_dict,
    extract_possible_edges,
    check_connectivity,
    analyze_sets,
    run_exact_tsp,
)
from parse_json import create_distance_matrices, load_instance

# Matches the convention used in Test_main.py — update to your local CPLEX path.
import os
# CPLEX_PATH = os.environ.get('CPLEX_PATH')

CPLEX_PATH = '/Users/konstantinosgiannakos/Applications/CPLEX_Studio2212/cplex/bin/arm64_osx/cplex'

# ---------------------------------------------------------------------------
# 1. Root LP relaxation model
# ---------------------------------------------------------------------------
def solve_root_lp_relaxation(distances, possible_edges, set_1, set_2, solver=None):
    """
    Builds and solves the ROOT LP relaxation of the BTSP IP model:
    identical objective and constraints (2)-(5) to solve_tsp() in
    exact_method.py, but with continuous decision variables x_ij in [0,1]
    instead of binary, and WITHOUT subtour-elimination constraints (6).

    This mirrors exactly what solve_tsp() builds before any subtour cuts
    are added by iterative_subtour_elimination() -- i.e. the true root
    relaxation bound.

    Parameters:
    - distances: dict of (i, j) -> cost, as built by create_distances_dict.
    - possible_edges: list of valid (i, j) edges.
    - set_1: kit holder nodes.
    - set_2: gravity rack nodes.
    - solver: a PuLP solver instance. Defaults to CPLEX_CMD if None.

    Returns:
    - lp_bound (float or None): the LP relaxation objective value, or None
      if the LP did not solve to optimality.
    - prob (pulp.LpProblem): the solved LP problem object (for inspection).
    """
    nodes = list(set([k[0] for k in distances.keys()] + [k[1] for k in distances.keys()]))
    nodes.remove('0.0.0')

    prob = pulp.LpProblem("Bipartite_TSP_Root_LP_Relaxation", pulp.LpMinimize)

    # Continuous relaxation of the exact model's binary variables
    x = pulp.LpVariable.dicts(
        "x", (nodes + ['0.0', '0.0.0'], nodes + ['0.0', '0.0.0']),
        lowBound=0, upBound=1, cat='Continuous'
    )

    # Objective (identical to solve_tsp)
    prob += pulp.lpSum([distances[(i, j)] * x[i][j] for i, j in possible_edges])

    # Constraint set (2): start at '0.0', go to set_2
    prob += pulp.lpSum([x['0.0'][j] for j in set_2 if ('0.0', j) in possible_edges]) == 1

    # Constraint set (3): go from set_1 to '0.0.0'
    prob += pulp.lpSum([x[i]['0.0.0'] for i in set_1 if (i, '0.0.0') in possible_edges]) == 1

    # Degree constraints on set_2 (<=1 each direction)
    for node in set_2:
        prob += pulp.lpSum([x[node][j] for j in set_1 if (node, j) in possible_edges]) <= 1
        prob += pulp.lpSum([x[i][node] for i in set_1 if (i, node) in possible_edges]) <= 1

    # Flow conservation constraints for set_1
    for node in set_1:
        prob += pulp.lpSum([x[i][node] for i in set_2 if (i, node) in possible_edges]) == 1
        prob += pulp.lpSum([x[node][j] for j in set_2 + ['0.0.0'] if (node, j) in possible_edges]) == 1

    # Incoming = outgoing for set_2
    for node in set_2:
        prob += pulp.lpSum(
            [x[i][node] for i in set_1 + ['0.0'] if i != node and (i, node) in possible_edges]
        ) - pulp.lpSum(
            [x[node][j] for j in set_1 + ['0.0'] if j != node and (node, j) in possible_edges]
        ) == 0

    # Tour closure: '0.0.0' -> '0.0'
    prob += pulp.lpSum([x['0.0.0']['0.0']]) == 1

    if solver is None:
        solver = pulp.CPLEX_CMD(msg=False)
    prob.solve(solver)

    if pulp.LpStatus[prob.status] != 'Optimal':
        return None, prob

    return pulp.value(prob.objective), prob


# ---------------------------------------------------------------------------
# 2. Combined exact + LP relaxation gap report for a single instance
# ---------------------------------------------------------------------------
def report_lp_gap_for_instance(input_data, label="", cplex_path=None):
    """
    Runs BOTH the exact BTSP solve (via exact_method.run_exact_tsp, using
    CPLEX) and the root LP relaxation for the same instance, then prints
    and returns the integrality gap.

    Parameters:
    - input_data: same input dict format expected by run_exact_tsp /
      create_distance_matrices.
    - label: optional string identifying the instance (e.g. "10x10" or
      "Industrial - Tuple 1") for the printed report.
    - cplex_path: optional path to the CPLEX binary, matching the format
      used elsewhere in this repo (e.g. pulp.CPLEX_CMD(path=...)).

    Returns:
    - dict with keys: label, ip_optimum, lp_bound, gap_pct
      (values are None if either solve failed).
    """
    cplex_solver = (pulp.CPLEX_CMD(msg=False) if cplex_path is None
                     else pulp.CPLEX_CMD(path=cplex_path, msg=False))

    # --- Exact solve (reuses the tested pipeline, unmodified) ---
    _, exact_tour_cost, _ = run_exact_tsp(input_data, solver=cplex_solver)

    # --- Build the same distances/edges the exact solve used, for the LP ---
    a_to_b_matrix = create_distance_matrices(input_data)
    set_a, set_b = classify_nodes(a_to_b_matrix)
    distances = create_distances_dict(a_to_b_matrix, set_a, set_b)
    possible_edges = extract_possible_edges(distances)
    check_connectivity(set_a, set_b, distances)
    analyze_sets(set_a, set_b, distances)

    # --- Root LP relaxation ---
    lp_bound, lp_prob = solve_root_lp_relaxation(
        distances, possible_edges, set_a, set_b, solver=cplex_solver
    )

    result = {"label": label, "ip_optimum": exact_tour_cost, "lp_bound": lp_bound, "gap_pct": None}

    if exact_tour_cost is None:
        print(f"[{label}] Exact solve failed — skipping.")
        return result
    if lp_bound is None:
        print(f"[{label}] LP relaxation failed to solve to optimality — skipping.")
        return result

    gap_pct = (exact_tour_cost - lp_bound) / exact_tour_cost * 100
    result["gap_pct"] = gap_pct

    print(f"[{label}] IP optimum = {exact_tour_cost:.2f} ms | "
          f"LP bound = {lp_bound:.2f} ms | "
          f"Integrality gap = {gap_pct:.4f}%")

    return result


# ---------------------------------------------------------------------------
# 3. Batch runner + summary table
# ---------------------------------------------------------------------------
def run_lp_gap_suite(instances, cplex_path=None, save_csv_path=None):
    """
    Runs report_lp_gap_for_instance() over a list of (label, input_data)
    pairs and prints a summary table. Optionally writes results to CSV.

    Parameters:
    - instances: list of (label, input_data) tuples.
    - cplex_path: optional CPLEX binary path, forwarded to each call.
    - save_csv_path: optional path; if given, writes the summary table there.

    Returns:
    - list of result dicts, one per instance (see report_lp_gap_for_instance).
    """
    results = []
    print("=" * 75)
    print("LP Relaxation Integrality Gap Report (CPLEX)")
    print("=" * 75)

    for label, input_data in instances:
        result = report_lp_gap_for_instance(input_data, label=label, cplex_path=cplex_path)
        results.append(result)

    print("=" * 75)
    print(f"{'Instance':<25}{'IP Optimum (ms)':>20}{'LP Bound (ms)':>20}{'Gap (%)':>12}")
    print("-" * 75)
    for r in results:
        ip_str = f"{r['ip_optimum']:.2f}" if r['ip_optimum'] is not None else "FAILED"
        lp_str = f"{r['lp_bound']:.2f}" if r['lp_bound'] is not None else "FAILED"
        gap_str = f"{r['gap_pct']:.4f}" if r['gap_pct'] is not None else "-"
        print(f"{r['label']:<25}{ip_str:>20}{lp_str:>20}{gap_str:>12}")
    print("=" * 75)

    if save_csv_path:
        import csv
        with open(save_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Instance", "IP_Optimum_ms", "LP_Bound_ms", "Integrality_Gap_pct"])
            for r in results:
                writer.writerow([r["label"], r["ip_optimum"], r["lp_bound"], r["gap_pct"]])
        print(f"Results written to {save_csv_path}")

    return results


# ---------------------------------------------------------------------------
# 4. Multi-run averaging (consistent with the 10-run methodology in Section 7)
# ---------------------------------------------------------------------------
def build_run_paths(case_num, gr_kh_folder, filename_stem, n_runs=10, ext='.json',
                     base_dir="Experiment Results All Cases v2"):
    """
    Builds the Run_1..Run_N file paths for a given instance size, following
    the folder convention already used by your experiment pipeline:

        {base_dir}/Case{case_num}/{gr_kh_folder}/Run_{n}/{filename_stem}_Run_{n}{ext}

    Example:
        build_run_paths(3, "GR_1x5x8_KH_5x8", "input_GR_1x5x8_KH_5x8", n_runs=10)
        -> ["Experiment Results All Cases v2/Case3/GR_1x5x8_KH_5x8/Run_1/"
            "input_GR_1x5x8_KH_5x8_Run_1.json", ... Run_10]
    """
    return [
        f"{base_dir}/Case{case_num}/{gr_kh_folder}/Run_{n}/{filename_stem}_Run_{n}{ext}"
        for n in range(1, n_runs + 1)
    ]


def report_lp_gap_averaged(label, file_paths, cplex_path=None):
    """
    Runs report_lp_gap_for_instance() over multiple run files of the SAME
    instance size/case (e.g. Run_1..Run_10) and reports the mean +/- std
    integrality gap across runs, mirroring the 10-independent-runs
    methodology used throughout Section 7 of the paper.

    Parameters:
    - label: base label for this size/case (run number is appended per file).
    - file_paths: list of .tsp/.json file paths, one per run.
    - cplex_path: optional CPLEX binary path.

    Returns a dict with n_runs, mean/std gap, and mean IP/LP values.
    """
    per_run_results = []
    for i, file_path in enumerate(file_paths, start=1):
        try:
            input_data = load_instance(file_path)
        except Exception as e:
            print(f"[{label} Run_{i}] Failed to load '{file_path}': {e}")
            continue
        result = report_lp_gap_for_instance(input_data, label=f"{label} Run_{i}", cplex_path=cplex_path)
        if result["gap_pct"] is not None:
            per_run_results.append(result)

    if not per_run_results:
        print(f"[{label}] No successful runs — cannot average.")
        return {"label": label, "n_runs": 0, "mean_gap_pct": None, "std_gap_pct": None,
                "mean_ip_optimum": None, "mean_lp_bound": None}

    gaps = [r["gap_pct"] for r in per_run_results]
    ips = [r["ip_optimum"] for r in per_run_results]
    lps = [r["lp_bound"] for r in per_run_results]

    mean_gap = statistics.mean(gaps)
    std_gap = statistics.stdev(gaps) if len(gaps) > 1 else 0.0
    mean_ip = statistics.mean(ips)
    mean_lp = statistics.mean(lps)

    print(f"[{label}] n={len(per_run_results)} runs | "
          f"Mean IP = {mean_ip:.2f} ms | Mean LP = {mean_lp:.2f} ms | "
          f"Mean gap = {mean_gap:.4f}% (+/- {std_gap:.4f})")

    return {
        "label": label,
        "n_runs": len(per_run_results),
        "mean_gap_pct": mean_gap,
        "std_gap_pct": std_gap,
        "mean_ip_optimum": mean_ip,
        "mean_lp_bound": mean_lp,
    }


def run_lp_gap_suite_averaged(configs, cplex_path=None, save_csv_path=None):
    """
    Batch-runs report_lp_gap_averaged() over multiple instance sizes/cases
    and prints a summary table.

    Parameters:
    - configs: list of (label, file_paths, experiment_case) tuples, where
      file_paths is a list of Run_1..Run_N paths for that size/case
      (see build_run_paths).
    """
    results = []
    print("=" * 95)
    print("LP Relaxation Integrality Gap Report — Averaged over multiple runs (CPLEX)")
    print("=" * 95)

    for label, file_paths, experiment_case in configs:
        full_label = f"{label} (Case {experiment_case})"
        agg = report_lp_gap_averaged(full_label, file_paths, cplex_path=cplex_path)
        results.append(agg)

    print("=" * 95)
    print(f"{'Instance':<28}{'n':>4}{'Mean IP (ms)':>18}{'Mean LP (ms)':>18}{'Mean Gap (%)':>15}{'Std Gap':>10}")
    print("-" * 95)
    for r in results:
        n_str = str(r["n_runs"])
        ip_str = f"{r['mean_ip_optimum']:.2f}" if r['mean_ip_optimum'] is not None else "FAILED"
        lp_str = f"{r['mean_lp_bound']:.2f}" if r['mean_lp_bound'] is not None else "FAILED"
        gap_str = f"{r['mean_gap_pct']:.4f}" if r['mean_gap_pct'] is not None else "-"
        std_str = f"{r['std_gap_pct']:.4f}" if r['std_gap_pct'] is not None else "-"
        print(f"{r['label']:<28}{n_str:>4}{ip_str:>18}{lp_str:>18}{gap_str:>15}{std_str:>10}")
    print("=" * 95)

    if save_csv_path:
        import csv
        with open(save_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Instance", "N_Runs", "Mean_IP_Optimum_ms", "Mean_LP_Bound_ms",
                              "Mean_Integrality_Gap_pct", "Std_Integrality_Gap_pct"])
            for r in results:
                writer.writerow([r["label"], r["n_runs"], r["mean_ip_optimum"],
                                  r["mean_lp_bound"], r["mean_gap_pct"], r["std_gap_pct"]])
        print(f"Results written to {save_csv_path}")

    return results


# ---------------------------------------------------------------------------
# 5. Config — same style as run_single_instance.py, but a list of instances
# ---------------------------------------------------------------------------
# Each entry: (label, file_path, experiment_case)
#   - file_path: path to a .tsp (TSPLIB) or .json instance file, exactly as
#     used in run_single_instance.py's INPUT_FILE.
#   - experiment_case: 1/2/3, kept only for the printed label — it does NOT
#     change how the LP relaxation is built. The compatibility structure
#     (Case 1 dense / Case 2 one-to-one / Case 3 family-sparse) is already
#     baked into which edges exist in the instance file's distanceMatrix,
#     so it flows through automatically via load_instance() + the same
#     create_distances_dict()/extract_possible_edges() calls used by the
#     exact solver.
INSTANCE_CONFIGS = [
    # ("Case1_10x10", "Experiment Results All Cases v2/Case1/GR_1x2x5_KH_2x5/Run_1/input_GR_1x2x5_KH_2x5_Run_1.tsp", 1),
    # ("Case1_50x50", "Experiment Results All Cases v2/Case1/.../input_....tsp", 1),
    # ("Case2_10x10", "Experiment Results All Cases v2/Case2/.../input_....tsp", 2),
    # ("Case3_10x10", "Experiment Results All Cases v2/Case3/.../input_....tsp", 3),
    # ("Industrial_Tuple1", "industrial/kh_tuple_1.json", 3),
]

OUTPUT_CSV = "lp_relaxation_gap_results.csv"

# Set to True to run the 10-run-averaged version instead of single-run.
USE_AVERAGING = True

# Each entry: (label, file_paths, experiment_case), where file_paths is a
# list of Run_1..Run_N paths for that size/case. Use build_run_paths() to
# generate these automatically from your folder convention, e.g.:
#
AVERAGED_INSTANCE_CONFIGS = [
  ("Case3_40x40", build_run_paths(3, "GR_1x5x8_KH_5x8",
                                   "input_GR_1x5x8_KH_5x8", n_runs=10), 3),
  ("Case1_10x10", build_run_paths(1, "GR_1x2x5_KH_2x5",
                                   "input_GR_1x2x5_KH_2x5", n_runs=10), 1),
]
# AVERAGED_INSTANCE_CONFIGS = [
#     # ("Case3_40x40", build_run_paths(3, "GR_1x5x8_KH_5x8",
#     #                                  "input_GR_1x5x8_KH_5x8", n_runs=10), 3),
# ]


def load_instances_from_config(configs):
    """
    Loads each (label, file_path, experiment_case) entry via
    parse_json.load_instance(), which transparently handles both .tsp
    (TSPLIB) and .json instance files — the same loader used by
    run_single_instance.py / Test_main.run_tsp().

    Returns a list of (label, input_data) pairs ready for run_lp_gap_suite().
    """
    loaded = []
    for label, file_path, experiment_case in configs:
        full_label = f"{label} (Case {experiment_case})"
        try:
            input_data = load_instance(file_path)
            loaded.append((full_label, input_data))
        except Exception as e:
            print(f"[{full_label}] Failed to load '{file_path}': {e}")
    return loaded


if __name__ == "__main__":
    if not CPLEX_PATH:
        raise EnvironmentError(
            "CPLEX_PATH environment variable is not set. "
            "Set it to your local CPLEX binary path before running this script, "
            "e.g.: export CPLEX_PATH=/path/to/cplex/bin/<platform>/cplex"
        )

    if USE_AVERAGING:
        if not AVERAGED_INSTANCE_CONFIGS:
            print("USE_AVERAGING is True but AVERAGED_INSTANCE_CONFIGS is empty. "
                  "Populate it with (label, file_paths, experiment_case) entries "
                  "— use build_run_paths() to generate the Run_1..Run_N list.")
        else:
            run_lp_gap_suite_averaged(
                AVERAGED_INSTANCE_CONFIGS,
                cplex_path=CPLEX_PATH,
                save_csv_path=OUTPUT_CSV,
            )
    else:
        if not INSTANCE_CONFIGS:
            print("No instances configured. Populate INSTANCE_CONFIGS above with "
                  "(label, file_path, experiment_case) entries — file_path can "
                  "point at the same .tsp/.json instance files used by "
                  "run_single_instance.py.")
        else:
            all_instances = load_instances_from_config(INSTANCE_CONFIGS)
            run_lp_gap_suite(
                all_instances,
                cplex_path=CPLEX_PATH,
                save_csv_path=OUTPUT_CSV,
            )