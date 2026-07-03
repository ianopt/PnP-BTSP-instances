"""
run_single_instance.py
──────────────────────
Run a single instance (JSON or TSPLIB .tsp) through the pipeline.
Edit the configuration below and run:
    python run_single_instance.py

Requires CPLEX_PATH to be set (see README.md) if 'exact_cplex' is in METHODS.
"""

from test_main import run_tsp

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_FILE  = "Experiment Results All Cases v2/Case1/GR_1x2x5_KH_2x5/Run_1/input_GR_1x2x5_KH_2x5_Run_1.tsp"   # or .json
OUTPUT_FILE = "output_single_run.json"

EXPERIMENT_CASE = 1   # 1 = BTSP Full, 2 = BTSP Single, 3 = BTSP Cluster

METHODS = [
    'nearest',
    '2-opt',
    'q-learning',
    'exact_cbc',
    'exact_cplex',
]
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_tsp(
        json_file_path=INPUT_FILE,
        output_json_file_path=OUTPUT_FILE,
        experiment_case=EXPERIMENT_CASE,
        methods_to_run=METHODS
    )

    print("\n── Results ──────────────────────────────────────────────────────────")
    for method, data in results.items():
        print(f"  {method:<20} cost={data['cost']}   time={data['exec_time']:.2f}ms")