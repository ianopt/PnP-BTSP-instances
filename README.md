# Robotic Pick-and-Place BTSP: Computational Experiment Pipeline

This repository contains the computational pipeline used to generate the
experimental results (Sections 5–8) reported in:

> K. Giannakos, D. Tsakoumis, G. Koronakos, S. Plitsos, P. Eirinakis,
> G. Vivo, A. Nicotera. *"Optimizing robotic Pick-and-Place operations
> and the application of RMS principles."* Submitted to the
> *International Journal of Production Research* (IJPR), under review.

The industrial dataset and system description are available separately at
[Zenodo](https://doi.org/10.5281/zenodo.17640037). This repository covers
the **computational/reproducibility pipeline only** — instance generation,
exact/heuristic/RL solution methods, and result aggregation — as a
companion to that dataset and to the industrial-framework implementation
referenced in the paper.

---

## Overview

The pipeline solves the **Bipartite Traveling Salesman Problem (BTSP)**
formulated in the paper for robotic Pick-and-Place (PnP) kitting
operations, across three compatibility-graph structures:

| Paper name | Internal case ID | Description |
|---|---|---|
| **BTSP Full** (Full Placing Node Compatibility) | Case 1 | Complete bipartite graph — every GR position compatible with every KH position |
| **BTSP Single** (Single Placing Node Compatibility) | Case 2 | Perfect one-to-one GR↔KH matching |
| **BTSP Cluster** (Cluster Placing Node Compatibility) | Case 3 | Family-based sparse compatibility, GR positions grouped by column |

Four solution methods are implemented and benchmarked:
- **Exact IP** — solved via both CBC (open-source) and CPLEX (commercial), with iterative subtour elimination (Section 5.1)
- **Nearest Neighbor (NN)** — standard construction heuristic (Section 5.2.1)
- **2-opt** — local search refinement, applicable to BTSP Full only (Section 5.2.1)
- **Adjusted Nearest Neighbor (ANN)** — MRV-priority variant of NN for BTSP Cluster (Section 5.2.2)
- **Q-learning** — tabular RL approach (Section 5.2.3)

---

## Repository structure

```
.
├── Test_Env.py                     # Main entry point — runs the full experiment suite
├── run_single_instance.py          # Run a single instance for quick testing/debugging
├── Test_main.py                    # Core orchestration: runs all requested methods on one instance
├── exact_method.py                 # Exact IP model + iterative subtour elimination (Section 5.1)
├── heuristic_methods.py            # NN, 2-opt, and ANN (Section 5.2.1 / 5.2.2)
├── reinforcement_learning_method.py # Q-learning implementation (Section 5.2.3)
├── graph_creation.py               # Builds the directed bipartite graph from a distance matrix
├── instance_generators.py          # Generates BTSP Full / Single / Cluster instances
├── parse_json.py                   # Instance I/O: JSON and TSPLIB (.tsp) loading
├── convert_to_tsplib.py            # Converts generated JSON instances to TSPLIB format
├── aggregate_results.py            # Aggregates per-run CSVs into summary Excel workbooks
├── exact_lp_relaxation_test.py     # LP relaxation / integrality gap diagnostic (see below)
└── README.md
```

---

## Requirements

- **Python 3.12** (as used in the paper; earlier 3.x versions likely work but are untested)
- **CBC solver** — bundled with PuLP, no separate installation needed
- **CPLEX 22.1.2** (optional, academic or commercial license) — required only
  for the `exact_cplex` method; the pipeline runs fully on CBC alone if
  CPLEX is not available

### Python packages

```
pip install pulp pandas numpy networkx matplotlib openpyxl
```

---

## Setup

### 1. CPLEX (optional)

If you have a CPLEX installation and want to reproduce the CBC-vs-CPLEX
comparison (Section 7), set the `CPLEX_PATH` environment variable to your
local CPLEX binary:

```bash
# macOS example
export CPLEX_PATH=/Applications/CPLEX_Studio2212/cplex/bin/arm64_osx/cplex

# Linux example
export CPLEX_PATH=/opt/ibm/ILOG/CPLEX_Studio2212/cplex/bin/x86-64_linux/cplex
```

If `CPLEX_PATH` is not set, any method list including `'exact_cplex'` will
raise a clear `EnvironmentError` rather than failing silently. Omit
`'exact_cplex'` from your methods list to run CBC-only.

### 2. Verify the install

```bash
python run_single_instance.py
```

This runs a single small BTSP Full instance through all five methods and
prints the cost/execution time for each — a quick way to confirm your
environment is set up correctly before launching the full suite.

---

## Quickstart: running a single instance

Edit the configuration block at the top of `run_single_instance.py`:

```python
INPUT_FILE  = "path/to/input_instance.tsp"   # or .json
OUTPUT_FILE = "output_single_run.json"
EXPERIMENT_CASE = 1   # 1 = BTSP Full, 2 = BTSP Single, 3 = BTSP Cluster
METHODS = ['nearest', '2-opt', 'q-learning', 'exact_cbc', 'exact_cplex']
```

Then run:

```bash
python run_single_instance.py
```

`EXPERIMENT_CASE` determines which Nearest Neighbor variant is used
(standard NN for BTSP Full/Single, ANN for BTSP Cluster) — it does not
change the instance file itself, since the compatibility structure is
already baked into which edges exist in the instance's distance matrix.

---

## Reproducing the full experiment suite

`Test_Env.py` is the main entry point used to generate the results behind
Tables 1–4 (Section 7). Edit the `CONFIGURATION` block at the top:

```python
CASES_TO_RUN = [1, 2, 3]
METHODS_PER_CASE = {
    1: ['nearest', '2-opt', 'q-learning', 'exact_cbc', 'exact_cplex'],
    2: ['nearest', '2-opt', 'q-learning', 'exact_cbc', 'exact_cplex'],
    3: ['nearest', 'exact_cbc', 'exact_cplex'],
}
N_RUNS = 10
INSTANCE_CONFIGS = [ ... ]   # 18 sizes, 10x10 through 500x500
BASE_DIR = "Experiment Results All Cases v2"
```

Then run:

```bash
python Test_Env.py
```

**Note on scale:** the full 18-size × 3-case × 10-run suite is
computationally heavy at the largest sizes (the 300×300 BTSP Single
instance alone takes on the order of 20+ minutes per run with CBC — see
Table 3 in the paper). Consider trimming `INSTANCE_CONFIGS` and `N_RUNS`
for a quick smoke test before committing to a full overnight run.

For the **unequal-set instances** (Table 2, BTSP Full only, |A| > |B|),
call `instance_generators.generate_case1()` directly with asymmetric GR
and KH position lists — `Test_Env.py`'s default configuration only covers
equal-set instances.

### Output structure

```
Experiment Results All Cases v2/
├── Case1/
│   └── GR_1x2x5_KH_2x5/
│       └── Run_1/
│           ├── input_GR_1x2x5_KH_2x5_Run_1.json
│           ├── output_GR_1x2x5_KH_2x5_Run_1.json
│           └── results_GR_1x2x5_KH_2x5_Run_1.csv
├── Case2/
└── Case3/
```

---

## Aggregating results

Once the experiment suite has completed, aggregate all per-run CSVs into
summary Excel workbooks (one per case):

```bash
python aggregate_results.py
```

This produces `Experiment Results Case1.xlsx`, `Case2.xlsx`, `Case3.xlsx`,
each with five sheets: raw data, per-instance mean/std summary, a
cost/gap/time pivot table, a paper-table-formatted view, and a CBC-vs-CPLEX
solver comparison with speedup ratios.

`BASE_BASE_DIR` in `aggregate_results.py` must match `BASE_DIR` in
`Test_Env.py` (both default to `"Experiment Results All Cases v2"`).

---

## TSPLIB format conversion

For reproducibility using standard TSP tooling, generated JSON instances
can be converted to TSPLIB format (`FULL_MATRIX` explicit-weight format):

```bash
python convert_to_tsplib.py
```

This recursively finds all `input_*.json` files under `BASE_DIR` and
writes a matching `.tsp` file alongside each one, without modifying the
original JSON. `.tsp` files can be loaded directly by `run_single_instance.py`
and `Test_Env.py` via `parse_json.load_instance()`, which auto-detects the
file format from its extension.

---

## LP relaxation / integrality gap diagnostic

`exact_lp_relaxation_test.py` is a standalone diagnostic that solves the
**root LP relaxation** of the exact IP model (Section 5.1, constraints
(2)–(5), without subtour-elimination constraint (6)) using CPLEX, and
compares it against the proven IP optimum to report the integrality gap.
This supports the discussion of solution tightness referenced in the
reviewer response letter and is not required to reproduce the paper's
main tables.

```bash
python exact_lp_relaxation_test.py
```

Requires `CPLEX_PATH` to be set. Edit `INSTANCE_CONFIGS` (single-run mode)
or `AVERAGED_INSTANCE_CONFIGS` + `USE_AVERAGING = True` (10-run averaged
mode, using `build_run_paths()`) at the bottom of the file to select which
instances to check.

---

## Instance file format

Instances are stored as JSON with the following structure:

```json
{
  "route": "robot-pick-seq-opt",
  "uuid": "...",
  "generated_at": 0,
  "data": {
    "method": "all",
    "start_node": "0.0",
    "end_node": "0.0.0",
    "distanceMatrix": [
      {"edge": "(1.1.1, 1.1)", "distance": 14532},
      ...
    ]
  }
}
```

Node naming convention:
- `0.0` — depot (fixed start node)
- `0.0.0` — return node (fixed end node)
- `row.col.component` (3-part) — GR (gravity rack / picking) positions
- `holder.block` (2-part) — KH (kit holder / placing) positions

---

## License

This repository is licensed under a Creative Commons
Attribution-NonCommercial-NoDerivs 3.0 Unported License. See the license
header in each source file, or
<http://creativecommons.org/licenses/by-nc-nd/3.0/>.

## Citation

If you use this code, please cite the paper referenced at the top of this
README. A full citation will be added here once the manuscript is
accepted.
