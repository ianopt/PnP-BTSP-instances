import os
import pandas as pd
import glob

# ── Configuration ──────────────────────────────────────────────────────────────
# Must match the base output directory used when generating results
# (see Test_Env.py's BASE_DIR). Update if you used a different folder name.
BASE_BASE_DIR = "Experiment Results All Cases v2"
CASES = [1, 2, 3]  # 1 = BTSP Full, 2 = BTSP Single, 3 = BTSP Cluster
# ──────────────────────────────────────────────────────────────────────────────

def collect_all_csvs(base_dir):
    """Recursively collect all results_*.csv files under base_dir."""
    pattern = os.path.join(base_dir, "**", "results_*.csv")
    files = glob.glob(pattern, recursive=True)
    print(f"Found {len(files)} CSV files.")
    return files


def load_and_concat(files):
    """Load all CSVs and concatenate into a single DataFrame."""
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"  Skipping {f}: {e}")
    if not dfs:
        raise ValueError("No valid CSV files found.")
    return pd.concat(dfs, ignore_index=True)


def compute_summary(df):
    """
    For each (Gravity Rack, Kit Holder, Method) combination,
    compute mean and std over all runs.
    """
    summary = (
        df.groupby(["Gravity Rack", "Kit Holder", "Method"])
        .agg(
            Runs=("Run", "count"),
            Mean_Cost=("Cost", "mean"),
            Std_Cost=("Cost", "std"),
            Mean_ExecTime_ms=("Exec time", "mean"),
            Std_ExecTime_ms=("Exec time", "std"),
            Mean_Gap=("Optimality Gap (%)", "mean"),
            Std_Gap=("Optimality Gap (%)", "std"),
        )
        .round(4)
        .reset_index()
    )
    return summary


def compute_instance_summary(df):
    """
    Pivot table: one row per (GR, KH, Run), one column per method,
    showing cost and gap side by side — useful for Table 1 / Table 2 format.
    """
    pivot_cost = df.pivot_table(
        index=["Gravity Rack", "Kit Holder", "Run"],
        columns="Method",
        values="Cost",
        aggfunc="first"
    ).round(2)

    pivot_gap = df.pivot_table(
        index=["Gravity Rack", "Kit Holder", "Run"],
        columns="Method",
        values="Optimality Gap (%)",
        aggfunc="first"
    ).round(4)

    pivot_time = df.pivot_table(
        index=["Gravity Rack", "Kit Holder", "Run"],
        columns="Method",
        values="Exec time",
        aggfunc="first"
    ).round(4)

    # Rename columns to avoid clashes when merging
    pivot_cost.columns = [f"Cost_{m}" for m in pivot_cost.columns]
    pivot_gap.columns  = [f"Gap_{m}"  for m in pivot_gap.columns]
    pivot_time.columns = [f"Time_{m}" for m in pivot_time.columns]

    combined = pd.concat([pivot_cost, pivot_gap, pivot_time], axis=1).reset_index()
    return combined


def compute_paper_table(df):
    rows = []
    for (gr, kh), grp in df.groupby(["Gravity Rack", "Kit Holder"]):
        row = {"Instance": f"|A|={gr}, |B|={kh}"}

        # Find exact reference — try all naming variants
        exact = grp[grp["Method"] == "Exact_cbc"]
        if exact.empty:
            exact = grp[grp["Method"] == "Exact_CBC"]
        if exact.empty:
            exact = grp[grp["Method"] == "Exact"]

        # Set exact time only if found
        row["Exact_ms"] = round(exact["Exec time"].mean(), 2) if not exact.empty else None

        for m in ["Nearest", "2-opt", "Q-learning"]:
            sub = grp[grp["Method"] == m]
            row[f"{m}_Gap_%"]   = round(sub["Optimality Gap (%)"].mean(), 2) if not sub.empty else None
            row[f"{m}_Time_ms"] = round(sub["Exec time"].mean(), 2)          if not sub.empty else None

        rows.append(row)
    return pd.DataFrame(rows)


def compute_solver_comparison(df):
    """Extra sheet: CBC vs CPLEX timing per instance with speedup."""
    solvers = [m for m in df['Method'].unique()
               if m.startswith('Exact')]
    rows = []
    for (gr, kh), grp in df.groupby(["Gravity Rack", "Kit Holder"]):
        row = {"Instance": f"{gr} x {kh}"}
        times = {}
        for s in solvers:
            sub = grp[grp["Method"] == s]
            if not sub.empty:
                mean_t = round(sub["Exec time"].mean(), 2)
                row[f"{s}_ms"] = mean_t
                times[s] = mean_t

        # Speedup: CBC / CPLEX
        cbc_key   = next((k for k in times if 'cbc'   in k.lower()), None)
        cplex_key = next((k for k in times if 'cplex' in k.lower()), None)
        if cbc_key and cplex_key and times[cplex_key] > 0:
            row["Speedup_CPLEX_vs_CBC"] = round(times[cbc_key] / times[cplex_key], 2)

        # Also verify costs match
        costs = {}
        for s in solvers:
            sub = grp[grp["Method"] == s]
            if not sub.empty:
                costs[s] = round(sub["Cost"].mean(), 0)
        if cbc_key and cplex_key:
            row["Cost_match"] = "✓" if costs.get(cbc_key) == costs.get(cplex_key) else "✗"

        rows.append(row)
    return pd.DataFrame(rows)


def write_excel(raw_df, summary_df, pivot_df, paper_df, output_path):
    """Write all views to separate sheets in one Excel file."""
    solver_df = compute_solver_comparison(raw_df)  # ← add this line
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        raw_df.to_excel(writer, sheet_name="Raw Data", index=False)
        summary_df.to_excel(writer, sheet_name="Summary (mean+std)", index=False)
        pivot_df.to_excel(writer, sheet_name="Pivot (cost+gap+time)", index=False)
        paper_df.to_excel(writer, sheet_name="Paper Table Format", index=False)
        solver_df.to_excel(writer, sheet_name="Solver Comparison", index=False)  # ← add this line
    print(f"\nResults saved to: {output_path}")
    print(f"Sheets: Raw Data | Summary (mean+std) | Pivot | Paper Table Format | Solver Comparison")

def main():
    for case in CASES:
        base_dir = os.path.join(BASE_BASE_DIR, f"Case{case}")
        output_excel = f"Experiment Results Case{case}.xlsx"

        print(f"\n{'='*50}")
        print(f"  Processing Case {case} from: {base_dir}")
        print(f"{'='*50}")

        if not os.path.exists(base_dir):
            print(f"  Directory not found, skipping: {base_dir}")
            continue

        files = collect_all_csvs(base_dir)
        if not files:
            print(f"  No CSV files found in {base_dir}, skipping.")
            continue

        raw_df = load_and_concat(files)

        print(f"Total rows loaded: {len(raw_df)}")
        print(f"Columns: {list(raw_df.columns)}")
        print(f"Methods found: {raw_df['Method'].unique()}")
        print(f"Instance sizes: {raw_df[['Gravity Rack','Kit Holder']].drop_duplicates().values}")

        summary_df = compute_summary(raw_df)
        pivot_df   = compute_instance_summary(raw_df)
        paper_df   = compute_paper_table(raw_df)

        write_excel(raw_df, summary_df, pivot_df, paper_df, output_excel)
        print(f"  → Written: {output_excel}")


if __name__ == "__main__":
    main()