"""
convert_to_tsplib.py
────────────────────
Reads existing JSON input files from experiment directories and writes
a .tsp file in TSPLIB format alongside each one.
The original JSON files are never modified.

Usage: python convert_to_tsplib.py
"""

import os
import json
import glob

# Must match the base output directory used by Test_Env.py / run_single_instance.py.
BASE_DIR = "Experiment Results All Cases"
LARGE_PENALTY = 999999  # for non-existent edges


def json_to_tsplib(json_path):
    with open(json_path) as f:
        data = json.load(f)

    dm = data['data']['distanceMatrix']

    # Collect all nodes
    nodes = set()
    edge_dict = {}
    for entry in dm:
        edge = entry['edge'].strip('()')
        a, b = edge.split(', ')
        nodes.add(a)
        nodes.add(b)
        edge_dict[(a, b)] = entry['distance']

    # Fix node order: depot first, then GR, then KH, then end
    node_list = ['0.0']
    gr_nodes = sorted([n for n in nodes if len(n.split('.')) == 3
                       and n != '0.0.0'])
    kh_nodes = sorted([n for n in nodes if len(n.split('.')) == 2
                       and n not in ('0.0', '0.0.0')])
    node_list += gr_nodes + kh_nodes + ['0.0.0']

    n = len(node_list)
    idx = {node: i for i, node in enumerate(node_list)}

    # Build full matrix — large penalty for non-existent edges
    matrix = [[LARGE_PENALTY] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 0
    for (a, b), dist in edge_dict.items():
        if a in idx and b in idx:
            matrix[idx[a]][idx[b]] = dist

    # Derive instance name from filename
    fname = os.path.splitext(os.path.basename(json_path))[0]
    # e.g. input_GR_1x2x5_KH_2x5_Run_1
    name = fname.replace('input_', '')

    # Node type comments
    gr_n = len(gr_nodes)
    kh_n = len(kh_nodes)

    lines = []
    lines.append(f"NAME: {name}")
    lines.append(f"TYPE: TSP")
    lines.append(f"COMMENT: BTSP instance |A|={gr_n} GR nodes "
                 f"|B|={kh_n} KH nodes depot=0.0 end=0.0.0")
    lines.append(f"DIMENSION: {n}")
    lines.append("EDGE_WEIGHT_TYPE: EXPLICIT")
    lines.append("EDGE_WEIGHT_FORMAT: FULL_MATRIX")
    lines.append("NODE_COORD_TYPE: NO_COORDS")
    lines.append("EDGE_WEIGHT_SECTION")
    for row in matrix:
        lines.append(" ".join(map(str, row)))
    lines.append("EOF")

    # Write .tsp file alongside the JSON
    out_path = json_path.replace('.json', '.tsp')
    with open(out_path, 'w') as f:
        f.write("\n".join(lines))

    return out_path


def main():
    pattern = os.path.join(BASE_DIR, "**", "input_*.json")
    files = glob.glob(pattern, recursive=True)
    print(f"Found {len(files)} JSON input files.")

    converted = 0
    for json_path in files:
        try:
            out = json_to_tsplib(json_path)
            converted += 1
        except Exception as e:
            print(f"  Failed: {json_path} — {e}")

    print(f"Converted {converted}/{len(files)} files to TSPLIB format.")


if __name__ == "__main__":
    main()