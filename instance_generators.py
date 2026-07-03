"""
instance_generators.py
──────────────────────
Three instance generators for BTSP experiments.

Case 1: Full graph, one component type, symmetric distances (baseline)
Case 2: Each KH needs a unique component — perfect 1-to-1 GR→KH mapping
Case 3: Component types can repeat — many-to-many sparse GR→KH mapping

Cost model (ALL cases):
- Bipartite edges (GR↔KH): uniform random in [10000, 20000] ms
- Depot edges (0.0→GR, KH→0.0.0): uniform random in [5000, 15000] ms
- Return edge (0.0.0→0.0): fixed cost = 1
- Distances are ALWAYS symmetric (same value both directions where edge exists)

Graph structure:
- 0.0  → any GR : always fully connected
- GR   → KH     : depends on case (full / 1-to-1 / component-type match)
- KH   → any GR : always fully connected (robot free to fetch next component)
- KH   → 0.0.0  : always fully connected
"""

import random
import uuid
from time import time
from collections import Counter


# ── Shared helpers ────────────────────────────────────────────────────────────

def _build_json(distance_matrix):
    """Wrap a distance matrix list into the standard JSON input format."""
    return {
        "route": "robot-pick-seq-opt",
        "uuid": str(uuid.uuid4()),
        "generated_at": int(time()),
        "data": {
            "method": "all",
            "start_node": "0.0",
            "end_node": "0.0.0",
            "distanceMatrix": distance_matrix
        }
    }


def _depot_edges(gr_positions, kh_positions):
    """
    Standard depot edges common to all cases.
    Uses uniform random distances consistent with the paper's cost model.
    """
    edges = []

    # 0.0 → all GR (fully connected)
    for gr in gr_positions:
        edges.append({
            "edge": f"(0.0, {gr})",
            "distance": random.randint(5000, 15000)
        })

    # all KH → 0.0.0 (fully connected)
    for kh in kh_positions:
        edges.append({
            "edge": f"({kh}, 0.0.0)",
            "distance": random.randint(5000, 15000)
        })

    # 0.0.0 → 0.0 (fixed return cost)
    edges.append({"edge": "(0.0.0, 0.0)", "distance": 1})

    return edges


def _assign_component_types(gr_positions, kh_positions, distribution):
    """
    Assign component types to GR and KH positions ensuring:
    - Every type needed by KH exists in at least one GR
    - Every GR type exists in at least one KH
    (no orphaned nodes that can never participate in a valid tour)

    distribution: dict {type_label: fraction}, fractions sum to 1.0
    """
    types = list(distribution.keys())
    weights = list(distribution.values())

    # Assign KH types first
    kh_types = {kh: random.choices(types, weights=weights)[0]
                for kh in kh_positions}

    # Find which types are actually needed by KH
    needed_types = list(set(kh_types.values()))

    # Shuffle GR positions for random coverage assignment
    shuffled_gr = list(gr_positions)
    random.shuffle(shuffled_gr)

    gr_types = {}

    # First pass: guarantee at least one GR per needed KH type
    for i, t in enumerate(needed_types):
        if i < len(shuffled_gr):
            gr_types[shuffled_gr[i]] = t

    # Second pass: remaining GR positions get random types
    # but ONLY from types that KH actually needs (no orphaned GR types)
    needed_weights = [distribution.get(t, 1.0) for t in needed_types]
    for gr in shuffled_gr[len(needed_types):]:
        gr_types[gr] = random.choices(needed_types, weights=needed_weights)[0]

    return gr_types, kh_types


# ── Case 1: Full graph, one component type ────────────────────────────────────

def generate_case1(gravity_rack_positions=None, kit_holder_positions=None):
    """
    Case 1 — Full graph, single component type, symmetric distances.

    Every GR can supply every KH (complete bipartite graph).
    GR→KH and KH→GR share the same randomly drawn distance (symmetric).
    This is the current baseline used in the paper (Section 7).
    """
    if gravity_rack_positions is None:
        gravity_rack_positions = [f"1.1.{i}" for i in range(1, 11)]
    if kit_holder_positions is None:
        kit_holder_positions = [f"1.{i}" for i in range(1, 11)]

    distance_matrix = []

    for gr in gravity_rack_positions:
        for kh in kit_holder_positions:
            dist = random.randint(10000, 20000)
            distance_matrix.append({"edge": f"({gr}, {kh})", "distance": dist})
            distance_matrix.append({"edge": f"({kh}, {gr})", "distance": dist})

    distance_matrix += _depot_edges(gravity_rack_positions, kit_holder_positions)
    return _build_json(distance_matrix)


# ── Case 2: Unique components — perfect 1-to-1 mapping ───────────────────────

def generate_case2(gravity_rack_positions=None, kit_holder_positions=None):
    """
    Case 2 — Each KH needs a unique component type.

    GR_i can ONLY supply KH_i (perfect 1-to-1 compatibility constraint).
    KH_i can travel back to ANY GR (robot free to fetch next component).
    GR→KH and KH→GR share the same randomly drawn distance (symmetric).

    Requires |GR| == |KH|.
    """
    if gravity_rack_positions is None:
        gravity_rack_positions = [f"1.1.{i}" for i in range(1, 11)]
    if kit_holder_positions is None:
        kit_holder_positions = [f"1.{i}" for i in range(1, 11)]

    assert len(gravity_rack_positions) == len(kit_holder_positions), \
        "Case 2 requires equal sets: one unique component per KH position."

    # Perfect 1-to-1 mapping: GR_i → KH_i (index-based pairing)
    compatible = {gr: kh for gr, kh in
                  zip(gravity_rack_positions, kit_holder_positions)}

    distance_matrix = []

    for gr in gravity_rack_positions:
        for kh in kit_holder_positions:
            dist = random.randint(10000, 20000)

            # GR → KH: only to its compatible KH (sparse, 1 edge per GR)
            if kh == compatible[gr]:
                distance_matrix.append({"edge": f"({gr}, {kh})", "distance": dist})

            # KH → GR: always exists — symmetric distance
            distance_matrix.append({"edge": f"({kh}, {gr})", "distance": dist})

    distance_matrix += _depot_edges(gravity_rack_positions, kit_holder_positions)
    return _build_json(distance_matrix)


# ── Case 3: Repeated component types — many-to-many sparse mapping ────────────

def generate_case3(gravity_rack_positions=None, kit_holder_positions=None,
                   component_distribution=None):
    """
    Component families are determined by GR column count.
    Number of families = number of distinct columns in GR positions.
    KH types assigned randomly, guaranteed at least one KH per family.
    """
    if gravity_rack_positions is None:
        gravity_rack_positions = [f"1.1.{i}" for i in range(1, 11)]
    if kit_holder_positions is None:
        kit_holder_positions = [f"1.{i}" for i in range(1, 11)]

    # ── Detect families from GR column structure ───────────────────────────
    # col number → family label (1→'A', 2→'B', ..., 26→'Z', 27→'AA', etc.)
    def col_to_family(col_num):
        """Convert column number to family label: 1→A, 2→B, ..., 26→Z, 27→AA"""
        result = ""
        while col_num > 0:
            col_num, remainder = divmod(col_num - 1, 26)
            result = chr(65 + remainder) + result
        return result

    # Get unique columns from GR positions
    gr_cols = sorted(set(int(gr.split('.')[1])
                         for gr in gravity_rack_positions))
    families = [col_to_family(c) for c in gr_cols]
    n_families = len(families)

    # GR type: deterministic by column
    gr_types = {
        gr: col_to_family(int(gr.split('.')[1]))
        for gr in gravity_rack_positions
    }

    # ── Distribution over families ─────────────────────────────────────────
    if component_distribution is None:
        # Equal split across all detected families
        weights = [1.0 / n_families] * n_families
    else:
        # Use provided distribution — keys must match family labels
        weights = [component_distribution.get(f, 1.0/n_families)
                   for f in families]

    # ── KH type assignment ─────────────────────────────────────────────────────
    # Each family can supply at most as many KH as it has GR positions.
    # This guarantees the Hamiltonian path is always feasible.
    gr_family_counts = Counter(gr_types.values())

    # Build assignment pool: family X appears exactly gr_count[X] times
    assignment_pool = []
    for family in families:
        assignment_pool.extend([family] * gr_family_counts[family])

    # Validate: enough GR slots to cover all KH positions
    if len(kit_holder_positions) > len(assignment_pool):
        raise ValueError(
            f"Cannot generate Case 3 instance: {len(kit_holder_positions)} KH positions "
            f"but only {len(assignment_pool)} total GR slots across all families. "
            f"Increase GR count or reduce KH count."
        )

    # Shuffle and assign — each KH gets one family slot
    random.shuffle(assignment_pool)
    kh_list = list(kit_holder_positions)
    random.shuffle(kh_list)
    kh_types = {kh: family for kh, family in zip(kh_list, assignment_pool)}

    # ── Build distance matrix ──────────────────────────────────────────────
    distance_matrix = []
    for gr in gravity_rack_positions:
        for kh in kit_holder_positions:
            dist = random.randint(10000, 20000)
            if gr_types[gr] == kh_types[kh]:
                distance_matrix.append({"edge": f"({gr}, {kh})", "distance": dist})
            distance_matrix.append({"edge": f"({kh}, {gr})", "distance": dist})

    distance_matrix += _depot_edges(gravity_rack_positions, kit_holder_positions)

    result = _build_json(distance_matrix)
    result['data']['component_types'] = {
        'gr': gr_types,
        'kh': kh_types,
        'families': families,
        'gr_type_counts': dict(Counter(gr_types.values())),
        'kh_type_counts': dict(Counter(kh_types.values()))
    }
    return result

# ── Feasibility check ─────────────────────────────────────────────────────────

def is_feasible(input_json):
    """
    Check that every KH node has at least one incoming GR→KH edge.
    Returns True if feasible, False otherwise.

    Note: Case 3 uses _assign_component_types which guarantees feasibility
    by construction, but this check is kept as a safety net.
    """
    kh_has_supply = set()
    kh_nodes = set()

    for entry in input_json['data']['distanceMatrix']:
        edge = entry['edge'].strip('()')
        a, b = edge.split(', ')
        a_is_gr = len(a.split('.')) == 3
        b_is_kh = len(b.split('.')) == 2 and b not in ('0.0', '0.0.0')
        b_is_end = b == '0.0.0'

        if b_is_kh or b_is_end:
            if b_is_kh:
                kh_nodes.add(b)
            if a_is_gr and b_is_kh:
                kh_has_supply.add(b)

    # also collect KH nodes from KH→0.0.0 edges
    for entry in input_json['data']['distanceMatrix']:
        edge = entry['edge'].strip('()')
        a, b = edge.split(', ')
        if b == '0.0.0' and len(a.split('.')) == 2 and a != '0.0':
            kh_nodes.add(a)

    missing = kh_nodes - kh_has_supply
    if missing:
        print(f"  Infeasible: {len(missing)} KH node(s) have no GR supply: "
              f"{sorted(missing)[:5]}{'...' if len(missing)>5 else ''}")
        return False
    return True