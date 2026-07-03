# REPOSITORY NAME (c) by the University of Piraeus, Greece.
#
# REPOSITORY NAME is licensed under a
# Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License.
#
# You should have received a copy of the license along with this
# work.  If not, see <http://creativecommons.org/licenses/by-nc-nd/3.0/>.

import pulp
import networkx as nx
import matplotlib.pyplot as plt
from parse_json import create_distance_matrices


def classify_nodes(a_to_b_matrix):
    """
    Classify nodes into set_1 (kit holders) and set_2 (gravity racks),
    excluding the special depot/return nodes '0.0' and '0.0.0'.

    Parameters:
    - a_to_b_matrix: DataFrame containing the distance matrix.

    Returns:
    - set_1: List of kit holders.
    - set_2: List of gravity racks.
    """
    set_1 = [node for node in a_to_b_matrix.index if len(node.split('.')) == 2 and node != '0.0']
    set_2 = [node for node in a_to_b_matrix.index if len(node.split('.')) == 3 and node != '0.0.0']
    return set_1, set_2


def create_distances_dict(a_to_b_matrix, set_1, set_2):
    """
    Build a dictionary of distances between node pairs from the distance matrix,
    handling the special depot ('0.0') and return ('0.0.0') nodes.

    Parameters:
    - a_to_b_matrix: DataFrame containing distances.
    - set_1: List of kit holders.
    - set_2: List of gravity racks.

    Returns:
    - distances: Dictionary mapping node pairs to distances.
    """
    distances = {}

    for point_a in set_1:
        for point_b in set_2:
            dist_a_to_b = a_to_b_matrix.at[point_a, point_b]
            dist_b_to_a = a_to_b_matrix.at[point_b, point_a]

            if point_b == '0.0.0':
                distances[(point_a, point_b)] = dist_a_to_b
                continue

            if point_a == '0.0.0' and point_b == '0.0':
                distances[(point_a, point_b)] = dist_a_to_b if dist_a_to_b != 0 else 1
                continue

            if point_a == '0.0' and point_b in set_2:
                distances[(point_a, point_b)] = dist_a_to_b
                continue

            if point_a in set_1 and point_b in set_2 and dist_a_to_b != 0:
                distances[(point_a, point_b)] = dist_a_to_b
                distances[(point_b, point_a)] = dist_b_to_a
                continue

    for point_b in set_2:
        dist_0_to_b = a_to_b_matrix.at['0.0', point_b]
        if dist_0_to_b != 0:
            distances[('0.0', point_b)] = dist_0_to_b

    for point_a in set_1:
        dist_a_to_0_0_0 = a_to_b_matrix.at[point_a, '0.0.0']
        if dist_a_to_0_0_0 != 0:
            distances[(point_a, '0.0.0')] = dist_a_to_0_0_0

    if ('0.0.0', '0.0') not in distances:
        distances[('0.0.0', '0.0')] = 1
    if ('0.0', '0.0.0') in distances:
        del distances['0.0', '0.0.0']

    return distances


def extract_possible_edges(distances):
    """
    Extract all possible edges from the distances dictionary, ensuring the
    depot/return edge convention is respected.

    Parameters:
    - distances: Dictionary of node pairs and distances.

    Returns:
    - List of possible edges.
    """
    possible_edges = [(i, j) for (i, j) in distances.keys()]

    if ('0.0', '0.0.0') in possible_edges:
        possible_edges.remove(('0.0', '0.0.0'))

    if ('0.0.0', '0.0') not in possible_edges:
        possible_edges.append(('0.0.0', '0.0'))

    return possible_edges


def check_connectivity(SetA, SetB, distances):
    """
    Check if every node in Set A and Set B has at least one connection
    to the opposite set, printing a warning for any that don't.

    Parameters:
    - SetA: List of nodes in set A.
    - SetB: List of nodes in set B.
    - distances: Dictionary of distances.
    """
    for a in SetA:
        if all((a, b) not in distances for b in SetB):
            print(f"No connections from {a} in Set A to any node in Set B")
    for b in SetB:
        if all((a, b) not in distances for a in SetA):
            print(f"No connections from {b} in Set B to any node in Set A")


def analyze_sets(set_a, set_b, distances):
    """
    Analyze basic connectivity properties of set_a and set_b. Retained as a
    lightweight safety check; extend with logging as needed.
    """
    disconnected_nodes_a = [node for node in set_a if all((node, b) not in distances for b in set_b)]
    disconnected_nodes_b = [node for node in set_b if all((a, node) not in distances for a in set_a)]


def eliminate_subtour(subtour, prob, x):
    """Add a constraint to eliminate the specific subtour."""
    constraint_name = f"subtour_elimination_{len(prob.constraints)}"
    constraint = pulp.lpSum([x[i][j] for i in subtour for j in subtour if i != j]) <= len(subtour) - 1
    prob += constraint, constraint_name
    return prob.constraints[constraint_name]


def check_edges_in_distances(possible_edges, distances):
    """Check if all edges in possible_edges exist in the distances dictionary."""
    missing_edges = []
    for edge in possible_edges:
        if edge not in possible_edges:
            missing_edges.append(edge)
    return missing_edges


def solve_tsp(distances, possible_edges, set_1, set_2, plot_initial=True, solver=None):
    """
    Solve the BTSP IP model (Section 5.1 of the paper) with constraints
    ensuring alternating connections between the two bipartite sets.
    This builds the root relaxation model (no subtour-elimination
    constraints) and is called repeatedly by iterative_subtour_elimination().

    Parameters:
    - distances: Dictionary of distances between nodes.
    - possible_edges: List of valid edges.
    - set_1: Nodes in set 1 (kit holders).
    - set_2: Nodes in set 2 (gravity racks).
    - plot_initial: Flag to print a status line after the initial solve.
    - solver: PuLP solver instance (defaults to CBC if None).

    Returns:
    - optimal_tour: List of edges in the optimal tour.
    - nodes: Full node list used in the model.
    - x: The PuLP decision-variable dictionary.
    - prob: The PuLP problem object.
    """
    nodes = list(set([key[0] for key in distances.keys()] + [key[1] for key in distances.keys()]))
    nodes.remove('0.0.0')

    check_edges_in_distances(possible_edges, distances)

    prob = pulp.LpProblem("Bipartite_TSP", pulp.LpMinimize)

    # Decision variables
    x = pulp.LpVariable.dicts("x", (nodes + ['0.0', '0.0.0'], nodes + ['0.0', '0.0.0']), cat='Binary')

    # Objective function: minimize total distance over valid edges
    prob += pulp.lpSum([distances[(i, j)] * x[i][j] for i, j in possible_edges])

    # Start at 0.0, go to set_2
    prob += pulp.lpSum([x['0.0'][j] for j in set_2 if ('0.0', j) in possible_edges]) == 1

    # Go from set_1 to 0.0.0
    prob += pulp.lpSum([x[i]['0.0.0'] for i in set_1 if (i, '0.0.0') in possible_edges]) == 1

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

    # Ensure the tour closes: 0.0.0 -> 0.0
    prob += pulp.lpSum([x['0.0.0']['0.0']]) == 1

    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    if pulp.LpStatus[prob.status] != 'Optimal':
        print("-------------------------------------------------------------------")
        print("The problem is infeasible or unbounded.")
        print("-------------------------------------------------------------------")
        return None, nodes, x, prob

    optimal_tour = [(i, j) for i, j in possible_edges if pulp.value(x[i][j]) == 1]

    if plot_initial:
        print("-------------------------------------------------------------------")

    return optimal_tour, nodes, x, prob


def find_subtours(optimal_tour, nodes, set_1, set_2):
    """
    Detects both (a) disjoint cycles and (b) any node not reachable
    from the start node '0.0' in the current solution. Both indicate
    an incomplete/invalid tour that must be cut.
    """
    graph = nx.DiGraph()
    graph.add_edges_from(optimal_tour)

    # (a) classic cycle detection
    cycles = list(nx.simple_cycles(graph))
    cycles = [c for c in cycles if len(c) < len(nodes) - abs(len(set_1) - len(set_2))]

    # (b) reachability check from '0.0' — catches orphaned/disconnected nodes
    #     that don't form a cycle but are still not part of the main tour
    if '0.0' in graph:
        reachable = nx.descendants(graph, '0.0') | {'0.0'}
    else:
        reachable = set()

    disconnected_components = []
    seen = set()
    for node in graph.nodes():
        if node in reachable or node in seen:
            continue
        component = nx.node_connected_component(graph.to_undirected(), node)
        seen |= component
        if component - reachable:
            disconnected_components.append(list(component))

    return cycles + disconnected_components


def iterative_subtour_elimination(distances, possible_edges, set_a, set_b, solver=None):
    """
    Solves the BTSP by repeatedly solving the IP relaxation, detecting
    subtours/disconnected components, adding the corresponding
    elimination constraints, and re-solving until a single connected
    tour is found (Section 5.1 of the paper).
    """
    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=False)

    optimal_tour, nodes, x, prob = solve_tsp(
        distances, possible_edges, set_a, set_b,
        plot_initial=True, solver=solver)

    if optimal_tour is None:
        return None

    subtours = find_subtours(optimal_tour, nodes, set_a, set_b)
    subtour_constraints = []

    while subtours:
        for constraint in subtour_constraints:
            prob.constraints.pop(constraint, None)

        subtour_constraints = []
        for subtour in subtours:
            subtour_constraint = pulp.lpSum([x[i][j] for i in subtour for j in subtour if i != j]) <= len(subtour) - 1
            prob += subtour_constraint
            subtour_constraints.append(subtour_constraint.name)

        prob.solve(solver)

        if pulp.LpStatus[prob.status] != 'Optimal':
            print("After subtour elimination, the problem became infeasible.")
            return None

        optimal_tour = [(i, j) for i, j in possible_edges
                        if pulp.value(x[i][j]) is not None and pulp.value(x[i][j]) > 0.5]
        subtours = find_subtours(optimal_tour, nodes, set_a, set_b)

    expected_edges = 2 * min(len(set_a), len(set_b)) + 2
    if len(optimal_tour) != expected_edges:
        print(f"WARNING: Unexpected tour length ({len(optimal_tour)} edges, "
              f"expected {expected_edges}). Discarding result.")
        return None

    return optimal_tour


def plot_tour(optimal_tour, distances, set_a, set_b, possible_edges, title):
    """Plot the optimal tour on the bipartite graph (visualization utility)."""
    B = nx.Graph()

    B.add_nodes_from(set_a, bipartite=0)
    B.add_nodes_from(set_b, bipartite=1)
    B.add_node('0.0.0')

    for (i, j) in possible_edges:
        B.add_edge(i, j)

    pos = {node: (1, i) for i, node in enumerate(set_a)}
    pos.update({node: (3, i) for i, node in enumerate(set_b)})
    pos['0.0.0'] = (2, -1)

    plt.figure(figsize=(8, 5))
    nx.draw(B, pos, with_labels=True,
            node_color=['lightblue' if node in set_a else ('lightgreen' if node in set_b else 'lightcoral') for node in
                        B.nodes()], edge_color='gray', alpha=0.5)

    nx.draw_networkx_edges(B, pos, edgelist=optimal_tour, edge_color='red', width=2, arrows=True, arrowsize=20,
                           arrowstyle='-|>')

    edge_labels = {(i, j): f"{idx + 1}\n{distances.get((i, j), 'Unknown')}" for idx, (i, j) in enumerate(optimal_tour)}
    nx.draw_networkx_edge_labels(B, pos, edge_labels=edge_labels, font_color='blue')

    plt.suptitle(f"{title}\nTotal Connections: {len(optimal_tour)}", fontsize=10)
    plt.show()


def run_exact_tsp(input_data, solver=None):
    """
    Run the exact BTSP solver from a dictionary input.

    This function:
        - Builds the distance matrix from the raw input data,
        - Classifies nodes into the two bipartite sets,
        - Builds the distances dictionary and possible edges,
        - Solves the BTSP using iterative subtour elimination,
        - Reconstructs the ordered tour and computes total cost and time details.

    Parameters:
        input_data (dict): Input data structure used by `create_distance_matrices`
                           to produce the distance matrix.
        solver: PuLP solver instance (defaults to CBC if None).

    Returns:
        tuple:
            - ordered_tour (list[tuple[str, str]] or None): Ordered sequence of edges in the tour.
            - exact_tour_cost (float or None): Total cost (distance) of the tour.
            - time_details (list[dict] or None): List of step dictionaries with keys
              "from", "to", and "distance".
    """
    try:
        if solver is None:
            solver = pulp.PULP_CBC_CMD(msg=False)
        a_to_b_matrix = create_distance_matrices(input_data)

        set_a, set_b = classify_nodes(a_to_b_matrix)
        distances = create_distances_dict(a_to_b_matrix, set_a, set_b)
        possible_edges = extract_possible_edges(distances)

        check_connectivity(set_a, set_b, distances)
        analyze_sets(set_a, set_b, distances)

        exact_tour = iterative_subtour_elimination(
            distances, possible_edges, set_a, set_b, solver=solver)

        if exact_tour:
            ordered_tour = reconstruct_tour(exact_tour, start_node='0.0')
            exact_tour_cost = sum(
                distances.get((i, j), 0) for i, j in ordered_tour)
            time_details = [{"from": i, "to": j,
                             "distance": distances.get((i, j), 'Unknown')}
                            for i, j in ordered_tour]
            return ordered_tour, exact_tour_cost, time_details
        else:
            return None, None, None
    except Exception as e:
        print(f"Error in exact method: {e}")
        return None, None, None


def reconstruct_tour(tour_edges, start_node):
    """Reconstruct an ordered tour (list of edges) from an unordered edge set."""
    nodes_seq = {}
    for i, j in tour_edges:
        nodes_seq[i] = j

    ordered_tour = []
    current_node = start_node
    visited = set()
    while True:
        if current_node in visited:
            break
        visited.add(current_node)
        next_node = nodes_seq.get(current_node)
        if next_node is None:
            break
        ordered_tour.append((current_node, next_node))
        current_node = next_node
        if current_node == start_node:
            break

    return ordered_tour