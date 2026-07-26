def nearest_tsp(G, start, end, set_1, set_2, large_value=1000000):
    """Nearest Neighbor heuristic for the bipartite TSP (BTSP-Full/Single).

    Alternates between GR (set_2) and KH (set_1) nodes, moving to the nearest
    feasible unvisited node at each step. Returns the computed tour.
    """
    visit = {node: False for node in G.nodes}
    tour = [start]
    visit[start] = True
    current = start

    while len(tour) < len(set_1) + len(set_2) - 1:  # visit all nodes before the end node
        if current in set_2 and current != end:  # don't reach '0.0.0' before all KH are visited
            neighbors = [node for node in set_1 if not visit[node] and G.has_edge(current, node) and G[current][node]['weight'] < large_value]
            next_node = min(neighbors, key=lambda node: G[current][node]['weight'], default=None)
        else:  # If in set_1, choose next from set_2
            neighbors = [node for node in set_2 if not visit[node] and node != end and G.has_edge(current, node) and G[current][node]['weight'] < large_value]
            next_node = min(neighbors, key=lambda node: G[current][node]['weight'], default=None)

        if next_node is None:
            raise ValueError(f"No valid neighbors found from {current}. The tour may be incomplete.")

        tour.append(next_node)
        visit[next_node] = True
        current = next_node

        # If all kit holders are visited, allow visiting the end node (0.0.0)
        if all(visit[node] for node in set_1 if node != '0.0'):
            print("All kit holders visited. Preparing to visit 0.0.0.")
            break

    # Add the end node to the tour
    tour.append(end)

    # Add the return leg from the end node to the start node
    if G.has_edge(end, start):
        tour.append(start)
        print(f"Returning from {end} to {start}.")

    return tour

def total_cost(G, tour):
    """Total cost of a tour; returns (cost, per-segment time details)."""
    cost = 0
    time_details = []
    for i in range(len(tour) - 1):
        u, v = tour[i], tour[i + 1]
        if u in G and v in G[u]:
            weight = G[u][v]['weight']
            cost += weight
            time_details.append({
                "from": u,
                "to": v,
                "totalTime": weight
            })
    return cost, time_details

def FindBestTwoOptMoveForBipartite(top, tour, G, set_1, set_2, start_node='0.0'):
    """Find the best 2-opt edge exchange for BTSP-Full; updates `top` in place."""
    number_of_neighboring_solutions = 0

    # Iterate through all pairs of edges in the tour
    for firstIndex in range(0, len(tour) - 2):  # Start from 0 to include the starting node
        A = tour[firstIndex]
        B = tour[firstIndex + 1]

        for secondIndex in range(firstIndex + 2, len(tour) - 1):
            K = tour[secondIndex]
            L = tour[secondIndex + 1]

            # Ensure bipartite constraints are respected
            if (A in set_1 and K in set_2) or (A in set_2 and K in set_1) or A == start_node:
                number_of_neighboring_solutions += 1

                # Only proceed if the edges exist in the graph
                if G.has_edge(A, K) and G.has_edge(B, L):
                    costAdded = G[A][K]['weight'] + G[B][L]['weight']
                    costRemoved = G[A][B]['weight'] + G[K][L]['weight']

                    moveCost = costAdded - costRemoved

                    if moveCost < top['moveCost']:
                        top['moveCost'] = moveCost
                        top['positionOfFirst'] = firstIndex
                        top['positionOfSecond'] = secondIndex

def ApplyTwoOptMoveForBipartite(top, tour):
    """Apply the best 2-opt move (reverse the sub-tour between the two edges) in place."""
    modifiedSequence = []

    # Add nodes before the first edge in reverse order
    i = 0
    while i <= top['positionOfFirst']:
        modifiedSequence.append(tour[i])
        i += 1

    # Reverse the nodes between the two edges
    i = top['positionOfSecond']
    while i > top['positionOfFirst']:
        modifiedSequence.append(tour[i])
        i -= 1

    # Add nodes after the second edge
    i = top['positionOfSecond'] + 1
    while i < len(tour):
        modifiedSequence.append(tour[i])
        i += 1

    # Update the tour with the modified sequence
    for idx in range(len(tour)):
        tour[idx] = modifiedSequence[idx]  # Update original tour list with modified one

class TwoOptMove:
    def __init__(self):
        self.positionOfFirst = None
        self.positionOfSecond = None
        self.moveCost = float('inf')  # Initialize with a high value to minimize

    def Initialize(self):
        self.positionOfFirst = None
        self.positionOfSecond = None
        self.moveCost = float('inf')  # Reset the move cost to infinity before each iteration

def two_opt_for_bipartite(tour, G, set_1, set_2, max_iterations=10000):
    """2-opt local search for BTSP-Full; returns (best_tour, best_cost)."""
    best_tour = tour[:]
    best_cost, _ = total_cost(G, best_tour)  # Calculate the cost of the initial tour
    iteration = 0  # Counter to track the number of iterations

    improved = True  # Flag to check if further improvements are possible
    while improved and iteration < max_iterations:
        improved = False

        # Use a dictionary to store the 2-opt move details
        top = {
            'positionOfFirst': None,
            'positionOfSecond': None,
            'moveCost': float('inf')
        }

        # Call the modified function to find the best 2-opt move for bipartite graphs
        FindBestTwoOptMoveForBipartite(top, best_tour, G, set_1, set_2)

        if top['positionOfFirst'] is not None and top['moveCost'] < 0:
            # If a valid move is found, apply the move and continue the search
            ApplyTwoOptMoveForBipartite(top, best_tour)
            improved = True  # Found an improvement, keep searching
            best_cost = total_cost(G, best_tour)[0]  # Update best cost
        else:
            # No further improvement possible, stop the search
            break

        iteration += 1  # Increment the iteration counter

    return best_tour, best_cost


def adjNN(G, start, end, set_1, set_2, large_value=1000000):
    """Adjusted Nearest Neighbor (adjNN) for BTSP-Single/Cluster.

    Standard NN plus a deactivation guard: a GR node is chosen only if it still
    has an unvisited compatible KH (reachable_count > 0). Once all KH of a
    component type are served, its GR nodes are deactivated, which prevents
    dead ends and guarantees completeness even when |A_f| >= |B_f|. Counts are
    maintained incrementally, giving O(n^2) construction. Returns the tour.
    """
    visit = {node: False for node in G.nodes}
    tour  = [start]
    visit[start] = True
    current = start

    kh_nodes = [n for n in set_1 if n != '0.0']
    gr_nodes = [n for n in set_2 if n != end]
    gr_set = set(gr_nodes)

    # Precompute GR<->KH compatibility once, then keep running counts:
    #   reachable_count[gr] = unvisited KH that gr can still serve (deactivation guard).
    gr_out_kh = {gr: [] for gr in gr_nodes}
    kh_in_gr = {kh: [] for kh in kh_nodes}
    supplier_count = {kh: 0 for kh in kh_nodes}
    for gr in gr_nodes:
        for kh in kh_nodes:
            if G.has_edge(gr, kh) and G[gr][kh]['weight'] < large_value:
                gr_out_kh[gr].append(kh)
                kh_in_gr[kh].append(gr)
                supplier_count[kh] += 1
    reachable_count = {gr: len(gr_out_kh[gr]) for gr in gr_nodes}

    while len(tour) < len(set_1) + len(set_2) - 1:
        remaining_kh = {node for node in set_1
                        if not visit[node] and node != '0.0'}

        if current in set_2 and current != end:
            # From GR: nearest compatible unvisited KH (all same family -> ties break by distance).
            reachable_kh = [
                node for node in remaining_kh
                if G.has_edge(current, node)
                and G[current][node]['weight'] < large_value
            ]

            if not reachable_kh:
                raise ValueError(
                    f"No valid neighbors found from {current}. "
                    f"Remaining KH: {list(remaining_kh)}")

            def kh_priority(kh):
                return (supplier_count[kh], G[current][kh]['weight'])

            next_node = min(reachable_kh, key=kh_priority)

        else:
            # From KH: nearest unvisited GR that still has a reachable unvisited KH.
            neighbors = [
                node for node in set_2
                if not visit[node]
                and node != end
                and G.has_edge(current, node)
                and G[current][node]['weight'] < large_value
                and reachable_count[node] > 0
            ]
            next_node = min(neighbors,
                            key=lambda n: G[current][n]['weight'],
                            default=None)

            if next_node is None:
                raise ValueError(
                    f"No valid neighbors found from {current}. "
                    f"Remaining KH: {list(remaining_kh)}")

        tour.append(next_node)
        visit[next_node] = True

        if next_node in gr_set:
            for kh in gr_out_kh[next_node]:
                supplier_count[kh] -= 1
        else:
            for gr in kh_in_gr[next_node]:
                reachable_count[gr] -= 1

        current = next_node

        if all(visit[node] for node in set_1 if node != '0.0'):
            break

    tour.append(end)
    if G.has_edge(end, start):
        tour.append(start)

    return tour


# ── Clustered 2-opt (C2-opt) for BTSP-Cluster ────────────────────────────────
# Standard 2-opt reverses a sub-tour, which can pair a GR with a KH of an
# incompatible family. Instead, C2-opt exchanges the KH nodes served by two
# same-family GR nodes without any reversal. The move touches four edges around
# the two KH positions p and q; all replacement edges are checked via has_edge,
# so feasibility (and same-family membership) is enforced implicitly.

def FindBestClusteredTwoOptMove(top, tour, G, set_1, set_2):
    """Find the best C2-opt exchange (best pair of same-family KH to swap); updates `top`."""
    n = len(tour)
    # Membership tests inside the nested loop must be O(1); use sets (lists -> O(n^3)).
    set_1 = set(set_1)
    set_2 = set(set_2)

    for p in range(1, n - 1):
        B = tour[p]
        if B not in set_1:
            continue
        A = tour[p - 1]
        if A not in set_2:
            continue
        X = tour[p + 1] if p + 1 < n else None

        for q in range(p + 2, n - 1):
            L = tour[q]
            if L not in set_1:
                continue
            K = tour[q - 1]
            if K not in set_2:
                continue
            Y = tour[q + 1] if q + 1 < n else None

            # New edges created by exchanging B and L
            if not (G.has_edge(A, L) and G.has_edge(K, B)):
                continue
            if X is not None and not G.has_edge(L, X):
                continue
            if Y is not None and not G.has_edge(B, Y):
                continue

            costRemoved = G[A][B]['weight'] + G[K][L]['weight']
            costAdded = G[A][L]['weight'] + G[K][B]['weight']
            if X is not None:
                costRemoved += G[B][X]['weight']
                costAdded += G[L][X]['weight']
            if Y is not None:
                costRemoved += G[L][Y]['weight']
                costAdded += G[B][Y]['weight']

            moveCost = costAdded - costRemoved

            if moveCost < top['moveCost']:
                top['moveCost'] = moveCost
                top['positionOfFirst'] = p
                top['positionOfSecond'] = q

def ApplyClusteredTwoOptMove(top, tour):
    """Apply a C2-opt move: swap the two KH nodes in place (no sub-tour reversal)."""
    p = top['positionOfFirst']
    q = top['positionOfSecond']
    tour[p], tour[q] = tour[q], tour[p]

def clustered_two_opt(tour, G, set_1, set_2, max_iterations=10000):
    """C2-opt local search for BTSP-Cluster; returns (best_tour, best_cost)."""
    best_tour = tour[:]
    best_cost, _ = total_cost(G, best_tour)
    iteration = 0

    improved = True
    while improved and iteration < max_iterations:
        improved = False

        top = {
            'positionOfFirst': None,
            'positionOfSecond': None,
            'moveCost': float('inf')
        }

        FindBestClusteredTwoOptMove(top, best_tour, G, set_1, set_2)

        if top['positionOfFirst'] is not None and top['moveCost'] < 0:
            ApplyClusteredTwoOptMove(top, best_tour)
            improved = True
            best_cost = total_cost(G, best_tour)[0]
        else:
            break

        iteration += 1

    return best_tour, best_cost
