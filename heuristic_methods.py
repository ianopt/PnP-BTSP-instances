def nearest_tsp(G, start, end, set_1, set_2, large_value=1000000):
    """
    Function to implement the nearest neighbor heuristic for the bipartite TSP.

    Solve the TSP using the nearest neighbor heuristic, maintaining bipartite constraints.

    Parameters:
    - G: A directed graph representing the problem.
    - start: The starting node ('0.0').
    - end: The end node ('0.0.0').
    - set_1: List of kit holder nodes.
    - set_2: List of gravity rack nodes.
    - large_value: A large value to represent unconnected nodes.

    Returns:
    - tour: List of nodes representing the computed tour.
    """
    visit = {node: False for node in G.nodes}
    tour = [start]
    visit[start] = True
    current = start

    # print(f"set_1 (kit holders): {set_1}")
    # print(f"set_2 (gravity racks and start/end node): {set_2}")

    while len(tour) < len(set_1) + len(set_2) - 1:  # Ensure we visit all nodes before adding the end node
        if current in set_2 and current != end:  # Ensure we don't move to `0.0.0` before all kit holders are visited
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

        # Debugging information
        # print(f"Added node: {next_node}, Tour so far: {tour}")

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
    """
    Function to calculate the total cost of a given tour.

    Calculate the total cost of a tour.

    Parameters:
    - G: A graph representing the problem.
    - tour: List of nodes representing the tour.

    Returns:
    - cost: Total cost of the tour.
    - time_details: List of details for each segment of the tour.
    """
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
    """
    Function to find the best 2-opt move for a bipartite graph.

    Find the best 2-opt move for the bipartite TSP.

    Parameters:
    - top: Dictionary to store the best move details.
    - tour: Current tour as a list of nodes.
    - G: Graph representing the problem.
    - set_1: List of kit holder nodes.
    - set_2: List of gravity rack nodes.
    - start_node: The start node ('0.0').

    Updates:
    - top: Updates the best move details if a better move is found.
    """
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
                    # else:
                        # Add the detailed debugging output here
                        # print(f"Edge missing between {A}->{K} or {B}->{L}")
                        # print(f"Current tour: {tour}")
                        # if G.has_edge(A, K):
                        #     print(f"Distance from {A} to {K}: {G[A][K]['weight']}")
                        # else:
                        #     print(f"Distance from {A} to {K} is missing.")
                        # if G.has_edge(B, L):
                        #     print(f"Distance from {B} to {L}: {G[B][L]['weight']}")
                        # else:
                        #     print(f"Distance from {B} to {L} is missing.")

    # # Additional debugging information for starting node consideration
    # if start_node in tour:
    #     print(f"Swaps including the start node {start_node} are considered.")
    # else:
    #     print(f"Start node {start_node} is not in the current tour.")

def ApplyTwoOptMoveForBipartite(top, tour):
    """
    Function to apply the best 2-opt move to the tour.

    Apply the best 2-opt move to the tour.

    Parameters:
    - top: Dictionary containing details of the best 2-opt move.
    - tour: Current tour as a list of nodes.

    Modifies:
    - tour: Updates the tour to include the 2-opt move.
    """
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
    """
    Function to perform the 2-opt meta-heuristic for the bipartite TSP.

    Perform the 2-opt meta-heuristic for a bipartite TSP.

    Parameters:
    - tour: Initial tour as a list of nodes.
    - G: Graph representing the problem.
    - set_1: List of kit holder nodes.
    - set_2: List of gravity rack nodes.
    - max_iterations: Maximum number of iterations for the heuristic.

    Returns:
    - best_tour: The best tour found.
    - best_cost: The cost of the best tour.
    """
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

    # print(f"2-opt terminated after {iteration} iterations.")
    return best_tour, best_cost


# ── ADD THIS FUNCTION TO heuristic_methods.py ─────────────────────────────────

def nearest_tsp_case3(G, start, end, set_1, set_2, large_value=1000000):
    """
    Priority-aware NN for Case 3.
    When choosing next KH from a GR, prioritizes KH nodes with
    fewest remaining compatible GR suppliers (most constrained first).
    When choosing next GR from a KH, picks nearest GR that can
    supply at least one remaining unvisited KH.
    """
    visit = {node: False for node in G.nodes}
    tour  = [start]
    visit[start] = True
    current = start

    while len(tour) < len(set_1) + len(set_2) - 1:
        remaining_kh = {node for node in set_1
                        if not visit[node] and node != '0.0'}

        if current in set_2 and current != end:
            # From GR: go to the most constrained reachable KH
            # (fewest remaining GR suppliers — break ties by distance)
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
                # Count how many unvisited GR can still supply this KH
                remaining_gr = {n for n in set_2
                                if not visit[n] and n != end}
                supplier_count = sum(
                    1 for gr in remaining_gr
                    if G.has_edge(gr, kh)
                    and G[gr][kh]['weight'] < large_value
                )
                dist = G[current][kh]['weight']
                return (supplier_count, dist)  # most constrained first

            next_node = min(reachable_kh, key=kh_priority)

        else:
            # From KH: go to nearest GR that can supply
            # at least one remaining unvisited KH
            neighbors = [
                node for node in set_2
                if not visit[node]
                and node != end
                and G.has_edge(current, node)
                and G[current][node]['weight'] < large_value
                and any(G.has_edge(node, kh)
                        and G[node][kh]['weight'] < large_value
                        for kh in remaining_kh)
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
        current = next_node

        if all(visit[node] for node in set_1 if node != '0.0'):
            break

    tour.append(end)
    if G.has_edge(end, start):
        tour.append(start)

    return tour