# REPOSITORY NAME (c) by the University of Piraues, Greece.
#
# REPOSITORY NAME is licensed under a
# Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License.
#
# You should have received a copy of the license along with this
# work.  If not, see <http://creativecommons.org/licenses/by-nc-nd/3.0/>.

import random
import pandas as pd

# Q-Learning Parameters
ALPHA = 0.1  # Learning rate: how much new information overrides old knowledge
GAMMA = 0.55  # Discount factor: importance of future rewards
EPSILON = 0.1  # Exploration factor: probability of exploring new actions

# Function to perform Q-learning for the bipartite TSP.
#
# Solve the bipartite TSP using Q-learning.
#
# Parameters:
# - G: Graph representing the problem.
# - start: Start node ('0.0').
# - end: End node ('0.0.0').
# - set_1: List of kit holder nodes.
# - set_2: List of gravity rack nodes.
# - large_value: Large value representing invalid connections.
# - episodes: Number of episodes for training.
#
# Returns:
# - best_tour: The best tour found using Q-values.
def q_learning_tsp(G, start, end, set_1, set_2, large_value=1000000, episodes=10000,
                    export_q_table=False, q_table_path='q_values.csv'):
    # Initialize the Q-table: Each node has a dictionary of its neighbors with Q-values
    q_table = {node: {neighbor: 0 for neighbor in G.neighbors(node)} for node in G.nodes}

    for episode in range(episodes):  # Run for the specified number of episodes
        current_node = start
        tour = [current_node]
        visit = {node: False for node in G.nodes}  # Track visited nodes
        visit[start] = True

        while len(tour) < len(set_1) + len(set_2):  # Ensure all nodes are visited before 0.0.0
            # Alternate between set_1 and set_2 strictly
            if current_node in set_2 and current_node != end:   # From set_2, move to set_1
                valid_neighbors = [node for node in set_1 if
                                   not visit[node] and G.has_edge(current_node, node) and G[current_node][node][
                                       'weight'] < large_value]
            else:  # From set_1, move to set_2
                valid_neighbors = [node for node in set_2 if
                                   not visit[node] and node != end and G.has_edge(current_node, node) and
                                   G[current_node][node]['weight'] < large_value]

            # Choose the next node based on exploration or exploitation
            if random.uniform(0, 1) < EPSILON: # Explore
                next_node = random.choice(valid_neighbors) if valid_neighbors else None
            else: # Exploit
                next_node = max(((n, q_table[current_node][n]) for n in valid_neighbors), key=lambda x: x[1], default=(None, None))[0]

            if next_node is None:   # If no valid neighbors are found
                break
                # raise ValueError(f"No valid neighbors found from {current_node}. The tour may be incomplete.")

            # Check if all kit holders (set_1) and gravity racks (set_2) have been visited
            if all(visit[node] for node in set_1) and all(visit[node] for node in set_2):
                next_node = end  # Force visit to 0.0.0
                visit[end] = True

            # Compute the reward and update the Q-value
            reward = get_reward(G, current_node, next_node, set_1, set_2, large_value)
            max_next_q = max(q_table[next_node].values(), default=0)   # Max Q-value of the next state
            q_table[current_node][next_node] = (1 - ALPHA) * q_table[current_node][next_node] + \
                                               ALPHA * (reward + GAMMA * max_next_q)

            current_node = next_node
            tour.append(next_node)
            visit[next_node] = True

        # After visiting all nodes, return to the start node
        tour.append(start)

        # Calculate the total cost of the current tour
        cost, _ = total_costRL(G, tour)

        # Print the current tour and its cost
        # print(f"Episode {episode + 1}, Tour: {tour}, Total Cost: {cost}")

    # Generate the best tour from the learned Q-values
    best_tour = best_q_tour(q_table, G, start, end, set_1, set_2, large_value)
    best_tour_cost, _ = total_costRL(G, best_tour)
    # print(f"Best Q-Learning Tour: {best_tour}, Total Cost: {best_tour_cost}")

    # Optionally export Q-values to a CSV file (off by default — avoid
    # writing a file to disk on every call during batch experiment runs)
    if export_q_table:
        q_df = pd.DataFrame.from_dict(q_table, orient='index').fillna(float('inf'))  # Use inf for non-visited
        q_df.to_csv(q_table_path, index=True)

    return best_tour

# Function to calculate rewards during Q-learning.
#
# Calculate the reward for moving from current to next_node.
#
# Parameters:
# - G: Graph representing the problem.
# - current: Current node.
# - next_node: Next node.
# - set_1: List of kit holder nodes.
# - set_2: List of gravity rack nodes.
# - large_value: Large value representing invalid connections.
#
# Returns:
# - Reward value for the transition.
def get_reward(G, current, next_node, set_1, set_2, large_value):
    if current in set_1 and next_node in set_2:  # Valid transition from set_1 to set_2
        weight = G[current][next_node]['weight']
        if weight >= large_value:
            return -float('inf')
        return -weight
    elif current in set_2 and next_node in set_1:
        weight = G[current][next_node]['weight']
        if weight >= large_value:
            return -float('inf')  # Discourage transitions with large weights
        return -weight
    else:
        return -float('inf')  # Invalid transition

# Function to extract the best tour based on Q-values.
#
# Generate the best tour based on the learned Q-values.
#
# Parameters:
# - q_table: Learned Q-values for each state-action pair.
# - G: Graph representing the problem.
# - start: Start node.
# - end: End node.
# - set_1: List of kit holder nodes.
# - set_2: List of gravity rack nodes.
# - large_value: Large value representing invalid connections.
#
# Returns:
# - tour: List of nodes representing the best tour.
def best_q_tour(q_table, G, start, end, set_1, set_2, large_value):
    tour = [start]
    current = start
    visit = {node: False for node in q_table}
    visit[start] = True

    while len(tour) < len(set_1) + len(set_2):  # Ensure we visit all nodes before 0.0.0
        if current in set_2 and current != end:  # Gravity rack nodes
            valid_neighbors = [node for node in set_1 if
                               not visit[node] and G.has_edge(current, node) and G[current][node][
                                   'weight'] < large_value]
        else:  # Kit holder nodes
            valid_neighbors = [node for node in set_2 if
                               not visit[node] and node != end and G.has_edge(current, node) and G[current][node][
                                   'weight'] < large_value]

        next_node = max(((n, q_table[current][n]) for n in valid_neighbors), key=lambda x: x[1], default=(None, None))[
            0]

        if next_node is None:
            # print(f"No valid neighbors found from {current}. Ending tour early.")
            break

        tour.append(next_node)
        visit[next_node] = True
        current = next_node

        # print(f"Tour so far: {tour}, Current node: {current}")

        # Check if all kit holders are visited, then move to 0.0.0
        if all(visit[node] for node in set_1 if node != start):
            # print("All kit holders visited. Moving to 0.0.0.")
            tour.append(end)
            visit[end] = True
            break

    # After visiting 0.0.0, return to 0.0
    tour.append(start)
    return tour

# Function to calculate the total cost of a tour.
#
# Calculate the total cost of a tour.
#
# Parameters:
# - G: Graph representing the problem.
# - tour: List of nodes in the tour.
#
# Returns:
# - cost: Total cost of the tour.
# - time_details: List of details for each segment of the tour.
def total_costRL(G, tour):
    cost = 0
    time_details = []
    for i in range(len(tour) - 1):
        u, v = tour[i], tour[i + 1]
        if u in G and v in G[u]:
            weight = G[u][v]['weight']
            # Debugging line to verify the edges and their weights
            # print(f"Edge {u} -> {v}, Weight: {weight}")
            cost += weight
            time_details.append({
                "from": u,
                "to": v,
                "totalTime": weight
            })
    # Debugging line to check the final cost
    # print(f"Total cost for tour {tour}: {cost}")
    return cost, time_details