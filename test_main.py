import json
import traceback
from datetime import datetime
from time import time
import pandas as pd
import numpy as np
import os
from parse_json import *
from graph_creation import *
from heuristic_methods import *
from reinforcement_learning_method import *
from exact_method import *
import pulp

CPLEX_PATH = '/Users/konstantinosgiannakos/Applications/CPLEX_Studio2212/cplex/bin/arm64_osx/cplex'

import uuid, tempfile

def make_cplex_solver():
    return pulp.CPLEX_CMD(
        path=CPLEX_PATH, msg=False, gapRel=0, timeLimit=None
    )

def make_cbc_solver():
    return pulp.PULP_CBC_CMD(msg=False, gapRel=0)

def convert_to_native_types(data):
    if isinstance(data, dict):
        return {k: convert_to_native_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_native_types(v) for v in data]
    elif isinstance(data, pd.DataFrame):
        return data.applymap(lambda x: int(x) if isinstance(x, (np.integer, np.int32, np.int64)) else x).to_dict()
    elif isinstance(data, (np.integer, np.int32, np.int64)):
        return int(data)
    else:
        return data

def run_tsp(json_file_path, output_json_file_path, gravity_rack_positions=None, kit_holder_positions=None, experiment_case=1, methods_to_run=None):
    total_time_start = int(time() * 1000)
    total_loading_time = None  # Initialize here to avoid UnboundLocalError

    # # Step 1: Generate a new random JSON instance if generate_new_instance is True
    # if generate_new_instance:
    #     random_json_input = generate_random_json_input()  # Assuming this function creates a valid JSON structure
    #     json_file_path = "random_json_input.json"  # Path to save the new random JSON file
    #     with open(json_file_path, 'w') as json_file:
    #         json.dump(random_json_input, json_file, indent=4)
    #     print(f"New random JSON input has been generated and saved to {json_file_path}.")
    #
    #     # Load this new input for further processing
    #     input_data = random_json_input
    # else:
    #     # Load input from file if it's not set to generate a new instance
    #     if json_file_path:
    #         with open(json_file_path, 'r') as f:
    #             input_data = json.load(f)
    #         print(f"Local JSON input has been loaded from {json_file_path}.")
    #     elif input_data:
    #         print("Remote input data received.")
    #     else:
    #         raise ValueError("Input data is required, either via JSON file or directly.")

    if methods_to_run is None:
        methods_to_run = ['nearest', '2-opt', 'q-learning',
                          'exact_cbc', 'exact_cplex']

    # # Load the input data
    # with open(json_file_path, 'r') as json_file:
    #     input_data = json.load(json_file)

    input_data = load_instance(json_file_path)

    # Parse the distance matrices
    a_to_b_matrix = create_distance_matrices(input_data)
    # print(a_to_b_matrix)
    # print("Distance matrices created!")

    # Create the bipartite graph
    B, set_1, set_2 = create_directed_bipartite_graph(a_to_b_matrix)
    # print("Graph generated!")

    start_node = input_data['data'].get('start_node', '0.0')
    end_node = input_data['data'].get('end_node', '0.0.0')
    method = input_data['data']['method']

    if end_node not in set_2:
        set_2.append(end_node)

    results = {}
    solution_time_start = int(time() * 1000)
    simple_tour = None


    # print("COMMENT METHODS FOR TESTING EXACTS")
    if 'nearest' in methods_to_run:
        try:
            print("Running Nearest Neighbor TSP...")
            start_time_nearest = (time() * 1000)

            if experiment_case in (1, 2):
                # Standard NN — fully connected or 1-to-1 (graph handles constraints)
                simple_tour = nearest_tsp(B, start_node, end_node, set_1, set_2)
            elif experiment_case == 3:
                # adjNN — deactivation guard avoids dead ends under compatibility
                simple_tour = adjNN(B, start_node, end_node, set_1, set_2)

            simple_tour_cost, time_details = total_cost(B, simple_tour)

            exec_time_nearest = (time() * 1000) - start_time_nearest
            results["nearest"] = {
                "tour": simple_tour,
                "cost": simple_tour_cost,
                "exec_time": exec_time_nearest,
                "time_details": time_details
            }
            print(f"Nearest Neighbor TSP. Cost: {simple_tour_cost}")
        except ValueError as e:
            print(f"Error in nearest method: {e}")

    if '2-opt' in methods_to_run:
        try:
            print("Running 2-Opt Optimization...")
            start_time_2opt = time() * 1000

            if simple_tour is None:
                print("No NN tour available for 2-opt. Skipping.")
                raise ValueError("No initial tour for 2-opt.")

            optimized_tour, optimized_tour_cost = two_opt_for_bipartite(
                simple_tour, B, set_1, set_2, max_iterations=1000)
            final_cost, time_details = total_cost(B, optimized_tour)

            if final_cost < simple_tour_cost:
                print(f"2-opt improved. {simple_tour_cost} -> {final_cost}")
                best_tour, best_cost = optimized_tour, final_cost
            else:
                print("2-opt did not improve. Keeping NN solution.")
                best_tour, best_cost = simple_tour, simple_tour_cost

            exec_time_2opt = (time() * 1000) - start_time_2opt
            results["2-opt"] = {
                "tour": best_tour,
                "cost": best_cost,
                "exec_time": exec_time_2opt,
                "time_details": time_details
            }
            print(f"2-opt completed. Final Cost: {best_cost}")
        except ValueError as e:
            print(f"Error in 2-opt method: {e}")

    if 'clustered-2-opt' in methods_to_run:
        try:
            print("Running Clustered 2-Opt Optimization...")
            start_time_c2opt = time() * 1000

            if simple_tour is None:
                print("No NN tour available for clustered 2-opt. Skipping.")
                raise ValueError("No initial tour for clustered 2-opt.")

            optimized_tour, optimized_tour_cost = clustered_two_opt(
                simple_tour, B, set_1, set_2, max_iterations=1000)
            final_cost, time_details = total_cost(B, optimized_tour)

            if final_cost < simple_tour_cost:
                print(f"Clustered 2-opt improved. {simple_tour_cost} -> {final_cost}")
                best_tour, best_cost = optimized_tour, final_cost
            else:
                print("Clustered 2-opt did not improve. Keeping NN solution.")
                best_tour, best_cost = simple_tour, simple_tour_cost

            exec_time_c2opt = (time() * 1000) - start_time_c2opt
            results["clustered-2-opt"] = {
                "tour": best_tour,
                "cost": best_cost,
                "exec_time": exec_time_c2opt,
                "time_details": time_details
            }
            print(f"Clustered 2-opt completed. Final Cost: {best_cost}")
        except ValueError as e:
            print(f"Error in clustered-2-opt method: {e}")

    if 'q-learning' in methods_to_run:
        try:
            print("Running Q-Learning TSP...")
            start_time_qlearning = (time() * 1000)
            q_learning_tour = q_learning_tsp(B, start_node, end_node, set_1, set_2)
            q_learning_tour_cost, time_details = total_cost(B, q_learning_tour)
            exec_time_qlearning = (time() * 1000) - start_time_qlearning
            results["q-learning"] = {
                "tour": q_learning_tour,
                "cost": q_learning_tour_cost,
                "exec_time": exec_time_qlearning,
                "time_details": time_details
            }
            print(f"Q-Learning TSP completed. Cost: {q_learning_tour_cost}")
        except ValueError as e:
            print(f"Error in q-learning method: {e}")

    # COMMENT OLD EXACT
    # try:
    #     print("Running Exact Method TSP...")
    #     start_time_exact = (time() * 1000)
    #     exact_tour, exact_tour_cost, time_details = run_exact_tsp(input_data)
    #     exec_time_exact = (time() * 1000) - start_time_exact
    #     if exact_tour:
    #         results["exact"] = {
    #             "tour": exact_tour,
    #             "cost": exact_tour_cost,
    #             "exec_time": exec_time_exact,
    #             "time_details": time_details
    #         }
    #         print(f"Exact Method TSP completed. Cost: {exact_tour_cost}")
    # except ValueError as e:
    #     print(f"Error in exact method: {e}")

    # In run_tsp(), replace the exact method block with:
    if 'exact_cbc' in methods_to_run or 'exact_cplex' in methods_to_run:
        try:
            print("Running Exact Method TSP...")
            SOLVERS = {}
            if 'exact_cbc' in methods_to_run:
                SOLVERS['cbc'] = make_cbc_solver()
            if 'exact_cplex' in methods_to_run:
                SOLVERS['cplex'] = make_cplex_solver()

            for solver_name, solver_obj in SOLVERS.items():
                try:
                    start_time_exact = time() * 1000
                    exact_tour, exact_tour_cost, time_details = run_exact_tsp(
                        input_data, solver=solver_obj)
                    exec_time_exact = time() * 1000 - start_time_exact
                    if exact_tour:
                        results[f"exact_{solver_name}"] = {
                            "tour": exact_tour,
                            "cost": exact_tour_cost,
                            "exec_time": exec_time_exact,
                            "time_details": time_details
                        }
                        print(f"Exact [{solver_name}] Cost: {exact_tour_cost} "
                              f"Time: {exec_time_exact:.2f}ms")
                except Exception as e:
                    print(f"Error in exact method [{solver_name}]: {e}")
        except Exception as e:
            print(f"Error in exact block: {e}")

    #Convert all data to native Python types to avoid JSON serialization issues
    results = convert_to_native_types(results)

    #ave the results to a file
    with open(output_json_file_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    return results


def save_results_to_csv(results, gravity_rack, kit_holder, run_id):
    df_results = pd.DataFrame(results)
    filename = f"experiment_results/GR_{gravity_rack}_KH_{kit_holder}_Run_{run_id}.csv"
    os.makedirs("experiment_results", exist_ok=True)
    df_results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
