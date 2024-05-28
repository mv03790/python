import utility as utility
import loader as loader
import numpy as np
import pandas as pd


def main():

    # Paths to the data and solution files.
    vrp_file = "n80-k10.vrp"  # "data/n80-k10.vrp"
    sol_file = "n80-k10.sol"  # "data/n80-k10.sol"

    # Loading the VRP data file.
    px, py, demand, capacity, depot = loader.load_data(vrp_file)

    # Displaying to console the distance and visualizing the optimal VRP solution.
    vrp_best_sol = loader.load_solution(sol_file)
    best_distance = utility.calculate_total_distance(vrp_best_sol, px, py, depot)
    print("\nBest VRP Distance:", best_distance)
    utility.visualise_solution(vrp_best_sol, px, py, depot, "Optimal Solution")

    # Executing and visualizing the nearest neighbour VRP heuristic.
    # Uncomment it to do your assignment!

    nnh_solution = nearest_neighbour_heuristic(px, py, demand, capacity, depot)
    nnh_distance = utility.calculate_total_distance(nnh_solution, px, py, depot)
    print("\nNearest Neighbour VRP Heuristic Distance:", nnh_distance)
    utility.visualise_solution(
        nnh_solution, px, py, depot, "Nearest Neighbour Heuristic"
    )

    # Executing and visualizing the saving VRP heuristic.
    # Uncomment it to do your assignment!

    sh_solution = savings_heuristic(px, py, demand, capacity, depot)
    sh_distance = utility.calculate_total_distance(sh_solution, px, py, depot)
    print("\nSaving VRP Heuristic Distance:", sh_distance)
    utility.visualise_solution(sh_solution, px, py, depot, "Savings Heuristic")


def nearest_neighbour_heuristic(px, py, demand, capacity, depot):
    """
    Algorithm for the nearest neighbour heuristic to generate VRP solutions.

    :param px: List of X coordinates for each node.
    :param py: List of Y coordinates for each node.
    :param demand: List of each nodes demand.
    :param capacity: Vehicle carrying capacity.
    :param depot: Depot.
    :return: List of vehicle routes (tours).
    """

    n = len(px)
    unvisited = set(range(n))
    unvisited.remove(depot)
    routes = []
    current_route = [depot]
    current_load = 0
    nearest_node = 0

    while unvisited:
        last_visited = current_route[-1]

        distances = []
        for node in unvisited:
            distance = utility.calculate_euclidean_distance(px, py, last_visited, node)
            distances.append([node, distance])

        distances.sort(key=lambda x: x[1])
        for node, distance in distances:
            if current_load + demand[node] <= capacity:
                nearest_node = node
                break

        if current_load + demand[nearest_node] <= capacity:
            current_route.append(nearest_node)
            current_load += demand[nearest_node]
            unvisited.remove(nearest_node)
        else:
            current_route.append(depot)
            routes.append(current_route)
            current_route = [depot]
            current_load = 0

    if current_route != [depot]:
        current_route.append(depot)
        routes.append(current_route)

    return routes


def calculate_savings(px, py, depot):
    n = len(px)
    savings = []
    for i in range(1, n):
        for j in range(i + 1, n):
            saving_value = (
                utility.calculate_euclidean_distance(px, py, depot, i)
                + utility.calculate_euclidean_distance(px, py, depot, j)
                - utility.calculate_euclidean_distance(px, py, i, j)
            )

            savings.append((saving_value, i, j))
            savings.append((saving_value, j, i))
    savings.sort(reverse=True, key=lambda x: x[0])
    return savings


def get_load(route, demand):
    return sum([demand[i] for i in route])


def get_route_index_by_node(routes, node):
    for i, route in enumerate(routes):
        if node in route:
            return i
    return -1


def savings_heuristic(px, py, demand, capacity, depot):
    """
    Algorithm for Implementing the savings heuristic to generate VRP solutions.

    :param px: List of X coordinates for each node.
    :param py: List of Y coordinates for each node.
    :param demand: List of each nodes demand.
    :param capacity: Vehicle carrying capacity.
    :param depot: Depot.
    :return: List of vehicle routes (tours).
    """

    savings = calculate_savings(px, py, depot)
    saving, i, j = savings.pop(0)
    final_routes = [[i, j]]

    savings = [s for s in savings if s[1] != i and s[2] != j and [s[1], s[2]] != [j, i]]

    while savings:
        saving, i, j = savings.pop(0)

        loop_first_item = i
        loop_last_item = j

        first_index = get_route_index_by_node(final_routes, loop_first_item)
        last_index = get_route_index_by_node(final_routes, loop_last_item)
        if first_index != -1 and last_index != -1 and first_index != last_index:
            load = get_load(
                final_routes[first_index] + final_routes[last_index], demand
            )
            if load <= capacity:
                final_routes[first_index] = (
                    final_routes[first_index] + final_routes[last_index]
                )
                final_routes.pop(last_index)
                savings = [
                    s
                    for s in savings
                    if s[1] != loop_first_item
                    and s[2] != loop_last_item
                    and [s[1], s[2]] != [loop_last_item, loop_first_item]
                ]
                continue
            else:
                continue

        index = first_index
        if index == -1:
            index = last_index

        if index != -1:
            route_array = final_routes[index]
        else:
            route_array = [loop_first_item, loop_last_item]

        route_first_item = route_array[0]
        route_last_item = route_array[-1]

        if route_first_item == loop_last_item:
            load = get_load([loop_first_item] + route_array, demand)
            if load <= capacity:
                final_routes[index] = [loop_first_item] + final_routes[index]
            else:
                continue
        elif route_last_item == loop_first_item:
            load = get_load(route_array + [loop_last_item], demand)
            if load <= capacity:
                final_routes[index] = final_routes[index] + [loop_last_item]
            else:
                continue
        else:
            final_routes.append([loop_first_item, loop_last_item])

        savings = [
            s
            for s in savings
            if s[1] != loop_first_item
            and s[2] != loop_last_item
            and [s[1], s[2]] != [loop_last_item, loop_first_item]
        ]

        for k in range(len(final_routes[index]) - 1):
            for l in range(k + 1, len(final_routes[index])):
                current_value = final_routes[index][k]
                next_value = final_routes[index][l]
                savings = [
                    s
                    for s in savings
                    if [s[1], s[2]] != [current_value, next_value]
                    and [s[1], s[2]] != [next_value, current_value]
                ]

    return list(final_routes)


if __name__ == "__main__":
    main()
