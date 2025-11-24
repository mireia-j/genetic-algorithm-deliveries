import random
import numpy as np
import matplotlib
import math
import copy
# Set non-interactive backend to avoid Tkinter dependency
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# 1. Problem Data Definition
WAREHOUSE = [20, 120]  # Warehouse location [x, y]
VEHICLE_CAPACITY = 60  # Maximum vehicle capacity in kg

# Define items with their weights in kg
ITEMS = {
    1: 1.2,   # Item 1: 1.2 kg
    2: 3.8,   # Item 2: 3.8 kg
    3: 7.5,   # Item 3: 7.5 kg
    4: 0.9,   # Item 4: 0.9 kg
    5: 15.4,  # Item 5: 15.4 kg
    6: 12.1,  # Item 6: 12.1 kg
    7: 4.3,   # Item 7: 4.3 kg
    8: 19.7,  # Item 8: 19.7 kg
    9: 8.6,   # Item 9: 8.6 kg
    10: 2.5   # Item 10: 2.5 kg
}

# Client locations [x, y]
CLIENT_LOCATIONS = {
    1: [35, 115],
    2: [50, 140],
    3: [70, 100],
    4: [40, 80],
    5: [25, 60],
    6: [90, 70]
}

# Define client item demands as a dictionary of client_id: {item_id: quantity}
CLIENT_ITEM_DEMANDS = {
    1: {1: 3, 3: 2},           # Client 1: 3×Item1 + 2×Item3
    2: {2: 6},                 # Client 2: 6×Item2
    3: {5: 2, 7: 4},           # Client 3: 2×Item5 + 4×Item7
    4: {3: 8},                 # Client 4: 8×Item3
    5: {6: 5, 9: 2},           # Client 5: 5×Item6 + 2×Item9
    6: {6:1}                   # Client 6: 6xItem1
}

# Calculate client demands in kg based on items and quantities
CLIENT_DEMANDS = {}
for client_id, items in CLIENT_ITEM_DEMANDS.items():
    total_weight = sum(ITEMS[item_id] * quantity for item_id, quantity in items.items())
    CLIENT_DEMANDS[client_id] = total_weight

# Default parameters
DEFAULT_PARAMS = {
    'population_size': 50, 
    'generations': 3000, 
    'prob_crossover': 0.8,
    'prob_mutation': 0.3,
    'selection_method': 'tournament'
}

# Creating the arrays we need
weights_arr = []
clients_arr = []

for client_id, items in CLIENT_ITEM_DEMANDS.items():
    for item_id, quantity in items.items():
        weight = ITEMS[item_id]
        # Add the weight and client_id for each unit demanded
        weights_arr.extend([weight] * quantity)
        clients_arr.extend([client_id] * quantity)

def create_individual():
    # We first initialize an array with 0 and 1's
    # This array will indicate what orders we are delivering (1 if we are delivering the order)
    deliveries_arr = [random.randint(0, 1) for _ in range(len(clients_arr))]


    # We need to decide the order in which we are going to deliver the orders

    # 1. We need to check how many clients we have for delivering and specify an order
    selected_clients = [c for d, c in zip(deliveries_arr, clients_arr) if d == 1]

    # We don't want repeated values
    selected_clients_no_rep = list(set(selected_clients))
    random.shuffle(selected_clients_no_rep)

    # We need to assign the random order to the items we want to deliver
    deliveries_arr = change_order(deliveries_arr, selected_clients_no_rep)

    return deliveries_arr

# Given a desired order and an individual, this function returns the individual with the desired order of clients
def change_order(individual, desired_order):
    # Change the numbers > 0 to 1
    individual2 = [1 if x > 1 else x for x in individual]

    # The variable order contains the corresponding order
    order = 1
    for client in desired_order:
        # Get which elements correspond to this client in the clients_arr
        indices = [i for i, val in enumerate(clients_arr) if val == client]
        for i in indices:
            if individual2[i] == 1:
                individual2[i] = order
        order += 1

    return individual2

# Given an individual returns the order of the clients
def get_order_clients(individual):
    order_clients_arr1 = []
    for i in range(1, max(individual)+1):
        index = individual.index(i)
        order_clients_arr1.append(clients_arr[index])
    
    return order_clients_arr1

def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: [x, y] coordinates
        point2: [x, y] coordinates
        
    Returns:
        float: Euclidean distance
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_total_distance(route):
    """
    Calculate the total distance of a route including return to warehouse.
    
    Args:
        route: List of client IDs
        
    Returns:
        float: Total distance
    """
    if not route:
        return 0
    
    total_distance = 0
    
    # Distance from warehouse to first client
    total_distance += calculate_distance(WAREHOUSE, CLIENT_LOCATIONS[route[0]])
    
    # Distance between consecutive clients
    for i in range(len(route) - 1):
        total_distance += calculate_distance(CLIENT_LOCATIONS[route[i]], CLIENT_LOCATIONS[route[i + 1]])
    
    # Distance from last client back to warehouse
    total_distance += calculate_distance(CLIENT_LOCATIONS[route[-1]], WAREHOUSE)
    
    return total_distance

def fitness(individual, weights_arr, capacity, penalty_factor=20, more_weight = 20):
    total_weight = sum(w for i, w in enumerate(weights_arr) if individual[i] > 0)

    # Calculate penalty for exceeding capacity
    penalty = max(0, total_weight - capacity) * penalty_factor
    # Return fitness as total value plus any penalties

    clients = get_order_clients(individual)
    total_distance = calculate_total_distance(clients)

    total_weight = total_weight * more_weight

    return max(0, total_weight - total_distance - penalty)
    # We want to minimize both total distance and penalty

def tournament_selection(population, fitnesses):
    # Randomly select 20 individuals from the population
    tournament_indices = random.sample(range(len(population)), 20) 

    # Find the index of the best individual in the tournament
    best_index = tournament_indices[0]  # Start with the first individual
    best_fitness = fitnesses[best_index]

    for i in tournament_indices[1:]:  # Check the rest
        if fitnesses[i] > best_fitness:
            best_fitness = fitnesses[i]
            best_index = i

    # Return the best individual
    return population[best_index]

def crossover(parent1, parent2):
    # Here we decide where we need to cut the parent 1 and 2
    point = random.randint(0, len(clients_arr) - 1)

    end_chosen_client = clients_arr[point]
    while end_chosen_client == clients_arr[point] and point != (len(clients_arr)-1):
        end_chosen_client = clients_arr[point]
        point += 1

    order_parent1 = {value: index + 1 for index, value in enumerate(get_order_clients(parent1))}
    final_order_parent1 = order_parent1.copy()
    for client in order_parent1.keys():
        if client not in clients_arr[:point]:
            del final_order_parent1[client]

    order_parent2 = {value: index + 1 for index, value in enumerate(get_order_clients(parent2))}
    final_order_parent2 = order_parent2.copy()
    for client in order_parent2.keys():
        if client not in clients_arr[point:]:
            del final_order_parent2[client]

    # Combine both dictionaries
    combined_order = final_order_parent1 | final_order_parent2
    sorted_combined_order = dict(sorted(combined_order.items(), key=lambda item: item[1]))

    order_child = list(sorted_combined_order.keys())
    child_arr = parent1[:point] + parent2[point:]
    
    child_arr = change_order(child_arr, order_child)

    return child_arr

# This mutation changes the items we are choosing for delivering
def delivery_mutation(individual):
    index = random.randint(0, len(individual)-1)
    order_clients = get_order_clients(individual)

    new_individual = copy.copy(individual)
    new_order = copy.copy(order_clients)

    # Cambiar un cero por un 1
    if individual[index] == 0:
        new_individual[index] = 1
        if clients_arr[index] not in order_clients:
            new_order.append(clients_arr[index])
    # Cambiar un 1 por un cero
    else:
        new_individual[index] = 0
        
        indexes = [i for i, x in enumerate(clients_arr) if x == clients_arr[index]]
        filtered_other = [new_individual[i] for i in indexes]
        # We now check if the client is still in the list for delivering
        if sum(filtered_other) == 0:
            new_order.remove(clients_arr[index])
    
    new_individual2 = change_order(new_individual, new_order)

    return new_individual2

# This mutation only swaps the order of the clients
def order_mutation(individual):
    order_mutation1 = get_order_clients(individual)
    random.shuffle(order_mutation1)

    new_individual = change_order(individual, order_mutation1)

    return new_individual

def mutate(individual, probability):
    """
    Apply one of the mutation operators randomly n times
    where n is the 30% of the length of the individual
    
    Args:
        individual: List of client IDs
        params: Dictionary of parameters
        
    Returns:
        list: Mutated individual
    """
    if random.random() > probability:
        return individual.copy()
    
    # I want to apply the mutation process approximately (30% the length of the array) times
    percentage = 0.3
    number_of_repetitions_process = max(2, math.floor(percentage*len(individual)))
    for i in range(0, number_of_repetitions_process):
        # Choose a mutation operator randomly
        mutation_op = random.choices([delivery_mutation, order_mutation], weights=[0.5, 0.5])[0] 

        individual = mutation_op(individual)
    
    # Apply the selected mutation
    return individual

def delivery_problem(params=None):
    if params is None:
        params = DEFAULT_PARAMS.copy()

    # Initialize a random population of individuals
    population = [create_individual() for _ in range(params['population_size'])]
    generation = 0
    best_fitness = 0

    while generation < params['generations']:
        # Calculate fitness for each possible solution in the population
        fitnesses = [fitness(individual, weights_arr, VEHICLE_CAPACITY) for individual in population]

        # if it is the best we have seen so far, let's report on it
        for i in range(params['population_size']):
            weigths_individual = sum(w for j, w in enumerate(weights_arr) if int(population[i][j]) >= 1)
            if fitnesses[i] > best_fitness and weigths_individual <= VEHICLE_CAPACITY:
                best_fitness = fitnesses[i]
                print("Generation:", generation, "Best individual:", population[i], "Value: ", best_fitness, "Weight: ", weigths_individual)

        new_population = []
        while len(new_population) < params['population_size']:
            # Select two parents from the population
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            # Create offspring using crossover
            child = crossover(parent1, parent2) 
            # Apply mutation to the offspring
            child = mutate(child, params['prob_mutation']) 
            # Add the new offspring to the new population
            new_population.append(child)

        # Update the population for the next generation
        population = new_population
        generation += 1  # Increase the generation counter

    # Find the best individual within the weight limit in the last generation
    best_fitness = 0
    fitnesses = [fitness(individual, weights_arr, VEHICLE_CAPACITY) for individual in population]

    weights_population = [sum(w for j, w in enumerate(weights_arr) if individual[j] >= 1) for individual in population]
    idx_valid_capacity = [i for i, value in filter(lambda item: item[1] < VEHICLE_CAPACITY, enumerate(weights_population))]
    # List of the populations where the population's weights are less than the capacity

    if len(idx_valid_capacity) > 0:
        for idx in idx_valid_capacity:
            if fitnesses[idx] >= best_fitness:
                best_value = fitnesses[idx]
                best_individual = population[idx]
        print("Final result -> Generation:", generation, "Best individual:", best_individual, "Value: ", best_value, "Weight: ", weights_population[idx])
        return best_individual
    else:
        print("The final generation did not yield an optimal solution.")
        return None

if __name__ == "__main__":
    individual1 = create_individual()

    delivery_problem()
