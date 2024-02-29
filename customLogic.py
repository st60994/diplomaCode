import random

import deap
from deap import creator, gp
from deap.tools import selection


def koza_custom_two_point_crossover(parent1, parent2):
    if random.uniform(0, 1) <= 0.9:
        indices_of_functions_parent1 = []
        indices_of_functions_parent2 = []
        for i in range(len(parent1)):
            if isinstance(parent1[i], deap.gp.Primitive):
                indices_of_functions_parent1.append(i)
        crossing_index_parent1 = random.choice(indices_of_functions_parent1)
        for i in range(len(parent2)):
            if isinstance(parent2[i], deap.gp.Primitive):
                indices_of_functions_parent2.append(i)
        crossing_index_parent2 = random.choice(indices_of_functions_parent2)
    else:
        indices_of_terminals_parent1 = []
        indices_of_terminals_parent2 = []
        for i in range(len(parent1)):
            if isinstance(parent1[i], deap.gp.Terminal):
                indices_of_terminals_parent1.append(i)
        crossing_index_parent1 = random.choice(indices_of_terminals_parent1)
        for i in range(len(parent2)):
            if isinstance(parent2[i], deap.gp.Terminal):
                indices_of_terminals_parent2.append(i)
        crossing_index_parent2 = random.choice(indices_of_terminals_parent2)

    slice1 = parent1.searchSubtree(crossing_index_parent1)
    slice2 = parent2.searchSubtree(crossing_index_parent2)

    start1, stop1 = slice1.start, slice1.stop
    start2, stop2 = slice2.start, slice2.stop

    # Perform crossover
    combined_tree1 = parent1[:start1] + parent2[start2:stop2] + parent1[stop1:]
    combined_tree2 = parent2[:start2] + parent1[start1:stop1] + parent2[stop2:]
    return creator.Individual(combined_tree1), creator.Individual(combined_tree2)


def get_individual_height(individual):
    stack = [0]
    max_depth = 0
    for elem in individual:
        if stack:  # Check if stack is not empty before popping
            depth = stack.pop()
            max_depth = max(max_depth, depth)
            stack.extend([depth + 1] * elem.arity)
        else:
            break  # Exit the loop if stack is empty
    return max_depth + 1

def get_nodes_with_height(individual, height):
    stack = [0]
    nodes = []
    max_depth = 0
    for i, elem in enumerate(individual):
        if stack:  # Check if stack is not empty before popping
            depth = stack.pop()
            if height == depth:
                nodes.append(i)
            max_depth = max(max_depth, depth)
            stack.extend([depth + 1] * elem.arity)
        else:
            break  # Exit the loop if stack is empty
    return nodes


def trim_individual(individual, max_tree_height):
    current_height = get_individual_height(individual)
    indices = get_nodes_with_height(individual, 1) # TODO complete this
    if current_height > max_tree_height:
        trim_length = current_height - max_tree_height
        individual = gp.PrimitiveTree(individual[:-trim_length])
    return individual


def find_nodes_at_height(tree, target_height, current_height=1, index=0):
    if index >= len(tree):
        return []

    if current_height == target_height:
        return [index]

    # Calculate the index of the first child of the current node
    child_index = index + 1

    # Recursively find nodes at the target height in the subtree
    nodes_at_height = []
    arity = tree[index].arity
    for child_number in range(arity):
        nodes_at_height.extend(find_nodes_at_height(tree, target_height, current_height + 1, child_index + child_number))
        # Move to the next sibling node
        child_index = get_next_sibling_index(tree, child_index)

    return nodes_at_height


def get_next_sibling_index(tree, index):
    """
    Function to get the index of the next sibling node.

    Parameters:
    - tree: The list representing the tree.
    - index: Index of the current node.

    Returns:
    - Index of the next sibling node.
    """
    parent_index = index
    while True:
        parent_arity = tree[parent_index].arity
        if parent_arity == 0:  # If parent has no siblings, return -1
            return -1

        # Move up the tree until finding a node with siblings or reaching the root
        while parent_index > 0 and tree[parent_index - 1].arity == 0:
            parent_index -= 1

        # Check if there's a next sibling at the current level
        if parent_index + parent_arity + 1 < len(tree) and tree[parent_index + parent_arity].arity != 0:
            return parent_index + parent_arity + 1
        else:
            parent_index = parent_index + parent_arity  # Move to the next parent


def koza_over_selection(individuals, k, tournsize, population_size):
    individuals = sorted(individuals, key=lambda x: x.fitness, reverse=True)
    fittest_individuals_percentage = (32000 / population_size) / 100
    total_fitness_sum = 0
    for individual in individuals:
        if len(individual.fitness.values) > 0:
            total_fitness_sum += individual.fitness.values[0]

    cumulative_fitness = 0
    top_individuals = []
    for ind in individuals:
        cumulative_fitness += ind.fitness.values[0]
        if cumulative_fitness < fittest_individuals_percentage * total_fitness_sum:
            top_individuals.append(ind)
        else:
            break
    rest_individuals = individuals[len(top_individuals):]

    if random.uniform(0, 1) <= 0.8:
        return selection.selTournament(top_individuals, k, tournsize)
    else:
        return selection.selTournament(rest_individuals, k, tournsize)
