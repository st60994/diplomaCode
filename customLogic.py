import random

import deap
from deap import creator, gp
from deap.tools import selection

from gpInitialization import MAX_TREE_HEIGHT


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

    combined_tree1 = parent1[:crossing_index_parent1] + parent2[crossing_index_parent2:]
    combined_tree2 = parent2[:crossing_index_parent2] + parent1[crossing_index_parent1:]
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
    return max_depth


def trim_individual(individual):
    if get_individual_height(individual) > MAX_TREE_HEIGHT:
        return gp.PrimitiveTree(individual[:MAX_TREE_HEIGHT])
    return individual


def koza_over_selection(individuals, k, tournsize, population_size):
    individuals = sorted(individuals, key=lambda x: x.fitness, reverse=True)
    fittest_individuals_percentage = 32000 / population_size

    top_count = int(fittest_individuals_percentage * population_size)
    top_individuals = individuals[:top_count]
    rest_individuals = individuals[top_count:]

    if random.uniform(0, 1) <= 0.8:
        return selection.selTournament(top_individuals, k, tournsize)
    else:
        return selection.selTournament(rest_individuals, k, tournsize)
