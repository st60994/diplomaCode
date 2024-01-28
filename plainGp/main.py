import operator
import random
import math

import numpy as np
from deap import algorithms, base, creator, tools, gp
import matplotlib.pyplot as plt
import networkx as nx

LOWER_BOUND = -50.0
UPPER_BOUND = 50.0
STEP_SIZE = 0.1
X_RANGE = np.arange(LOWER_BOUND, UPPER_BOUND + STEP_SIZE, STEP_SIZE)
EPHEMERAL_CONSTANT_PRECISION = 3


def target_polynomial(x):
    return x ** 2 + x + 1


def terminal_ephemeral():
    return round(random.uniform(LOWER_BOUND, UPPER_BOUND), EPHEMERAL_CONSTANT_PRECISION)


def protectedAdd(left, right):
    try:
        result = left + right
        if math.isinf(result) or math.isnan(result):
            raise OverflowError("Result is infinity or NaN")
        return result
    except OverflowError as e:
        print(f"Warning: {e}")
        print("Left:", left)
        print("Right:", right)
        return 1


def protectedDiv(left, right):
    try:
        result = left / right
        if math.isinf(result) or math.isnan(result):
            raise OverflowError("Result is infinity or NaN")
        return result
    except ZeroDivisionError:
        print("Warning: Division by zero encountered.")
        print("Left:", left)
        print("Right:", right)
        return 1
    except OverflowError as e:
        print(f"Warning: {e}")
        print("Left:", left)
        print("Right:", right)
        return 1


pset = gp.PrimitiveSet("MAIN", 1)  # 1 input variable
pset.renameArguments(ARG0="x")
pset.addEphemeralConstant("rand_const", terminal_ephemeral)

pset.addPrimitive(protectedAdd, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Define the fitness measure
def evaluate_individual(individual):
    try:
        function = gp.compile(expr=individual, pset=pset)
        x_values = X_RANGE
        errors = []
        for x in x_values:
            individual_output = function(x)
            error = abs(target_polynomial(x) - individual_output)
            errors.append(error)
        total_error = sum(errors)
        return total_error,
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return float('inf'),  # Return a high fitness in case of an error


toolbox.register("evaluate", evaluate_individual)


# Custom one-point crossover
def custom_one_point_crossover(parent1, parent2):
    # Create two offspring by copying parents
    offspring1 = creator.Individual(parent1)
    offspring2 = creator.Individual(parent2)

    # Perform one-point crossover at a random position
    crossover_point = random.randint(0, min(len(parent1), len(parent2)) - 1)

    # Swap the elements after the crossover point
    offspring1[0], offspring2[0] = offspring2[0], offspring1[0]

    return offspring1, offspring2


toolbox.register("mate", custom_one_point_crossover)
toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=8)

if __name__ == "__main__":
    population_size = 1000
    best_fitness = 10000
    best_individual = None
    fitness_values = []
    for x in range(1):
        population = toolbox.population(n=population_size)
        algorithms.eaSimple(population, toolbox, cxpb=0.9, mutpb=0.01, ngen=50, stats=None, verbose=True)
        if len(population) != 0:
            best_current_individual = tools.selBest(population, k=1)[0]
            fitness_values.append(best_current_individual.fitness.values[0])
            if best_current_individual.fitness.values[0] < best_fitness:
                best_fitness = best_current_individual.fitness.values[0]
                best_individual = best_current_individual
            print(f"Best individual: {best_current_individual}, Fitness: {best_current_individual.fitness.values[0]}")

    nodes, edges, labels = gp.graph(best_individual)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    print("Fitness values: " + str(fitness_values))
    # print ("Avg fitness value: ")
    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    plt.show()
