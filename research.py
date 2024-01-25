import operator
import random
import math

import deap.gp
import numpy as np
from deap import algorithms, base, creator, tools, gp
import matplotlib.pyplot as plt
import networkx as nx

LOWER_BOUND = -50.0
UPPER_BOUND = 50.0
STEP_SIZE = 0.1
X_RANGE = np.arange(LOWER_BOUND, UPPER_BOUND + STEP_SIZE, STEP_SIZE)
EPHEMERAL_CONSTANT_PRECISION = 3
TOURNAMENT_SIZE = 2
ELITES_SIZE = 1
NUMBER_OF_GENERATIONS = 200
POPULATION_SIZE = 50


def target_polynomial(x):
    return 1 / x


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


def sqrt(x):
    if x < 0:
        return 1
    return math.sqrt(x)


def sin(x):
    return math.sin(x)


def pow2(x):
    return x * x


def pow3(x):
    return x * x * x


pset = gp.PrimitiveSet("MAIN", 1)  # 1 input variable
pset.addTerminal(-1.0)
pset.addTerminal(1.0)
pset.addTerminal(2.0)
pset.addTerminal(3.0)
pset.renameArguments(ARG0="x")

pset.addPrimitive(protectedAdd, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(sqrt, 1)
pset.addPrimitive(sin, 1)
pset.addPrimitive(pow2, 1)
pset.addPrimitive(pow3, 1)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genGrow, pset=pset, min_=3, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)
deap.gp.staticLimit(operator.attrgetter('height'), max_value=2)


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


toolbox.register("mate", koza_custom_two_point_crossover)
toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

if __name__ == "__main__":
    best_fitness = float('inf')
    best_individual = None
    fitness_values = []
    hall_of_fame = tools.HallOfFame(maxsize=ELITES_SIZE)
    population = toolbox.population(n=POPULATION_SIZE)
    for x in range(NUMBER_OF_GENERATIONS):
        print("Generation " + str(x))
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.9, mutpb=0.01)  # perform only mutation + crossover

        # Need to manually evaluate the offspring
        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        combined_population = population + offspring

        # Implementation of elitism
        hall_of_fame.update(combined_population)
        offspring = toolbox.select(combined_population, POPULATION_SIZE)

        # Replace the current population by the offspring
        population[:] = offspring

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
