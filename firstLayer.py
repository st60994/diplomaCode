import concurrent.futures
import operator
import random

import deap.gp
import numpy as np
from deap import algorithms, base, creator, tools, gp
import traceback
import multiprocessing

from customLogic import koza_custom_two_point_crossover
from secondLayer import SecondLayer
from methodDefinitions import protectedAdd, sqrt, pow2, pow3

LOWER_BOUND = 0.1
UPPER_BOUND = 15
STEP_SIZE = 0.1
X_RANGE = np.arange(LOWER_BOUND, UPPER_BOUND + STEP_SIZE, STEP_SIZE)
TOURNAMENT_SIZE = 2
ELITES_SIZE = 1
NUMBER_OF_GENERATIONS = 120
POPULATION_SIZE = 500
NUMBER_OF_SUB_MODELS = 1
MAX_TREE_HEIGHT = 17
MIN_TREE_INIT = 3
MAX_TREE_INIT = 3


def target_polynomial(x):
    return 1 / x


class FirstLayer:
    toolbox = None
    pset = None

    def __add_primitive_set(self):
        self.pset.addPrimitive(protectedAdd, 2)
        self.pset.addPrimitive(operator.sub, 2)
        self.pset.addPrimitive(operator.mul, 2)
        self.pset.addPrimitive(sqrt, 1)
        # pset.addPrimitive(sin, 1)
        self.pset.addPrimitive(pow2, 1)
        self.pset.addPrimitive(pow3, 1)

    def __create_terminal_set(self):
        self.pset.addTerminal(-1.0)
        self.pset.addTerminal(1.0)
        self.pset.addTerminal(2.0)
        self.pset.addTerminal(3.0)
        self.pset.renameArguments(ARG0="x")

    # Define the fitness measure
    def __evaluate_individual(self, individual):
        try:
            function = gp.compile(expr=individual, pset=self.pset)
            x_values = X_RANGE
            errors = []
            for index in x_values:
                individual_output = function(index)
                error = abs(target_polynomial(index) - individual_output)
                errors.append(error)
            total_error = sum(errors)
            return total_error,
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return float('inf'),  # Return a high fitness in case of an error

    def __initialize_gp_run(self):
        self.pset = gp.PrimitiveSet("MAIN", 1)  # 1 input variable
        self.__create_terminal_set()
        self.__add_primitive_set()
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=self.pset)

        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genGrow, pset=self.pset, min_=MIN_TREE_INIT, max_=MAX_TREE_INIT)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", koza_custom_two_point_crossover)
        self.toolbox.register("mutate", gp.mutNodeReplacement, pset=self.pset)
        self.toolbox.register("evaluate", self.__evaluate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

        #self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT))
        #self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT)) TODO doesnt work no clue why

    def first_layer_evolution(self, process_id, new_terminal_list):
        try:
            self.__initialize_gp_run()
            fitness_values = []
            hall_of_fame = tools.HallOfFame(maxsize=ELITES_SIZE)
            population = self.toolbox.population(n=POPULATION_SIZE)
        except:
            print("err")
            traceback.print_exc()
        for index in range(NUMBER_OF_GENERATIONS):
            try:
                print(str(process_id) + ": Generation " + str(index))
                offspring = algorithms.varAnd(population, self.toolbox, cxpb=0.9,
                                              mutpb=0.01)  # perform only mutation + crossover

                # Need to manually evaluate the offspring
                fitnesses = self.toolbox.map(self.toolbox.evaluate, offspring)
                for ind, fit in zip(offspring, fitnesses):
                    ind.fitness.values = fit

                combined_population = population + offspring

                # Implementation of elitism
                hall_of_fame.update(combined_population)
                offspring = self.toolbox.select(combined_population, POPULATION_SIZE)

                # Replace the current population by the offspring
                population[:] = offspring
                combined_population += population
            except:
                print("err")
                traceback.print_exc()
        new_terminal_list += population
        if len(population) != 0:
            best_current_individual = tools.selBest(population, k=1)[0]
            fitness_values.append(best_current_individual.fitness.values[0])
            print(f"Best individual: {best_current_individual}, Fitness: {best_current_individual.fitness.values[0]}")


if __name__ == "__main__":
    manager = multiprocessing.Manager()
    new_terminals = manager.list()
    first_layer_instance = FirstLayer()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(first_layer_instance.first_layer_evolution, process_id, new_terminals) for process_id in
                   range(NUMBER_OF_SUB_MODELS)]

        concurrent.futures.wait(futures)

    functions = []
    terminal_map = {}
    best_fitness = float("inf")
    best_individual = None
    for i, terminal in enumerate(new_terminals):
        terminal_name = f'{terminal}'
        terminal_map[terminal_name] = terminal
        if best_fitness > terminal.fitness.values[0]:
            best_fitness = terminal.fitness.values[0]
            best_individual = terminal
    print("Best fitness from first run: " + str(best_fitness))
    print("---------Second run---------")
    secondLayer = SecondLayer(terminal_map)
    secondLayer.execute_run()

    # print("Fitness values: " + str(fitness_values))
    # print ("Avg fitness value: ")
