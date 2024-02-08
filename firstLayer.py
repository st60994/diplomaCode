import concurrent.futures
import random

import numpy as np
from deap import algorithms, base, creator, tools, gp
import traceback
import multiprocessing

from csvExport import CsvExporter
from customLogic import koza_custom_two_point_crossover
from gpInitialization import GpFirstLayerInitializer, GpSecondLayerInitializer, target_polynomial
from secondLayer import SecondLayer

LOWER_BOUND_X = 0.1
UPPER_BOUND_X = 15
LOWER_BOUND_Y = 0.1
UPPER_BOUND_Y = 15
STEP_SIZE_X = 0.1
STEP_SIZE_Y = 0.1
X_RANGE = np.arange(LOWER_BOUND_X, UPPER_BOUND_X + STEP_SIZE_X, STEP_SIZE_X)
Y_RANGE = np.arange(LOWER_BOUND_Y, UPPER_BOUND_Y + STEP_SIZE_Y, STEP_SIZE_Y)
BOOTSTRAPPING_PERCENTAGE = 60

TOURNAMENT_SIZE = 2
ELITES_SIZE = 1
NUMBER_OF_GENERATIONS = 20
POPULATION_SIZE = 100
NUMBER_OF_SUB_MODELS = 2
MAX_TREE_HEIGHT = 17
MIN_TREE_INIT_HEIGHT = 3
MAX_TREE_INIT_HEIGHT = 3

first_layer_params = {
    'TOURNAMENT_SIZE': TOURNAMENT_SIZE,
    'ELITES_SIZE': ELITES_SIZE,
    'NUMBER_OF_GENERATIONS': NUMBER_OF_GENERATIONS,
    'POPULATION_SIZE': POPULATION_SIZE,
    'NUMBER_OF_SUB_MODELS': NUMBER_OF_SUB_MODELS,
    'MAX_TREE_HEIGHT': MAX_TREE_HEIGHT,
    'MIN_TREE_INIT_HEIGHT': MIN_TREE_INIT_HEIGHT,
    'MAX_TREE_INIT_HEIGHT': MAX_TREE_INIT_HEIGHT,
    'LOWER_BOUND_X': LOWER_BOUND_X,
    'UPPER_BOUND_X': UPPER_BOUND_X,
    'LOWER_BOUND_Y': LOWER_BOUND_Y,
    'UPPER_BOUND_Y': UPPER_BOUND_Y,
    'STEP_SIZE_X': STEP_SIZE_X,
    'STEP_SIZE_Y': STEP_SIZE_Y,
    'BOOTSTRAPPING_PERCENTAGE': BOOTSTRAPPING_PERCENTAGE
}


# TODO create csv exports
#  some visualisation of generations
#  create for example 1000 runs of each parameter setting and create a boxplot visualisation of the final averages

class FirstLayer:

    def __init__(self, pset, grid_points):
        self.pset = pset
        self.toolbox = None
        self.grid_points = self.__get_points_for_run(grid_points)

    def __get_points_for_run(self, new_grid_points):
        # retrieve a certain percentage of points from the original set based on BOOTSTRAPPING_PERCENTAGE
        amount_of_points = round(len(new_grid_points) * (BOOTSTRAPPING_PERCENTAGE / 100))
        indices = np.arange(len(new_grid_points))
        selected_indices = np.random.choice(indices, size=amount_of_points, replace=False)
        return new_grid_points[selected_indices]

    # Define the fitness measure
    def __evaluate_individual(self, individual):
        try:
            function = gp.compile(expr=individual, pset=self.pset)
            errors = []
            for point in self.grid_points:
                x, y = point
                individual_output = function(x, y)
                error = abs(target_polynomial(x, y) - individual_output)
                errors.append(error)
            total_error = sum(errors)
            return total_error,
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return float('inf'),  # Return a high fitness in case of an error

    # self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT)) TODO doesnt work => produce an exception when trying to get tree height using a stack
    # self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_TREE_HEIGHT))

    def __initialize_toolbox(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=self.pset)
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genGrow, pset=self.pset, min_=MIN_TREE_INIT_HEIGHT, max_=MAX_TREE_INIT_HEIGHT)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", koza_custom_two_point_crossover)
        self.toolbox.register("mutate", gp.mutNodeReplacement, pset=self.pset)
        self.toolbox.register("evaluate", self.__evaluate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

    def first_layer_evolution(self, process_id, new_terminal_list):
        try:
            self.__initialize_toolbox()
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
                print("Exception in first layer generation loop")
                traceback.print_exc()
        new_terminal_list += population
        if len(population) != 0:
            best_current_individual = tools.selBest(population, k=1)[0]
            fitness_values.append(best_current_individual.fitness.values[0])
            print(f"Best individual: {best_current_individual}, Fitness: {best_current_individual.fitness.values[0]}")


if __name__ == "__main__":
    try:
        manager = multiprocessing.Manager()
        new_terminals = manager.list()
        gp_first_layer_initializer = GpFirstLayerInitializer()
        gp_first_layer_initializer.initialize_gp_run()
        X, Y = np.meshgrid(X_RANGE, Y_RANGE)
        grid_points = np.column_stack((X.flatten(), Y.flatten()))
        first_layer_instance = FirstLayer(gp_first_layer_initializer.pset, grid_points)
        with concurrent.futures.ProcessPoolExecutor() as executor:

            futures = [executor.submit(first_layer_instance.first_layer_evolution, process_id, new_terminals) for
                       process_id
                       in
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

        gp_second_layer_initializer = GpSecondLayerInitializer(terminal_map)
        gp_second_layer_initializer.initialize_gp_run()
        second_layer = SecondLayer(gp_first_layer_initializer.pset, gp_second_layer_initializer.pset)
        best_overall_individual = second_layer.execute_run()
        csvExporter = CsvExporter(first_layer_params, second_layer.second_layer_params, best_overall_individual)
        csvExporter.export_run_data_to_csv()
    except:
        traceback.print_exc()

    # print("Fitness values: " + str(fitness_values))
    # print ("Avg fitness value: ")
