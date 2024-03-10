import concurrent.futures
from datetime import datetime

import numpy as np
from deap import base, creator, tools, gp
import traceback
import multiprocessing

from csvExport import CsvExporter
from customLogic import koza_custom_two_point_crossover, trim_individual, gp_evolution
from gpInitialization import GpFirstLayerInitializer, GpSecondLayerInitializer, target_polynomial, LOWER_BOUND_X, \
    UPPER_BOUND_X, LOWER_BOUND_Y, UPPER_BOUND_Y, STEP_SIZE_X, STEP_SIZE_Y, X_RANGE, Y_RANGE, NUMBER_OF_RUNS, \
    MAX_TREE_HEIGHT
from secondLayer import SecondLayer

BOOTSTRAPPING_PERCENTAGE = 100

TOURNAMENT_SIZE = 2
ELITES_SIZE = 1
NUMBER_OF_GENERATIONS = 3
POPULATION_SIZE = 50
NUMBER_OF_SUB_MODELS = 30
MIN_TREE_INIT_HEIGHT = 2
MAX_TREE_INIT_HEIGHT = 6
TERMINALS_FROM_FIRST_LAYER = 1
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.01

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
    'BOOTSTRAPPING_PERCENTAGE': BOOTSTRAPPING_PERCENTAGE,
    'TERMINALS_FROM_FIRST_LAYER': TERMINALS_FROM_FIRST_LAYER,
    'CROSSOVER_PROBABILITY': CROSSOVER_PROBABILITY,
    'MUTATION_PROBABILITY': MUTATION_PROBABILITY,
}


class FirstLayer:

    def __init__(self, pset, grid_points, csv_exporter):
        self.pset = pset
        self.toolbox = None
        self.grid_points = self.__get_points_for_run(grid_points)
        self.csv_exporter: CsvExporter = csv_exporter

    def __get_points_for_run(self, new_grid_points):
        # retrieve a certain percentage of points from the original set based on BOOTSTRAPPING_PERCENTAGE
        amount_of_points = round(len(new_grid_points) * (BOOTSTRAPPING_PERCENTAGE / 100))
        indices = np.arange(len(new_grid_points))
        selected_indices = np.random.choice(indices, size=amount_of_points, replace=False)
        return new_grid_points[selected_indices]

    # Define the fitness measure, unused
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

    def evaluate_individual_mse(self, individual):
        try:
            function = gp.compile(expr=individual, pset=self.pset)
            errors = []
            for point in self.grid_points:
                x, y = point
                individual_output = function(x, y)
                error = pow(target_polynomial(x, y) - individual_output, 2)
                errors.append(error)
            total_error = sum(errors) / len(errors)
            return total_error,
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return float('inf')

    def initialize_toolbox(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=self.pset)
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genGrow, pset=self.pset, min_=MIN_TREE_INIT_HEIGHT, max_=MAX_TREE_INIT_HEIGHT)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", koza_custom_two_point_crossover)
        self.toolbox.register("mutate", gp.mutNodeReplacement, pset=self.pset)
        self.toolbox.register("evaluate", self.evaluate_individual_mse)
        self.toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
        self.toolbox.register("trim", trim_individual, max_tree_height=MAX_TREE_HEIGHT, pset=self.pset,
                              csv_export=self.csv_exporter)


if __name__ == "__main__":
    try:
        now = datetime.now()
        csv_exporter = CsvExporter(now)
        manager = multiprocessing.Manager()
        gp_first_layer_initializer = GpFirstLayerInitializer()
        gp_first_layer_initializer.initialize_gp_run()
        X, Y = np.meshgrid(X_RANGE, Y_RANGE)
        grid_points = np.column_stack((X.flatten(), Y.flatten()))
        for run_number in range(NUMBER_OF_RUNS):
            print("Starting run " + str(run_number))
            new_terminals = manager.list()
            first_layer_instance = FirstLayer(gp_first_layer_initializer.pset, grid_points, csv_exporter)
            first_layer_instance.initialize_toolbox()
            print("toolbox init")
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(gp_evolution, process_id, new_terminals, ELITES_SIZE, POPULATION_SIZE,
                                           NUMBER_OF_GENERATIONS, CROSSOVER_PROBABILITY,
                                           MUTATION_PROBABILITY, TERMINALS_FROM_FIRST_LAYER,
                                           first_layer_instance.toolbox, first_layer_instance.csv_exporter, 1, "approximation", None, 2) for
                           process_id
                           in
                           range(NUMBER_OF_SUB_MODELS)]
            concurrent.futures.wait(futures)
            csv_exporter.save_sub_models(new_terminals, run_number)
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

            print("Best fitness from first layer: " + str(best_fitness))
            print("---------Second layer---------")

            gp_second_layer_initializer = GpSecondLayerInitializer(terminal_map)
            gp_second_layer_initializer.initialize_gp_run()
            second_layer = SecondLayer(gp_first_layer_initializer.pset, gp_second_layer_initializer.pset, csv_exporter)
            if run_number == 0:
                csv_exporter.export_run_params_to_csv(first_layer_params, second_layer.second_layer_params)
            best_overall_individual = second_layer.execute_run()
            csv_exporter.save_best_individual(best_overall_individual, run_number)
    except:
        traceback.print_exc()

    # print("Fitness values: " + str(fitness_values))
    # print ("Avg fitness value: ")
