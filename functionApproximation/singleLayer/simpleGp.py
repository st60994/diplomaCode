from datetime import datetime

import numpy as np
from deap import base, creator, tools, gp
import traceback
import multiprocessing

from csvExport import CsvExporter
from customLogic import koza_custom_two_point_crossover, trim_individual, gp_evolution
from functionApproximation.gpInitialization import MAX_TREE_HEIGHT, LOWER_BOUND_X, UPPER_BOUND_X, LOWER_BOUND_Y, \
    UPPER_BOUND_Y, STEP_SIZE_X, STEP_SIZE_Y, target_polynomial, NUMBER_OF_RUNS, X_RANGE, Y_RANGE, \
    GpFirstLayerInitializer

TOURNAMENT_SIZE = 2
ELITES_SIZE = 1
NUMBER_OF_GENERATIONS = 100
POPULATION_SIZE = 200
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
    'MAX_TREE_HEIGHT': MAX_TREE_HEIGHT,
    'MIN_TREE_INIT_HEIGHT': MIN_TREE_INIT_HEIGHT,
    'MAX_TREE_INIT_HEIGHT': MAX_TREE_INIT_HEIGHT,
    'LOWER_BOUND_X': LOWER_BOUND_X,
    'UPPER_BOUND_X': UPPER_BOUND_X,
    'LOWER_BOUND_Y': LOWER_BOUND_Y,
    'UPPER_BOUND_Y': UPPER_BOUND_Y,
    'STEP_SIZE_X': STEP_SIZE_X,
    'STEP_SIZE_Y': STEP_SIZE_Y,
    'TERMINALS_FROM_FIRST_LAYER': TERMINALS_FROM_FIRST_LAYER,
    'CROSSOVER_PROBABILITY': CROSSOVER_PROBABILITY,
    'MUTATION_PROBABILITY': MUTATION_PROBABILITY,
}


class FirstLayer:

    def __init__(self, pset, grid_points, csv_exporter):
        self.pset = pset
        self.toolbox = None
        self.grid_points = grid_points
        self.csv_exporter: CsvExporter = csv_exporter
        self.number_of_approximations = 0

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

    def __evaluate_individual_mse(self, individual):
        try:
            function = gp.compile(expr=individual, pset=self.pset)
            errors = []
            for point in self.grid_points:
                x, y = point
                individual_output = function(x, y)
                error = pow(target_polynomial(x, y) - individual_output, 2)
                errors.append(error)
            total_error = sum(errors) / len(errors)
            self.number_of_approximations += 1
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
        self.toolbox.register("evaluate", self.__evaluate_individual_mse)
        self.toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
        self.toolbox.register("trim", trim_individual, max_tree_height=MAX_TREE_HEIGHT, pset=self.pset,
                              csv_export=self.csv_exporter)


if __name__ == "__main__":
    try:
        now = datetime.now()
        csvExporter = CsvExporter(now)
        manager = multiprocessing.Manager()
        gp_first_layer_initializer = GpFirstLayerInitializer()
        gp_first_layer_initializer.initialize_gp_run()
        X, Y = np.meshgrid(X_RANGE, Y_RANGE)
        grid_points = np.column_stack((X.flatten(), Y.flatten()))
        for run_number in range(NUMBER_OF_RUNS):
            print("Starting run " + str(run_number))
            first_layer_instance = FirstLayer(gp_first_layer_initializer.pset, grid_points, csvExporter)
            first_layer_instance.initialize_toolbox()
            best_overall_individual = gp_evolution(0, None, ELITES_SIZE, POPULATION_SIZE, NUMBER_OF_GENERATIONS,
                                                   CROSSOVER_PROBABILITY, MUTATION_PROBABILITY, 0,
                                                   first_layer_instance.toolbox, csvExporter, 1, "approximation", None, 1)
            if run_number == 0:
                csvExporter.export_run_params_to_csv(first_layer_params, {})
            csvExporter.save_best_individual(best_overall_individual, run_number)
            csvExporter.save_number_of_approximations(first_layer_instance.number_of_approximations, run_number)
    except:
        traceback.print_exc()
