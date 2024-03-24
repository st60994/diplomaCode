import concurrent.futures
import random
from datetime import datetime

import numpy as np
from deap import base, creator, tools, gp
import traceback
import multiprocessing

from booleanMultiplexer.gpBooleanMultiplexerInitialization import GpFirstLayerMUXInitializer, NUMBER_OF_RUNS, \
    MAX_TREE_HEIGHT, GpSecondLayerInitializer
from booleanMultiplexer.secondLayerMultiplexer import SecondLayerMultiplexer
from csvExport import CsvExporter
from customLogic import koza_custom_two_point_crossover, trim_individual, gp_evolution

TOURNAMENT_SIZE = 2
ELITES_SIZE = 1
NUMBER_OF_GENERATIONS = 51
POPULATION_SIZE = 50
NUMBER_OF_SUB_MODELS = 8
MIN_TREE_INIT_HEIGHT = 2
MAX_TREE_INIT_HEIGHT = 5
TERMINALS_FROM_FIRST_LAYER = 1
NUMBER_OF_ADDRESSES_TO_APPROXIMATE = 4
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.05

first_layer_params = {
    'TOURNAMENT_SIZE': TOURNAMENT_SIZE,
    'ELITES_SIZE': ELITES_SIZE,
    'NUMBER_OF_GENERATIONS': NUMBER_OF_GENERATIONS,
    'POPULATION_SIZE': POPULATION_SIZE,
    'NUMBER_OF_SUB_MODELS': NUMBER_OF_SUB_MODELS,
    'MAX_TREE_HEIGHT': MAX_TREE_HEIGHT,
    'MIN_TREE_INIT_HEIGHT': MIN_TREE_INIT_HEIGHT,
    'MAX_TREE_INIT_HEIGHT': MAX_TREE_INIT_HEIGHT,
    'TERMINALS_FROM_FIRST_LAYER': TERMINALS_FROM_FIRST_LAYER,
    'NUMBER_OF_ADDRESSES_TO_APPROXIMATE': NUMBER_OF_ADDRESSES_TO_APPROXIMATE,
    'CROSSOVER_PROBABILITY': CROSSOVER_PROBABILITY,
    'MUTATION_PROBABILITY': MUTATION_PROBABILITY,
}


class FirstLayer:
    input_combinations = []

    def __init__(self, pset, csv_exporter):
        self.pset = pset
        self.toolbox = None
        self.csv_exporter: CsvExporter = csv_exporter

    # Define the fitness measure
    def evaluate_individual(self, individual, input_combinations):
        function = gp.compile(expr=individual, pset=self.pset)
        correct_assessments = 0
        for input_combination in input_combinations:
            expected_output = bool(self.__get_expected_output(input_combination))
            actual_output = function(
                bool(int(input_combination[0])), bool(int(input_combination[1])), bool(int(input_combination[2])),
                bool(int(input_combination[3])), bool(int(input_combination[4])), bool(int(input_combination[5])),
                bool(int(input_combination[6])), bool(int(input_combination[7])), bool(int(input_combination[8])),
                bool(int(input_combination[9])), bool(int(input_combination[10])))
            if expected_output == actual_output:
                correct_assessments += 1
        return correct_assessments,

    def __get_expected_output(self, input_combination):
        address_string = input_combination[:3]
        address = int(address_string, 2)
        return int(input_combination[10 - address])

    def initialize_toolbox(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genGrow, pset=self.pset, min_=MIN_TREE_INIT_HEIGHT, max_=MAX_TREE_INIT_HEIGHT)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", koza_custom_two_point_crossover)
        self.toolbox.register("mutate", gp.mutNodeReplacement, pset=self.pset)
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
        self.toolbox.register("trim", trim_individual, max_tree_height=MAX_TREE_HEIGHT, pset=self.pset,
                              csv_export=self.csv_exporter, second_layer=False)

    def __calculate_avg_fitness(self, population):
        total_fitness = 0
        for individual in population:
            fitness = individual.fitness.values[0]
            total_fitness += fitness
        return total_fitness / len(population)


if __name__ == "__main__":
    try:
        now = datetime.now()
        csv_exporter = CsvExporter(now)
        manager = multiprocessing.Manager()
        gp_first_layer_initializer = GpFirstLayerMUXInitializer()
        gp_first_layer_initializer.initialize_gp_run()
        for run_number in range(NUMBER_OF_RUNS):
            print("Starting run " + str(run_number))
            new_terminals = manager.list()
            first_layer_instance = FirstLayer(gp_first_layer_initializer.pset, csv_exporter)
            first_layer_instance.initialize_toolbox()
            address_pool = list(range(0, 8))
            process_addresses_map = {}
            for process_id in range(NUMBER_OF_SUB_MODELS):
                if len(address_pool) < NUMBER_OF_ADDRESSES_TO_APPROXIMATE:
                    address_pool = list(
                        range(0, 8))  # TODO not a great solution: could make some addresses not present in final map
                process_addresses_map[process_id] = random.sample(address_pool, NUMBER_OF_ADDRESSES_TO_APPROXIMATE)
                for address in process_addresses_map[process_id]:
                    address_pool.remove(address)
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(gp_evolution, process_id, new_terminals, ELITES_SIZE, POPULATION_SIZE,
                                           NUMBER_OF_GENERATIONS, CROSSOVER_PROBABILITY,
                                           MUTATION_PROBABILITY, TERMINALS_FROM_FIRST_LAYER,
                                           first_layer_instance.toolbox, first_layer_instance.csv_exporter, 1, "MUX",
                                           process_addresses_map, 2)
                           for
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
            second_layer = SecondLayerMultiplexer(gp_first_layer_initializer.pset, gp_second_layer_initializer.pset,
                                                  csv_exporter)
            if run_number == 0:
                csv_exporter.export_run_params_to_csv(first_layer_params, second_layer.second_layer_params)
            best_overall_individual = second_layer.execute_run()
            csv_exporter.save_best_individual(best_overall_individual, run_number)
    except:
        traceback.print_exc()

    # print("Fitness values: " + str(fitness_values))
    # print ("Avg fitness value: ")
