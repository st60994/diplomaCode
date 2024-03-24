from datetime import datetime

from deap import base, creator, tools, gp
import traceback
import multiprocessing

from booleanMultiplexer.gpBooleanMultiplexerInitialization import GpFirstLayerMUXInitializer, NUMBER_OF_RUNS, \
    MAX_TREE_HEIGHT
from csvExport import CsvExporter
from customLogic import koza_custom_two_point_crossover, trim_individual, gp_evolution

TOURNAMENT_SIZE = 2
ELITES_SIZE = 1
NUMBER_OF_GENERATIONS = 100
POPULATION_SIZE = 200
MIN_TREE_INIT_HEIGHT = 2
MAX_TREE_INIT_HEIGHT = 6
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.05

first_layer_params = {
    'TOURNAMENT_SIZE': TOURNAMENT_SIZE,
    'ELITES_SIZE': ELITES_SIZE,
    'NUMBER_OF_GENERATIONS': NUMBER_OF_GENERATIONS,
    'POPULATION_SIZE': POPULATION_SIZE,
    'MAX_TREE_HEIGHT': MAX_TREE_HEIGHT,
    'MIN_TREE_INIT_HEIGHT': MIN_TREE_INIT_HEIGHT,
    'MAX_TREE_INIT_HEIGHT': MAX_TREE_INIT_HEIGHT,
    'CROSSOVER_PROBABILITY': CROSSOVER_PROBABILITY,
    'MUTATION_PROBABILITY': MUTATION_PROBABILITY,
}


class FirstLayer:

    def __init__(self, pset, csv_exporter):
        self.pset = pset
        self.toolbox = None
        self.csv_exporter: CsvExporter = csv_exporter
        self.input_combinations = self.generate_all_possible_input_combination()

    # Define the fitness measure
    def __evaluate_individual_MUX(self, individual):
        function = gp.compile(expr=individual, pset=self.pset)
        correct_assessments = 0
        for input_combination in self.input_combinations:
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

    def generate_all_possible_input_combination(self):
        all_combinations = []
        for x in range(2048):
            all_combinations.append('{0:011b}'.format(
                x))  # https://stackoverflow.com/questions/10411085/converting-integer-to-binary-in-python
        return all_combinations

    def initialize_toolbox(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genGrow, pset=self.pset, min_=MIN_TREE_INIT_HEIGHT, max_=MAX_TREE_INIT_HEIGHT)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", koza_custom_two_point_crossover)
        self.toolbox.register("mutate", gp.mutNodeReplacement, pset=self.pset)
        self.toolbox.register("evaluate", self.__evaluate_individual_MUX)
        self.toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE) # no longer need kozas overselection, because the population is quite small
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
        csvExporter = CsvExporter(now)
        manager = multiprocessing.Manager()
        gp_boolean_multiplexer_init = GpFirstLayerMUXInitializer()
        gp_boolean_multiplexer_init.initialize_gp_run()
        for run_number in range(NUMBER_OF_RUNS):
            print("Starting run " + str(run_number))
            first_layer_instance = FirstLayer(gp_boolean_multiplexer_init.pset, csvExporter)
            first_layer_instance.initialize_toolbox()
            best_individual = gp_evolution(0, None, ELITES_SIZE, POPULATION_SIZE, NUMBER_OF_GENERATIONS,
                                           CROSSOVER_PROBABILITY, MUTATION_PROBABILITY, 0,
                                           first_layer_instance.toolbox, csvExporter, 1, "MUX", None, 1)
            if run_number == 0:
                csvExporter.export_run_params_to_csv(first_layer_params, {})
            csvExporter.save_best_individual(best_individual, run_number)
    except:
        traceback.print_exc()

    # print("Fitness values: " + str(fitness_values))
    # print ("Avg fitness value: ")
