import concurrent.futures
from datetime import datetime

from deap import algorithms, base, creator, tools, gp
import traceback
import multiprocessing

from booleanMultiplexer.gpBooleanMultiplexerInitialization import GpFirstLayerMUXInitializer, NUMBER_OF_RUNS, \
    MAX_TREE_HEIGHT, GpSecondLayerInitializer
from booleanMultiplexer.secondLayerMultiplexer import SecondLayerMultiplexer
from csvExport import CsvExporter
from customLogic import koza_custom_two_point_crossover, trim_individual, koza_over_selection

TOURNAMENT_SIZE = 2
ELITES_SIZE = 1
NUMBER_OF_GENERATIONS = 2
POPULATION_SIZE = 4000
NUMBER_OF_SUB_MODELS = 8
MIN_TREE_INIT_HEIGHT = 4
MAX_TREE_INIT_HEIGHT = 6
TERMINALS_FROM_FIRST_LAYER = 1

first_layer_params = {
    'TOURNAMENT_SIZE': TOURNAMENT_SIZE,
    'ELITES_SIZE': ELITES_SIZE,
    'NUMBER_OF_GENERATIONS': NUMBER_OF_GENERATIONS,
    'POPULATION_SIZE': POPULATION_SIZE,
    'NUMBER_OF_SUB_MODELS': NUMBER_OF_SUB_MODELS,
    'MAX_TREE_HEIGHT': MAX_TREE_HEIGHT,
    'MIN_TREE_INIT_HEIGHT': MIN_TREE_INIT_HEIGHT,
    'MAX_TREE_INIT_HEIGHT': MAX_TREE_INIT_HEIGHT,
    'TERMINALS_FROM_FIRST_LAYER': TERMINALS_FROM_FIRST_LAYER
}


class FirstLayer:
    input_combinations = []
    def __init__(self, pset, csv_exporter):
        self.pset = pset
        self.toolbox = None
        self.csv_exporter: CsvExporter = csv_exporter

    # Define the fitness measure
    def __evaluate_individual(self, individual):
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

    def __generate_all_input_combinations_for_model(self, process_id):
        all_combinations = []
        process_id_bin_string = '{0:03b}'.format(process_id)
        for x in range(256):
            all_combinations.append(process_id_bin_string + '{0:08b}'.format(
                x))  # https://stackoverflow.com/questions/10411085/converting-integer-to-binary-in-python
        return all_combinations

    def __initialize_toolbox(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genGrow, pset=self.pset, min_=MIN_TREE_INIT_HEIGHT, max_=MAX_TREE_INIT_HEIGHT)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", koza_custom_two_point_crossover)
        self.toolbox.register("mutate", gp.mutNodeReplacement, pset=self.pset)
        self.toolbox.register("evaluate", self.__evaluate_individual)
        self.toolbox.register("select", koza_over_selection, tournsize=TOURNAMENT_SIZE, population_size=POPULATION_SIZE)
        self.toolbox.register("trim", trim_individual, max_tree_height=MAX_TREE_HEIGHT)

    def first_layer_evolution(self, process_id, new_terminal_list):
        try:
            self.__initialize_toolbox()
            self.input_combinations = self.__generate_all_input_combinations_for_model(process_id)
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
                for individual in offspring:
                    self.toolbox.trim(individual)

                # Need to manually evaluate the offspring
                fitnesses = self.toolbox.map(self.toolbox.evaluate, offspring)
                for ind, fit in zip(offspring, fitnesses):
                    ind.fitness.values = fit
                print("Avg fitness: " + str(self.__calculate_avg_fitness(offspring)))
                combined_population = population + offspring

                # Implementation of elitism
                hall_of_fame.update(combined_population)
                offspring = self.toolbox.select(combined_population, POPULATION_SIZE)

                # Replace the current population by the offspring
                population[:] = offspring
                if process_id == 0:  # Only save the for process_id = 0 to avoid unnecessary delays
                    self.csv_exporter.save_best_individual_for_each_generation(tools.selBest(population, k=1)[0], index)
                combined_population += population
            except:
                print("Exception in first layer generation loop")
                traceback.print_exc()
        if TERMINALS_FROM_FIRST_LAYER == 1:
            new_terminal_list.append(tools.selBest(population, k=1)[0])
        else:
            new_terminal_list += population
        if len(population) != 0:
            best_current_individual = tools.selBest(population, k=1)[0]
            fitness_values.append(best_current_individual.fitness.values[0])
            print(f"Best individual: {best_current_individual}, Fitness: {best_current_individual.fitness.values[0]}")
            return best_current_individual

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
        gp_first_layer_initializer = GpFirstLayerMUXInitializer()
        gp_first_layer_initializer.initialize_gp_run()
        for run_number in range(NUMBER_OF_RUNS):
            print("Starting run " + str(run_number))
            new_terminals = manager.list()
            first_layer_instance = FirstLayer(gp_first_layer_initializer.pset, csvExporter)
            with concurrent.futures.ProcessPoolExecutor() as executor:

                futures = [executor.submit(first_layer_instance.first_layer_evolution, process_id, new_terminals) for
                           process_id
                           in
                           range(NUMBER_OF_SUB_MODELS)]

                concurrent.futures.wait(futures)
            csvExporter.save_sub_models(new_terminals, run_number)
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
            second_layer = SecondLayerMultiplexer(gp_first_layer_initializer.pset, gp_second_layer_initializer.pset)
            if run_number == 0:
                csvExporter.export_run_params_to_csv(first_layer_params, second_layer.second_layer_params)
            best_overall_individual = second_layer.execute_run()
            csvExporter.save_best_individual(best_overall_individual, run_number)
    except:
        traceback.print_exc()

    # print("Fitness values: " + str(fitness_values))
    # print ("Avg fitness value: ")
