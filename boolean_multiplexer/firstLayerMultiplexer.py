from datetime import datetime

from deap import algorithms, base, creator, tools, gp
import traceback
import multiprocessing

from boolean_multiplexer.gpBooleanMultiplexerInitialization import GpFirstLayerMUXInitializer, NUMBER_OF_RUNS, \
    MAX_TREE_HEIGHT
from csvExport import CsvExporter
from customLogic import koza_custom_two_point_crossover, trim_individual
from util import draw_individual

BOOTSTRAPPING_PERCENTAGE = 60

TOURNAMENT_SIZE = 2
ELITES_SIZE = 1
NUMBER_OF_GENERATIONS = 51
POPULATION_SIZE = 4000
NUMBER_OF_SUB_MODELS = 1
MIN_TREE_INIT_HEIGHT = 2
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
    'BOOTSTRAPPING_PERCENTAGE': BOOTSTRAPPING_PERCENTAGE,
    'TERMINALS_FROM_FIRST_LAYER': TERMINALS_FROM_FIRST_LAYER
}


class FirstLayer:

    def __init__(self, pset, csv_exporter):
        self.pset = pset
        self.toolbox = None
        self.csv_exporter: CsvExporter = csv_exporter
        self.input_combinations = self.__generate_all_possible_input_combination()

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

    def __generate_all_possible_input_combination(self):
        all_combinations = []
        for x in range(2048):
            all_combinations.append('{0:011b}'.format(x)) # https://stackoverflow.com/questions/10411085/converting-integer-to-binary-in-python
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
        self.toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)
        self.toolbox.register("trim", trim_individual)

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
        gp_boolean_multiplexer_init = GpFirstLayerMUXInitializer()
        gp_boolean_multiplexer_init.initialize_gp_run()
        for run_number in range(NUMBER_OF_RUNS):
            print("Starting run " + str(run_number))
            new_terminals = manager.list()
            first_layer_instance = FirstLayer(gp_boolean_multiplexer_init.pset, csvExporter)
            best_individual = first_layer_instance.first_layer_evolution(0, new_terminals)
            csvExporter.save_sub_models(new_terminals, run_number)
            csvExporter.save_best_individual(best_individual, run_number)
            if best_individual.fitness.values[0] == 2048.0:
                break
            draw_individual(best_individual, first_layer_instance.pset)
    except:
        traceback.print_exc()

    # print("Fitness values: " + str(fitness_values))
    # print ("Avg fitness value: ")
