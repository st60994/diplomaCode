from deap import gp, creator, base, tools, algorithms

from booleanMultiplexer.gpBooleanMultiplexerInitialization import MAX_TREE_HEIGHT
from customLogic import koza_custom_two_point_crossover, trim_individual, koza_over_selection, gp_evolution


class SecondLayerMultiplexer:
    TOURNAMENT_SIZE = 2
    ELITES_SIZE = 1
    NUMBER_OF_GENERATIONS = 51
    POPULATION_SIZE = 500
    MIN_TREE_INIT_HEIGHT = 2
    MAX_TREE_INIT_HEIGHT = 6
    CROSSOVER_PROBABILITY = 0.9
    MUTATION_PROBABILITY = 0.05

    second_layer_params = {
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

    pset = None
    first_layer_pset = None

    def __init__(self, first_layer_pset, second_layer_pset, csv_exporter):
        self.first_layer_pset = first_layer_pset
        self.pset = second_layer_pset
        self.input_combinations = self.generate_all_possible_input_combination()
        self.csv_exporter = csv_exporter

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

    def generate_all_possible_input_combination(self):
        all_combinations = []
        for x in range(2048):
            all_combinations.append('{0:011b}'.format(
                x))  # https://stackoverflow.com/questions/10411085/converting-integer-to-binary-in-python
        return all_combinations

    def __prepare_run(self):
        # creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=self.pset)
        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genGrow, pset=self.pset, min_=self.MIN_TREE_INIT_HEIGHT,
                         max_=self.MAX_TREE_INIT_HEIGHT)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", koza_custom_two_point_crossover)
        toolbox.register("mutate", gp.mutNodeReplacement, pset=self.pset)
        toolbox.register("evaluate", self.__evaluate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=self.TOURNAMENT_SIZE)

        self.toolbox.register("trim", trim_individual, max_tree_height=MAX_TREE_HEIGHT, pset=self.pset,
                              csv_export=self.csv_exporter)
        return toolbox

    def __second_layer_evolution(self, toolbox):
        fitness_values = []
        population = toolbox.population(n=self.POPULATION_SIZE)
        algorithms.eaSimple(population, toolbox, cxpb=0.9, mutpb=0.01, ngen=self.NUMBER_OF_GENERATIONS, stats=None,
                            verbose=False)
        if len(population) != 0:
            best_current_individual = tools.selBest(population, k=1)[0]
            fitness_values.append(best_current_individual.fitness.values[0])
            print(f"Best individual: {best_current_individual}, Fitness: {best_current_individual.fitness.values[0]}")
            return best_current_individual

    def execute_run(self):
        toolbox = self.__prepare_run()
        return gp_evolution(0, None, self.ELITES_SIZE, self.POPULATION_SIZE, self.NUMBER_OF_GENERATIONS,
                            self.CROSSOVER_PROBABILITY, self.MUTATION_PROBABILITY, 0, toolbox, self.csv_exporter, 2,
                            "MUX")
