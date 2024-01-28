import operator
import numpy as np
from deap import gp, creator, base, tools, algorithms

from customLogic import koza_custom_two_point_crossover
from methodDefinitions import protectedAdd, sqrt, pow2, pow3
from util import draw_individual


def target_polynomial(x):
    return 1 / x


class SecondLayer:
    LOWER_BOUND = 0.1
    UPPER_BOUND = 15.0
    STEP_SIZE = 0.1
    X_RANGE = np.arange(LOWER_BOUND, UPPER_BOUND + STEP_SIZE, STEP_SIZE)
    TOURNAMENT_SIZE = 2
    ELITES_SIZE = 1
    NUMBER_OF_GENERATIONS = 30
    POPULATION_SIZE = 50
    MAX_TREE_HEIGHT = 65
    MIN_TREE_INIT = 3
    MAX_TREE_INIT = 3

    subsets = []
    pset = None
    first_layer_data = None
    terminal_values = {}

    def __init__(self, subsets):
        self.subsets = subsets

    def __add_primitive_set(self, pset):
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(protectedAdd, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(sqrt, 1)
        # pset.addPrimitive(sin, 1)
        pset.addPrimitive(pow2, 1)
        pset.addPrimitive(pow3, 1)
        #   pset.addPrimitive(avg, 1)
        return pset

    def __create_terminal_set(self, pset):

        pset.addTerminal(-1.0)
        pset.addTerminal(1.0)
        pset.addTerminal(2.0)
        pset.addTerminal(3.0)
        for individual in self.subsets:
            if individual not in self.subsets.keys:  # check if the subset is not already present as a terminal in the second layer
                pset.addTerminal(self.subsets[individual], name=str(individual))
        print("Primitive Set Terminals:", pset.terminals)
        pset.renameArguments(ARG0="x")
        return pset

    # Define the fitness measure
    def __evaluate_individual(self, individual):
        try:
            compiled_individual = gp.compile(expr=individual, pset=self.pset)

            x_values = self.X_RANGE
            errors = []
            for i in x_values:
                individual_output = compiled_individual(i)
                error = abs(target_polynomial(i) - individual_output)
                errors.append(error)
            total_error = sum(errors)
            return total_error,
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return float('inf'),  # Return a high fitness in case of an error

    def __prepare_run(self):
        self.pset = gp.PrimitiveSet("MAIN", 1)  # 1 input variable
        self.pset = self.__create_terminal_set(self.pset)
        self.pset = self.__add_primitive_set(self.pset)

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=self.pset)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genGrow, pset=self.pset, min_=self.MIN_TREE_INIT, max_=self.MAX_TREE_INIT)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        gp.staticLimit(operator.attrgetter('height'), max_value=self.MAX_TREE_HEIGHT)
        toolbox.register("evaluate", self.__evaluate_individual)
        toolbox.register("mate", koza_custom_two_point_crossover)
        toolbox.register("mutate", gp.mutNodeReplacement, pset=self.pset)
        toolbox.register("select", tools.selTournament, tournsize=self.TOURNAMENT_SIZE)
        return toolbox

    def __second_layer_evolution(self, toolbox):
        fitness_values = []
        population = toolbox.population(n=self.POPULATION_SIZE)
        algorithms.eaSimple(population, toolbox, cxpb=0.9, mutpb=0.01, ngen=self.NUMBER_OF_GENERATIONS, stats=None,
                            verbose=True)
        if len(population) != 0:
            best_current_individual = tools.selBest(population, k=1)[0]
            fitness_values.append(best_current_individual.fitness.values[0])
            print(f"Best individual: {best_current_individual}, Fitness: {best_current_individual.fitness.values[0]}")
            draw_individual(best_current_individual)

    def execute_run(self):
        toolbox = self.__prepare_run()
        self.__second_layer_evolution(toolbox)
