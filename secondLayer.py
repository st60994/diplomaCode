import numpy as np
from deap import gp, creator, base, tools, algorithms

from customLogic import koza_custom_two_point_crossover
from gpInitialization import target_polynomial


class SecondLayer:
    LOWER_BOUND_X = 0.1
    UPPER_BOUND_X = 15.0
    LOWER_BOUND_Y = 0.1
    UPPER_BOUND_Y = 15.0
    STEP_SIZE_X = 0.1
    STEP_SIZE_Y = 0.1
    X_RANGE = np.arange(LOWER_BOUND_X, UPPER_BOUND_X + STEP_SIZE_X, STEP_SIZE_X)
    Y_RANGE = np.arange(LOWER_BOUND_Y, UPPER_BOUND_Y + STEP_SIZE_Y, STEP_SIZE_Y)
    TOURNAMENT_SIZE = 2
    ELITES_SIZE = 1
    NUMBER_OF_GENERATIONS = 30
    POPULATION_SIZE = 50
    MAX_TREE_HEIGHT = 65
    MIN_TREE_INIT_HEIGHT = 3
    MAX_TREE_INIT_HEIGHT = 3

    second_layer_params = {
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
    }

    pset = None
    first_layer_pset = None

    def __init__(self, first_layer_pset, second_layer_pset):
        self.first_layer_pset = first_layer_pset
        self.pset = second_layer_pset

    # Define the fitness measure
    def __evaluate_individual(self, individual):
        try:
            compiled_individual = gp.compile(expr=individual, pset=self.pset)

            x_values = self.X_RANGE
            y_values = self.X_RANGE
            errors = []
            for x in x_values:
                for y in y_values:
                    individual_output = compiled_individual(x, y)
                    error = abs(target_polynomial(x, y) - individual_output)
                    errors.append(error)
            total_error = sum(errors)
            return total_error,
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return float('inf'),  # Return a high fitness in case of an error

    def __prepare_run(self):
        # creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual2", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=self.pset)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genGrow, pset=self.pset, min_=self.MIN_TREE_INIT_HEIGHT,
                         max_=self.MAX_TREE_INIT_HEIGHT)
        toolbox.register("individual", tools.initIterate, creator.Individual2, toolbox.expr)

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        # gp.staticLimit(operator.attrgetter('height'), max_value=self.MAX_TREE_HEIGHT) # TODO doesnt work either
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
            return best_current_individual

    def execute_run(self):
        toolbox = self.__prepare_run()
        return self.__second_layer_evolution(toolbox)
