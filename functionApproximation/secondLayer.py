from deap import gp, creator, base, tools, algorithms

from customLogic import koza_custom_two_point_crossover, trim_individual
from gpInitialization import target_polynomial, X_RANGE, Y_RANGE, MAX_TREE_HEIGHT


class SecondLayer:
    TOURNAMENT_SIZE = 2
    ELITES_SIZE = 1
    NUMBER_OF_GENERATIONS = 10
    POPULATION_SIZE = 200
    MIN_TREE_INIT_HEIGHT = 2
    MAX_TREE_INIT_HEIGHT = 6

    second_layer_params = {
        'TOURNAMENT_SIZE': TOURNAMENT_SIZE,
        'ELITES_SIZE': ELITES_SIZE,
        'NUMBER_OF_GENERATIONS': NUMBER_OF_GENERATIONS,
        'POPULATION_SIZE': POPULATION_SIZE,
        'MAX_TREE_HEIGHT': MAX_TREE_HEIGHT,
        'MIN_TREE_INIT_HEIGHT': MIN_TREE_INIT_HEIGHT,
        'MAX_TREE_INIT_HEIGHT': MAX_TREE_INIT_HEIGHT,
    }

    pset = None
    first_layer_pset = None

    def __init__(self, first_layer_pset, second_layer_pset):
        self.first_layer_pset = first_layer_pset
        self.pset = second_layer_pset
        self.number_of_approximations = 0

    # Define the fitness measure
    def __evaluate_individual(self, individual):
        try:
            compiled_individual = gp.compile(expr=individual, pset=self.pset)

            x_values = X_RANGE
            y_values = Y_RANGE
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

    def __evaluate_individual_mse(self, individual):
        try:
            compiled_individual = gp.compile(expr=individual, pset=self.pset)

            x_values = X_RANGE
            y_values = Y_RANGE
            errors = []
            for x in x_values:
                for y in y_values:
                    individual_output = compiled_individual(x, y)
                    error = pow(target_polynomial(x, y) - individual_output, 2)
                    errors.append(error)
            total_error = sum(errors) / len(errors)
            self.number_of_approximations += 1
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
        toolbox.register("evaluate", self.__evaluate_individual_mse)
        toolbox.register("mate", koza_custom_two_point_crossover)
        toolbox.register("trim", trim_individual, max_tree_height=MAX_TREE_HEIGHT, csv_export=self.csv_exporter)
        toolbox.register("mutate", gp.mutNodeReplacement, pset=self.pset)
        toolbox.register("select", tools.selTournament, tournsize=self.TOURNAMENT_SIZE)
        return toolbox

    def __second_layer_evolution(self, toolbox):
        fitness_values = []
        population = toolbox.population(n=self.POPULATION_SIZE)
        algorithms.eaSimple(population, toolbox, cxpb=0.9, mutpb=0.01, ngen=self.NUMBER_OF_GENERATIONS, stats=None,
                            verbose=False)
        print("Second layer approx: " + str(self.number_of_approximations))
        if len(population) != 0:
            best_current_individual = tools.selBest(population, k=1)[0]
            fitness_values.append(best_current_individual.fitness.values[0])
            print(f"Best individual: {best_current_individual}, Fitness: {best_current_individual.fitness.values[0]}")
            return best_current_individual

    def execute_run(self):
        toolbox = self.__prepare_run()
        return self.__second_layer_evolution(toolbox)
