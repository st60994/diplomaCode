import random

import deap
from deap import creator, gp, tools, algorithms
from deap.tools import selection
import traceback

from booleanMultiplexer.muxCustomLogic import generate_all_input_combinations_for_model


def koza_custom_two_point_crossover(parent1, parent2):
    if random.uniform(0, 1) <= 0.9:
        indices_of_functions_parent1 = []
        indices_of_functions_parent2 = []
        for i in range(len(parent1)):
            if isinstance(parent1[i], deap.gp.Primitive):
                indices_of_functions_parent1.append(i)
        crossing_index_parent1 = random.choice(indices_of_functions_parent1)
        for i in range(len(parent2)):
            if isinstance(parent2[i], deap.gp.Primitive):
                indices_of_functions_parent2.append(i)
        crossing_index_parent2 = random.choice(indices_of_functions_parent2)
    else:
        indices_of_terminals_parent1 = []
        indices_of_terminals_parent2 = []
        for i in range(len(parent1)):
            if isinstance(parent1[i], deap.gp.Terminal):
                indices_of_terminals_parent1.append(i)
        crossing_index_parent1 = random.choice(indices_of_terminals_parent1)
        for i in range(len(parent2)):
            if isinstance(parent2[i], deap.gp.Terminal):
                indices_of_terminals_parent2.append(i)
        crossing_index_parent2 = random.choice(indices_of_terminals_parent2)

    slice1 = parent1.searchSubtree(crossing_index_parent1)
    slice2 = parent2.searchSubtree(crossing_index_parent2)

    start1, stop1 = slice1.start, slice1.stop
    start2, stop2 = slice2.start, slice2.stop

    # Perform crossover
    combined_tree1 = parent1[:start1] + parent2[start2:stop2] + parent1[stop1:]
    combined_tree2 = parent2[:start2] + parent1[start1:stop1] + parent2[stop2:]
    return creator.Individual(combined_tree1), creator.Individual(combined_tree2)


def get_individual_height(individual):
    stack = [0]
    max_depth = 0
    for elem in individual:
        if stack:  # Check if stack is not empty before popping
            depth = stack.pop()
            max_depth = max(max_depth, depth)
            stack.extend([depth + 1] * elem.arity)
        else:
            break  # Exit the loop if stack is empty
    return max_depth


def get_nodes_with_height(individual, height):
    stack = [0]
    nodes = []
    max_depth = 0
    for i, elem in enumerate(individual):
        if stack:  # Check if stack is not empty before popping
            depth = stack.pop()
            if height == depth:
                nodes.append(i)
            max_depth = max(max_depth, depth)
            stack.extend([depth + 1] * elem.arity)
        else:
            break  # Exit the loop if stack is empty
    return nodes


def trim_individual(individual, max_tree_height, pset, csv_export, second_layer):
    try:
        if second_layer:
            individual = gp.PrimitiveTree.from_string(str(individual),
                                                      pset)  # We have to reparse it to get the real tree height
        base_individual = individual
        current_height = get_individual_height(individual)
        list_individual = None
        if current_height > max_tree_height:
            indices = get_nodes_with_height(individual, max_tree_height)
            list_individual = create_a_copy_of_individual_as_list(individual)
            number_of_removed_nodes = 0
            for i, index in enumerate(indices):
                altered_index = index - number_of_removed_nodes
                if isinstance(individual[index], deap.gp.Primitive):
                    subtree_slice = individual.searchSubtree(index)
                    start, end = subtree_slice.start, subtree_slice.stop
                    range = end - start - 1
                    number_of_removed_nodes = number_of_removed_nodes + range
                    prune_start = altered_index + 1  # we dont want to remove the node we are changing for a terminal
                    # Cut the individual from start to end by creating a copy of an individual and removing the parts from start to end
                    list_individual = prune_subtree(list_individual, prune_start, prune_start + range)
                    list_individual[altered_index] = random.choice(pset.terminals[object])
        if list_individual is not None:  # needs trimming
            individual = creator.Individual(list_individual)
            csv_export.save_pruned_tree(base_individual)
    except:
        print("Exception in first layer generation loop")
        traceback.print_exc()
    return individual


def create_a_copy_of_individual_as_list(individual):
    list_individual = []
    for elem in individual:
        list_individual.append(elem)
    return list_individual


def prune_subtree(individual, start, end):
    return individual[:start] + individual[end:]


def koza_over_selection(individuals, k, tournsize, population_size):
    individuals = sorted(individuals, key=lambda x: x.fitness, reverse=True)
    fittest_individuals_percentage = (32000 / population_size) / 100
    total_fitness_sum = 0
    for individual in individuals:
        if len(individual.fitness.values) > 0:
            total_fitness_sum += individual.fitness.values[0]

    cumulative_fitness = 0
    top_individuals = []
    for ind in individuals:
        cumulative_fitness += ind.fitness.values[0]
        if cumulative_fitness < fittest_individuals_percentage * total_fitness_sum:
            top_individuals.append(ind)
        else:
            break
    rest_individuals = individuals[len(top_individuals):]

    if random.uniform(0, 1) <= 0.8:
        return selection.selTournament(top_individuals, k, tournsize)
    else:
        return selection.selTournament(rest_individuals, k, tournsize)


def gp_evolution(process_id, new_terminal_list, elites_size, population_size, number_of_generations,
                 crossover_probability, mutation_probability,
                 terminals_from_first_layer, toolbox, csv_exporter, layer_number, algorithm_type, process_address_map,
                 number_of_layers):
    input_combinations = None
    try:
        hall_of_fame = tools.HallOfFame(maxsize=elites_size)
        population = toolbox.population(n=population_size)
        if algorithm_type == "MUX" and layer_number == 1 and number_of_layers == 2:
            input_combinations = generate_all_input_combinations_for_model(process_id, process_address_map)
        if algorithm_type == "MUX" and layer_number == 1 and number_of_layers == 2:  # only do it for first layer of two layer mux
            fitnesses = toolbox.map(toolbox.evaluate, population, [input_combinations] * len(population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
        else:
            fitnesses = toolbox.map(toolbox.evaluate, population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
    except:
        print("Error during initial population generation")
        traceback.print_exc()

    for index in range(number_of_generations):
        try:
            print(str(process_id) + ": Generation " + str(index))

            offspring = toolbox.select(population, population_size)

            # Genetic operations
            offspring = algorithms.varAnd(offspring, toolbox, cxpb=crossover_probability,
                                          mutpb=mutation_probability)  # perform only mutation + crossover
            # Trimming
            for i, individual in enumerate(offspring):
                offspring[i] = toolbox.trim(individual)

            # Need to manually evaluate the offspring
            if algorithm_type == "MUX" and layer_number == 1 and number_of_layers == 2:  # only do it for first layer of two layer mux
                fitnesses = toolbox.map(toolbox.evaluate, offspring, [input_combinations] * len(offspring))
                for ind, fit in zip(offspring, fitnesses):
                    ind.fitness.values = fit
            else:
                fitnesses = toolbox.map(toolbox.evaluate, offspring)
                for ind, fit in zip(offspring, fitnesses):
                    ind.fitness.values = fit

            hall_of_fame.update(population)
            elites = hall_of_fame.items
            population[:] = offspring + elites
            # Save best individual
            best_individual = tools.selBest(population, k=1)[0]
            if process_id == 0:  # Only save the for process_id = 0 to avoid unnecessary delays
                csv_exporter.save_best_individual_for_each_generation(best_individual, index, layer_number)

            # Break condition
            if algorithm_type == "approximation":
                if best_individual.fitness.values[0] == 0:
                    break
            else:
                if layer_number == 1 and number_of_layers == 2 and best_individual.fitness.values[0] == len(
                        input_combinations):
                    break
                elif best_individual.fitness.values[0] == 2048.0:
                    break

        except:
            if layer_number == 1:
                print("Exception in first layer generation loop")
            else:
                print("Exception in second layer generation loop")
            traceback.print_exc()

    if terminals_from_first_layer != 0 and new_terminal_list is not None:
        if terminals_from_first_layer == 1:
            new_terminal_list.append(tools.selBest(population, k=1)[0])
        else:
            new_terminal_list += population

    if len(population) != 0:
        best_current_individual = tools.selBest(population, k=1)[0]
        print(f"Best individual: {best_current_individual}, Fitness: {best_current_individual.fitness.values[0]}")
        return best_current_individual
