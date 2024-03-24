import operator

import numpy as np
from deap import gp

from methodDefinitions import protected_add, sqrt, pow2, pow3

LOWER_BOUND_X = -2.0
UPPER_BOUND_X = 2
LOWER_BOUND_Y = -2.0
UPPER_BOUND_Y = 2
STEP_SIZE_X = 0.1
STEP_SIZE_Y = 0.1
X_RANGE = np.arange(LOWER_BOUND_X, UPPER_BOUND_X + STEP_SIZE_X, STEP_SIZE_X)
Y_RANGE = np.arange(LOWER_BOUND_Y, UPPER_BOUND_Y + STEP_SIZE_Y, STEP_SIZE_Y)
NUMBER_OF_RUNS = 500
MAX_TREE_HEIGHT = 17


def target_polynomial(x, y):
    return (x * y) * (y - x)


class GpFirstLayerInitializer:

    def __init__(self):
        self.pset = None

    def initialize_gp_run(self):
        self.pset = gp.PrimitiveSet("MAIN", 2)  # 2 input variables
        self.__create_terminal_set()
        self.__add_primitive_set()

    def __add_primitive_set(self):
        self.pset.addPrimitive(protected_add, 2)
        self.pset.addPrimitive(operator.sub, 2)
        self.pset.addPrimitive(operator.mul, 2)
        # self.pset.addPrimitive(sqrt, 1)
        # self.pset.addPrimitive(pow2, 1)
        # self.pset.addPrimitive(pow3, 1)

    def __create_terminal_set(self):
        self.pset.addTerminal(-1.0)
        self.pset.addTerminal(1.0)
        self.pset.addTerminal(2.0)
        self.pset.addTerminal(3.0)
        self.pset.renameArguments(ARG0="x")
        self.pset.renameArguments(ARG1="y")


class GpSecondLayerInitializer:
    subsets = {}

    def __init__(self, subsets):
        self.pset = None
        self.subsets = subsets

    def initialize_gp_run(self):
        self.pset = gp.PrimitiveSet("MAIN", 2)  # 2 input variables
        self.__create_terminal_set()
        self.__add_primitive_set()

    def __add_primitive_set(self):
        self.pset.addPrimitive(operator.mul, 2)
        self.pset.addPrimitive(protected_add, 2)
        self.pset.addPrimitive(operator.sub, 2)
        self.pset.addPrimitive(sqrt, 1)
        # self.pset.addPrimitive(sin, 1)
        self.pset.addPrimitive(pow2, 1)
        self.pset.addPrimitive(pow3, 1)
        #   self.pset.addPrimitive(avg, 1)

    def __create_terminal_set(self):
        self.pset.addTerminal(-1.0)
        self.pset.addTerminal(1.0)
        self.pset.addTerminal(2.0)
        self.pset.addTerminal(3.0)
        for individual in self.subsets:
            present = False
            # check if individual is not already present in the terminal list to avoid an exception due to the same terminal name
            for terminal_list in self.pset.terminals.values():
                for terminal in terminal_list:
                    if individual == terminal.name:
                        present = True
                        break
            if not present:
                self.pset.addTerminal(self.subsets[individual], name=str(individual))
        print("Primitive Set Terminals:", self.pset.terminals)
        self.pset.renameArguments(ARG0="x")
        self.pset.renameArguments(ARG1="y")
