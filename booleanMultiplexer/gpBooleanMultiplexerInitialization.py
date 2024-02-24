from deap import gp

from methodDefinitions import custom_and, custom_or, custom_not, custom_if

NUMBER_OF_RUNS = 1000
MAX_TREE_HEIGHT = 17


class GpFirstLayerMUXInitializer:

    def __init__(self):
        self.pset = None

    def initialize_gp_run(self):
        self.pset = gp.PrimitiveSet("MAIN", 11)  # 11 input variables
        self.__create_terminal_set()
        self.__add_primitive_set()

    def __add_primitive_set(self):
        self.pset.addPrimitive(custom_and, 2)
        self.pset.addPrimitive(custom_or, 2)
        self.pset.addPrimitive(custom_not, 1)
        self.pset.addPrimitive(custom_if, 3)

    def __create_terminal_set(self):
        self.pset.renameArguments(ARG0="A2")
        self.pset.renameArguments(ARG1="A1")
        self.pset.renameArguments(ARG2="A0")
        self.pset.renameArguments(ARG3="D7")
        self.pset.renameArguments(ARG4="D6")
        self.pset.renameArguments(ARG5="D5")
        self.pset.renameArguments(ARG6="D4")
        self.pset.renameArguments(ARG7="D3")
        self.pset.renameArguments(ARG8="D2")
        self.pset.renameArguments(ARG9="D1")
        self.pset.renameArguments(ARG10="D0")


class GpSecondLayerInitializer:
    subsets = {}

    def __init__(self, subsets):
        self.pset = None
        self.subsets = subsets

    def initialize_gp_run(self):
        self.pset = gp.PrimitiveSet("MAIN", 11)  # 11 input variables
        self.__create_terminal_set()
        self.__add_primitive_set()

    def __add_primitive_set(self):
        self.pset.addPrimitive(custom_and, 2)
        self.pset.addPrimitive(custom_or, 2)
        self.pset.addPrimitive(custom_not, 1)
        self.pset.addPrimitive(custom_if, 3)

    def __create_terminal_set(self):
        self.pset.renameArguments(ARG0="A0")
        self.pset.renameArguments(ARG1="A1")
        self.pset.renameArguments(ARG2="A2")
        self.pset.renameArguments(ARG3="D0")
        self.pset.renameArguments(ARG4="D1")
        self.pset.renameArguments(ARG5="D2")
        self.pset.renameArguments(ARG6="D3")
        self.pset.renameArguments(ARG7="D4")
        self.pset.renameArguments(ARG8="D5")
        self.pset.renameArguments(ARG9="D6")
        self.pset.renameArguments(ARG10="D7")
        for individual in self.subsets:
            present = False
            # check if individual is not already present in the terminal list to avoid an exception due to the same
            # terminal name
            for terminal_list in self.pset.terminals.values():
                for terminal in terminal_list:
                    if individual == terminal.name:
                        present = True
                        break
            if not present:
                self.pset.addTerminal(self.subsets[individual], name=str(individual))
