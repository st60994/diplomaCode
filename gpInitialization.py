import operator

from deap import gp

from methodDefinitions import protectedAdd, sqrt, pow2, pow3


class GpInitializer:

    def __init__(self):
        self.pset = None

    def initialize_gp_run(self):
        self.pset = gp.PrimitiveSet("MAIN", 1)  # 1 input variable
        self.__create_terminal_set()
        self.__add_primitive_set()

    def __add_primitive_set(self):
        self.pset.addPrimitive(protectedAdd, 2)
        self.pset.addPrimitive(operator.sub, 2)
        self.pset.addPrimitive(operator.mul, 2)
        self.pset.addPrimitive(sqrt, 1)
        # pset.addPrimitive(sin, 1)
        self.pset.addPrimitive(pow2, 1)
        self.pset.addPrimitive(pow3, 1)

    def __create_terminal_set(self):
        self.pset.addTerminal(-1.0)
        self.pset.addTerminal(1.0)
        self.pset.addTerminal(2.0)
        self.pset.addTerminal(3.0)
        self.pset.renameArguments(ARG0="x")
