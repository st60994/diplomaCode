from functionApproximation.gpInitialization import GpFirstLayerInitializer
from util import draw_individual

if __name__ == "__main__":
    gp_first_layer_initializer = GpFirstLayerInitializer()
    gp_first_layer_initializer.initialize_gp_run()
    individual = "protected_add(pow2(sqrt(sub(mul(mul(-1.0, -1.0), sub(sub(x, y), y)), sub(protected_add(x, mul(-1.0, y)), y)))), mul(protected_add(y, mul(-1.0, x)), mul(y, x)))"
    draw_individual(individual ,gp_first_layer_initializer.pset)