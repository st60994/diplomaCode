import traceback

import numpy as np
from datetime import datetime

from csvExport import CsvExporter
from firstLayer import FirstLayer, first_layer_params
from gpInitialization import GpFirstLayerInitializer, X_RANGE, Y_RANGE, NUMBER_OF_RUNS

if __name__ == "__main__":
    try:
        now = datetime.now()
        csvExporter = CsvExporter(now)
        gp_first_layer_initializer = GpFirstLayerInitializer()
        gp_first_layer_initializer.initialize_gp_run()
        X, Y = np.meshgrid(X_RANGE, Y_RANGE)
        grid_points = np.column_stack((X.flatten(), Y.flatten()))
        new_terminals = []
        for run_number in range(NUMBER_OF_RUNS):
            print("Starting run " + str(run_number))
            first_layer_instance = FirstLayer(gp_first_layer_initializer.pset, grid_points, csvExporter)
            best_individual = first_layer_instance.first_layer_evolution(0, [])
            best_fitness = float("inf")
            if run_number == 0:
                csvExporter.export_run_params_to_csv(first_layer_params, {})
            csvExporter.save_best_individual(best_individual, run_number)
    except:
        traceback.print_exc()
