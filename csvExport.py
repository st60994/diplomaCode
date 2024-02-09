import os
import traceback
from datetime import datetime
import csv
from pathlib import Path


class CsvExporter:

    def __init__(self, first_layer_params, second_layer_params, best_individual):
        self.first_layer_params = first_layer_params
        self.second_layer_params = second_layer_params
        self.best_individual = best_individual

    def export_run_data_to_csv(self):
        now = datetime.now()
        project_path = Path(__file__).parent
        folder_name_date = now.strftime("%Y_%m_%d")
        folder_name_time = now.strftime("%H_%M_%S")
        folder_name = project_path / 'data' / folder_name_date / folder_name_time
        print("Folder name {}", folder_name)
        self.__create_a_folder(folder_name)
        self.__save_first_layer_params_to_csv(folder_name)
        self.__save_second_layer_params_to_csv(folder_name)
        self.__save_best_individual(folder_name)

    def __save_first_layer_params_to_csv(self, folder_name):
        file_path = os.path.join(folder_name, 'first_layer_params.csv')
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(['Parameter', 'Value'])
            for key, value in self.first_layer_params.items():
                writer.writerow([key, value])

    def __save_second_layer_params_to_csv(self, folder_name):
        file_path = os.path.join(folder_name, 'second_layer_params.csv')
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(['Parameter', 'Value'])
            for key, value in self.second_layer_params.items():
                writer.writerow([key, value])

    def __save_best_individual(self, folder_name):
        file_path = os.path.join(folder_name, 'best_individual.csv')
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(['Individual', 'Fitness'])
            writer.writerow([self.best_individual, self.best_individual.fitness.values[0]])

    @staticmethod
    def __create_a_folder(folder_name):
        try:
            Path(folder_name).mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            traceback.print_exc()
