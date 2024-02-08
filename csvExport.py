import os
from datetime import datetime
import csv


class CsvExporter:

    def __init__(self, first_layer_params, second_layer_params, best_individual):
        self.first_layer_params = first_layer_params
        self.second_layer_params = second_layer_params
        self.best_individual = best_individual

    def export_run_data_to_csv(self):
        now = datetime.now()
        folder_name = now.strftime("%Y_%m_%d__%H_%M_%S")
        self.__create_a_folder(folder_name)
        self.__save_first_layer_params_to_csv(folder_name)
        self.__save_second_layer_params_to_csv(folder_name)
        self.__save_best_individual(folder_name)

    def __save_first_layer_params_to_csv(self, folder_name):
        file_path = os.path.join('data', folder_name, 'first_layer_params.csv')
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(['Parameter', 'Value'])
            for key, value in self.first_layer_params.items():
                writer.writerow([key, value])

    def __save_second_layer_params_to_csv(self, folder_name):
        file_path = os.path.join('data', folder_name, 'second_layer_params.csv')
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(['Parameter', 'Value'])
            for key, value in self.second_layer_params.items():
                writer.writerow([key, value])

    def __save_best_individual(self, folder_name):
        file_path = os.path.join('data', folder_name, 'best_individual.csv')
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(['Individual', 'Fitness'])
            writer.writerow([self.best_individual, self.best_individual.fitness.values[0]])

    def __create_a_folder(self, folder_name):
        project_path = os.path.dirname(__file__)
        folder_path = os.path.join(project_path, 'data', folder_name)
        os.makedirs(folder_path)
