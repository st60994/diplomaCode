import traceback
import csv
from pathlib import Path


class CsvExporter:

    def __init__(self, now):
        self.folder_name = self.__get_folder_name(now)
        self.__create_a_folder()

    @staticmethod
    def __get_folder_name(now):
        project_path = Path(__file__).parent
        folder_name_date = now.strftime("%Y_%m_%d")
        folder_name_time = now.strftime("%H_%M_%S")
        return project_path / 'data' / folder_name_date / folder_name_time

    def export_run_params_to_csv(self, first_layer_params, second_layer_params):
        self.__save_first_layer_params_to_csv(first_layer_params)
        self.__save_second_layer_params_to_csv(second_layer_params)

    def save_best_individual(self, best_individual, run_number):
        file_path = self.folder_name / 'best_individual.csv'
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            if run_number == 0:
                writer.writerow(['Run number', 'Individual', 'Fitness'])
            writer.writerow([run_number, best_individual, best_individual.fitness.values[0]])

    def save_best_individual_for_each_generation(self, best_individual, generation_number, layer_number):
        folder_path = self.folder_name / str(layer_number)
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        file_path = folder_path / 'best_individual_generation.csv'
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            if generation_number == 0:
                writer.writerow(['Generation Number', 'Individual', 'Fitness'])
            writer.writerow([generation_number, best_individual, best_individual.fitness.values[0]])

    def save_whole_population_for_each_generation(self, population, generation_number, run_number):
        file_name = 'population' + str(generation_number) + '.csv'
        folder_path = self.folder_name / str(run_number)
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        file_path = folder_path / file_name
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(['Individual', 'Fitness'])
            for individual in population:
                writer.writerow([individual, individual.fitness.values[0]])

    def save_sub_models(self, sub_models, run_number):
        folder_path = self.folder_name / 'sub_models'
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        file_path = folder_path / f'sub_models_{run_number}.csv'
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(['Individual', 'Fitness'])
            for sub_model in sub_models:
                writer.writerow([sub_model, sub_model.fitness.values[0]])

    def __save_first_layer_params_to_csv(self, first_layer_params):
        file_path = self.folder_name / 'first_layer_params.csv'
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(['Parameter', 'Value'])
            for key, value in first_layer_params.items():
                writer.writerow([key, value])

    def __save_second_layer_params_to_csv(self, second_layer_params):
        file_path = self.folder_name / 'second_layer_params.csv'
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(['Parameter', 'Value'])
            for key, value in second_layer_params.items():
                writer.writerow([key, value])

    def save_number_of_approximations(self, number_of_approximations, run_number):
        file_path = self.folder_name / 'number_of_approximations.csv'
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            if run_number == 0:
                writer.writerow(['Run number', 'Number of approximations'])
            writer.writerow([run_number, number_of_approximations])

    def save_badly_pruned_tree(self, pruned_tree):
        file_path = self.folder_name / 'pruned_trees.csv'
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow([str(pruned_tree)])

    def __create_a_folder(self):
        try:
            Path(self.folder_name).mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            traceback.print_exc()
