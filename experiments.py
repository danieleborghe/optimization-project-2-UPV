# import the main libraries
import csv
from GA import GA
import time
import os

# CONSTANTS
N_RUNS = 3                              # number of runs for each configuration
TIME_DEADLINE = 180                     # time deadline for each run in seconds
INSTANCES_DIRECTORY = "instances"       # instances directory name

# Define the different types of operators
MUTATION_OPERATORS = (
    #'swap', 
    #'insertion', 
    #'scramble', 
    #'inversion',
    "mixed"
)
CROSSOVER_OPERATORS = (
    'cut_and_crossfill', 
    'pmx', 
    'edge',
)
SELECTION_METHODS = (
    'roulette_wheel', 
    'linear_ranking', 
    'exponential_ranking', 
    'tournament', 
    'uniform'
)
POPULATION_REPLACEMENT_STRATEGIES = (
    'generational', 
    'replace_worst_genitor', 
    'round_robin', 
    'lambda_mu'
)

POPULATION_SIZES = (25, 50, 100, 200, 400)
CROSSOVER_RATES = (0.60, 0.70, 0.80, 0.90, 1.00)
MUTATION_RATES = (0.10, 0.20, 0.30, 0.40, 0.50)
EARLY_STOPPING_LIMIT = (25, 50, 100)

ROUND_ROBIN_MATCHES = (0.1, 0.2, 0.3, 0.4, 0.5)
ELITE_PERCENTAGE = (0.1, 0.2, 0.3, 0.4, 0.5)
TOURNAMENT_SIZE = (0.1, 0.2, 0.3, 0.4, 0.5)
TORUNAMENT_PROBABILITY = (0.1, 0.3, 0.5, 0.7, 0.9)
EXPONENTIAL_RANKING_C = (0.1, 0.3, 0.5, 0.7, 0.9)
LINEAR_RANKING_S = (1.1, 1.3, 1.5, 1.7, 1.9)

# define a function for run a configuration of the genetic algorithm
def run_single_configuration(configuration, instance):
    total_fitness = 0                                       # initalize total fitness as 0
    elapsed_times = []
    # repeat the process for the predefined number of runs
    for n_run in range(N_RUNS):
        ga = GA(TIME_DEADLINE, instance, random_seed = n_run, **configuration)   # genetic algorithm
        ga.run()                                            # run the algorithm
        total_fitness += ga.get_best_fitness()              # get the best fitness from the GA
        elapsed_time = ga.get_elapsed_time()                # get the elapsed time
        elapsed_times.append(elapsed_time)

    average_fitness = total_fitness / N_RUNS                # calculate the average fitness on the runs
    total_elapsed_time = sum(elapsed_times)
    average_elapsed_time = total_elapsed_time / N_RUNS

    return average_fitness, average_elapsed_time, total_elapsed_time    # return the average fitness and the elapsed time

def grid_search_operators_new(instances):
    # calculate the total configurations to test (JUST FOR SEE THE PROGRESS OF THE GRID-SEARCH)
    total_configs = len(MUTATION_OPERATORS) * len(CROSSOVER_OPERATORS) * len(SELECTION_METHODS) * len(POPULATION_REPLACEMENT_STRATEGIES)
    # set the counter of configurations tested to 0 (JUST FOR SEE THE PROGRESS OF THE GRID-SEARCH)
    current_config = 0
    total_elapsed_time = 0

    # initialize a list to store results for each combination of instances and configurations
    all_results = []

    # iterate all the possible combinations of configurations
    for mutation_operator in MUTATION_OPERATORS:
        for crossover_operator in CROSSOVER_OPERATORS:
            for selection_method in SELECTION_METHODS:
                for population_replacement_strategy in POPULATION_REPLACEMENT_STRATEGIES:
                    result_entry = {}
                    # set the configuration
                    configuration = {
                        'mutation_operator': mutation_operator,
                        'crossover_operator': crossover_operator,
                        'selection_method': selection_method,
                        'population_replacement_strategy': population_replacement_strategy
                    }

                    # initialize lists to store results for each instance
                    fitness_across_instances = []
                    run_elapsed_time_across_instances = []
                    total_config_elapsed_time = 0            # initialize the total elapsed time counter as 0

                    # test all the instances
                    for instance_name, instance in instances.items():
                        #print("a")
                        # test the current configuration
                        average_fitness, average_run_elapsed_time, elapsed_time = run_single_configuration(
                            configuration,
                            instance
                        )

                        result_entry[f'fitness_{instance_name}'] = average_fitness
                        result_entry[f'time_{instance_name}'] = average_run_elapsed_time

                        fitness_across_instances.append(average_fitness)
                        run_elapsed_time_across_instances.append(average_run_elapsed_time)

                        total_config_elapsed_time += elapsed_time

                    # store results for each combination of instances and configurations
                    result_entry['configuration'] = configuration
                    result_entry['average_fitness_across_instances'] = sum(fitness_across_instances) / len(
                        instances)
                    result_entry['average_elapsed_time_across_instances'] = sum(run_elapsed_time_across_instances) / len(
                        instances)
                    all_results.append(result_entry)

                    # JUST FOR SEE THE PROGRESS OF THE GRID-SEARCH

                    # 1. update the progress
                    current_config += 1 
                    progress_percentage = (current_config / total_configs) * 100
                    total_elapsed_time += total_config_elapsed_time
                    #print(total_elapsed_time)
                    average_elapsed_time = total_elapsed_time / (current_config + 1)
                    #print(total_elapsed_time)

                    # Display configuration being tested
                    current_configuration = f"Config {current_config}/{total_configs} - " \
                                            f"Mutation: {mutation_operator}, " \
                                            f"Crossover: {crossover_operator}, " \
                                            f"Selection: {selection_method}, " \
                                            f"Replacement: {population_replacement_strategy}"

                    print(f"Progress: {progress_percentage:.2f}% - {current_configuration}")

                    # 2. estimate time left
                    remaining_configs = total_configs - current_config
                    estimated_seconds_left = remaining_configs * average_elapsed_time
                    estimated_hours_left, remainder = divmod(estimated_seconds_left, 3600)
                    estimated_minutes_left, estimated_seconds_left = divmod(remainder, 60)
                    print(f"Estimated time left: {int(estimated_hours_left)}h {int(estimated_minutes_left)}m {int(estimated_seconds_left)}s")

    # calculate the ranking of each configuration across all instances
    for instance_name in instances:
        all_results.sort(key=lambda x: (
            x[f'fitness_{instance_name}'], x[f'time_{instance_name}']))  # sort by fitness and time in case of ties
        for i, result in enumerate(all_results):
            result[f'rank_{instance_name}'] = i + 1

    # calculate the average rank for each configuration across all instances
    for result in all_results:
        total_rank = sum(result[f'rank_{instance_name}'] for instance_name in instances)
        result['average_rank'] = total_rank / len(instances)

    # get the best configuration operators obtained so far
    best_configuration_operators = min(all_results, key=lambda x: x['average_rank'])
    print(
        f"Best Configuration (Operators): {best_configuration_operators}, Average Rank: {best_configuration_operators['average_rank']}, Average Fitness: {best_configuration_operators['average_fitness_across_instances']}")

    # save results to CSV
    save_results_to_csv(all_results, "results_operators.csv")
    
def grid_search_hyperparameters_new(instances):
    # calculate the total configurations to test (JUST FOR SEE THE PROGRESS OF THE GRID-SEARCH)
    total_configs = len(POPULATION_SIZES) * len(CROSSOVER_RATES) * len(MUTATION_RATES) * len(EARLY_STOPPING_LIMIT)
    # set the counter of configurations tested to 0 (JUST FOR SEE THE PROGRESS OF THE GRID-SEARCH)
    current_config = 0
    total_elapsed_time = 0

    # initialize a list to store results for each combination of instances and configurations
    all_results = []

    # iterate all the possible combinations of configurations
    for pop_size in POPULATION_SIZES:
        for cross_rate in CROSSOVER_RATES:
            for mut_rate in MUTATION_RATES:
                for es_limit in EARLY_STOPPING_LIMIT:
                    result_entry = {}
                    # update the configuration with the hyperparameters
                    configuration = {
                        'population_size': pop_size,
                        'cross_rate': cross_rate,
                        'mut_rate': mut_rate,
                        'early_stopping_limit': es_limit
                    }
                                       
                    # initialize lists to store results for each instance
                    fitness_across_instances = []
                    run_elapsed_time_across_instances = []
                    total_config_elapsed_time = 0            # initialize the total elapsed time counter as 0

                    # test all the instances
                    for instance_name, instance in instances.items():
                        #print("a")
                        # test the current configuration
                        average_fitness, average_run_elapsed_time, elapsed_time = run_single_configuration(
                            configuration,
                            instance
                        )

                        result_entry[f'fitness_{instance_name}'] = average_fitness
                        result_entry[f'time_{instance_name}'] = average_run_elapsed_time

                        fitness_across_instances.append(average_fitness)
                        run_elapsed_time_across_instances.append(average_run_elapsed_time)

                        total_config_elapsed_time += elapsed_time

                    # store results for each combination of instances and configurations
                    result_entry['configuration'] = configuration
                    result_entry['average_fitness_across_instances'] = sum(fitness_across_instances) / len(
                        instances)
                    result_entry['average_elapsed_time_across_instances'] = sum(run_elapsed_time_across_instances) / len(
                        instances)
                    all_results.append(result_entry)

                    # JUST FOR SEE THE PROGRESS OF THE GRID-SEARCH

                    # 1. update the progress
                    current_config += 1 
                    progress_percentage = (current_config / total_configs) * 100
                    total_elapsed_time += total_config_elapsed_time
                    #print(total_elapsed_time)
                    average_elapsed_time = total_elapsed_time / (current_config + 1)
                    #print(total_elapsed_time)

                    # display configuration being tested
                    current_configuration = f"Config {current_config}/{total_configs} - " \
                                            f"Population Size: {pop_size}, " \
                                            f"Crossover Rate: {cross_rate}, " \
                                            f"Mutation Rate: {mut_rate}, " \
                                            f"Early stopping limit: {es_limit}"
                    
                    print(f"Progress: {progress_percentage:.2f}% - {current_configuration}")

                    # 2. estimate time left
                    remaining_configs = total_configs - current_config
                    estimated_seconds_left = remaining_configs * average_elapsed_time
                    estimated_hours_left, remainder = divmod(estimated_seconds_left, 3600)
                    estimated_minutes_left, estimated_seconds_left = divmod(remainder, 60)
                    print(f"Estimated time left: {int(estimated_hours_left)}h {int(estimated_minutes_left)}m {int(estimated_seconds_left)}s")

    # calculate the ranking of each configuration across all instances
    for instance_name in instances:
        all_results.sort(key=lambda x: (
            x[f'fitness_{instance_name}'], x[f'time_{instance_name}']))  # sort by fitness and time in case of ties
        for i, result in enumerate(all_results):
            result[f'rank_{instance_name}'] = i + 1

    # calculate the average rank for each configuration across all instances
    for result in all_results:
        total_rank = sum(result[f'rank_{instance_name}'] for instance_name in instances)
        result['average_rank'] = total_rank / len(instances)

    # get the best configuration operators obtained so far
    best_configuration_hyperparameters = min(all_results, key=lambda x: x['average_rank'])
    print(
        f"Best Configuration (Hyperparameters): {best_configuration_hyperparameters}, Average Rank: {best_configuration_hyperparameters['average_rank']}, Average Fitness: {best_configuration_hyperparameters['average_fitness_across_instances']}")

    # save results to CSV
    save_results_to_csv(all_results, "results_hyperparameters.csv")

def save_results_to_csv(results, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in results:
            writer.writerow(row)

def main():
    instances = {}

    for i in range(1, 9):
        instances[f"instance_{i}"] = os.path.join(INSTANCES_DIRECTORY, f"instance0{i}.txt")

    # Choose which grid search to run
    choice = input("Enter 1 for Operators Grid Search, 2 for Hyperparameters Grid Search: ")

    if choice == '1':
        grid_search_operators_new(instances)
        print("Operators Grid Search completed.")
    elif choice == '2':
        grid_search_hyperparameters_new(instances)
        print("Hyperparameters Grid Search completed.")
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()