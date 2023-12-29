# import the main libraries
import csv
import numpy as np
from GA import GA
from scipy.optimize import minimize
from sklearn.model_selection import KFold
import time

# define a function for run a configuration of the genetic algorithm
def run_single_configuration(configuration, instance):
    total_fitness = 0                                       # initalize total fitness as 0
    # repeat the process for the predefined number of runs
    for _ in range(n_runs):
        ga = GA(time_deadline, instance, **configuration)   # genetic algorithm
        start_time = time.time()                            # register starting time
        ga.run()                                            # run the algorithm
        total_fitness += ga.get_best_fitness()              # get the best fitness from the GA
        elapsed_time = time.time() - start_time             # get the elapsed time

        # Log or print any relevant information

    average_fitness = total_fitness / n_runs                # calculate the average fitness on the runs
    return average_fitness, elapsed_time                    # return the average fitness and the elapsed time

# define a grid-search function to test the operators
def grid_search_operators(instances):
    # define the different types of operator
    mutation_operators                  =   ['swap', 'insertion', 'scramble', 'inversion']
    crossover_operators                 =   ['cut_and_crossfill', 'pmx', 'edge_crossover']
    selection_methods                   =   ['fitness_proportional', 'linear_ranking', 'exponential_ranking', 'tournament', 'uniform']
    population_replacement_strategies   =   ['generational', 'steady_state', 'replace_worst_genitor', 'elitism', 'round_robin', 'lambda_mu_replacement']

    # initialize an empty array for store the results
    results = []        
    # calculate the total configurations to test (JUST FOR SEE THE PROGRESS OF THE GRID-SEARCH)
    total_configs = len(mutation_operators) * len(crossover_operators) * len(selection_methods) * len(population_replacement_strategies)
    # set the counter of configurations tested to 0 (JUST FOR SEE THE PROGRESS OF THE GRID-SEARCH)
    current_config = 0

    # iterate all the possibile combinations of configurations
    for mutation_operator in mutation_operators:
        for crossover_operator in crossover_operators:
            for selection_method in selection_methods:
                for population_replacement_strategy in population_replacement_strategies:
                    # set the configuration
                    configuration = {
                        'mutation_operator': mutation_operator,
                        'crossover_operator': crossover_operator,
                        'selection_method': selection_method,
                        'population_replacement_strategy': population_replacement_strategy
                    }
                    
                    total_fitness_across_instances = 0      # initialize the total fitness counter as 0
                    total_elapsed_time = 0                  # initialize the total elapsed time counter as 0

                    # test all the instances
                    for instance in instances:
                        # test the current configuration
                        average_fitness, elapsed_time = run_single_configuration(  
                            configuration, 
                            instance
                        )
                        total_fitness_across_instances += average_fitness   # add the obtained fitness to the total fitness
                        total_elapsed_time += elapsed_time                  # add the elapsed time to the counter (JUST FOR SEE THE PROGRESS OF THE GRID-SEARCH)

                    average_fitness_across_instances = total_fitness_across_instances / len(instances)  # calculate the average fitness
                    average_elapsed_time = total_elapsed_time / len(instances)                          # calculate the average elapsed time (JUST FOR SEE THE PROGRESS OF THE GRID-SEARCH)

                    result_entry = configuration.copy()                                     # copy the configuration just tested                                       
                    result_entry['average_fitness'] = average_fitness_across_instances      # store the average fitness obtained
                    result_entry['average_elapsed_time'] = average_elapsed_time             # store the average elapsed time 
                    results.append(result_entry)                                            # store all the data about the test
                    
                    # JUST FOR SEE THE PROGRESS OF THE GRID-SEARCH

                    # 1. update the progress
                    current_config += 1 
                    progress_percentage = (current_config / total_configs) * 100

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

    # get the best configuration operators obtained so-far
    best_configuration_operators = max(results, key=lambda x: x['average_fitness'])
    print(f"Best Configuration (Operators): {best_configuration_operators}, Average Fitness: {best_configuration_operators['average_fitness']}")

    # save results to CSV
    save_results_to_csv(results, "results_operators.csv")

    # return the best configuraton of operators found
    return best_configuration_operators

# define a grid-search function to test different combinations of hyperparameters
def grid_search_hyperparameters(best_operators_configuration, instances):
    # define the hyperparameters to test
    population_sizes            =   [50, 100, 200]
    crossover_rates             =   [0.8, 0.9, 1.0]
    mutation_rates              =   [0.1, 0.2, 0.3]
    tournament_size             =   [2, 4, 8]
    linear_ranking_alpha        =   [1.5, 2.0, 2.5]
    exponential_ranking_alpha   =   [1.5, 2.0, 2.5]
    lambda_value                =   [5, 10, 20]
    mu_value                    =   [2, 5, 10]

    # create an empty list to store the results
    results = []
    # calculate the total configurations to test (JUST FOR SEE THE PROGRESS OF THE GRID-SEARCH)
    total_configs = len(population_sizes) * len(crossover_rates) * len(mutation_rates) * len(tournament_size) * len(linear_ranking_alpha) * len(exponential_ranking_alpha) * len(lambda_value) * len(mu_value)
    # set the counter of configurations tested to 0 (JUST FOR SEE THE PROGRESS OF THE GRID-SEARCH)
    current_config = 0

    # iterate all the possibile combinations of hyperparameters
    for pop_size in population_sizes:
        for cross_rate in crossover_rates:
            for mut_rate in mutation_rates:
                for tour_size in tournament_size:
                    for alpha_linear in linear_ranking_alpha:
                        for alpha_exponential in exponential_ranking_alpha:
                            for lambda_val in lambda_value:
                                for mu_val in mu_value:
                                    # get the best configuration of operators found
                                    configuration = best_operators_configuration.copy()
                                    # update the configuration with the hyperparameters
                                    configuration.update({
                                        'population_size': pop_size,
                                        'crossover_rate': cross_rate,
                                        'mutation_rate': mut_rate,
                                        'tournament_size': tour_size,
                                        'linear_ranking_alpha': alpha_linear,
                                        'exponential_ranking_alpha': alpha_exponential,
                                        'lambda_value': lambda_val,
                                        'mu_value': mu_val
                                    })
                                
                                    total_fitness_across_instances = 0  # initialize the total fitness counter as 0
                                    total_elapsed_time = 0              # initialize the total elapsed time counter as 0
                                    
                                    # test all the instances
                                    for instance in instances:
                                        # test the current configuration
                                        average_fitness, elapsed_time = run_single_configuration(
                                            configuration, 
                                            instance
                                        )
                                        total_fitness_across_instances += average_fitness   # add the obtained fitness to the total fitness
                                        total_elapsed_time += elapsed_time                  # add the elapsed time to the counter (JUST FOR SEE THE PROGRESS OF THE GRID-SEARCH)

                                    average_fitness_across_instances = total_fitness_across_instances / len(instances)      # calculate the average fitness
                                    average_elapsed_time = total_elapsed_time / len(instances)                              # calculate the average elapsed time (JUST FOR SEE THE PROGRESS OF THE GRID-SEARCH)
                                    
                                    result_entry = configuration.copy()                                     # copy the configuration just tested                                       
                                    result_entry['average_fitness'] = average_fitness_across_instances      # store the average fitness obtained
                                    result_entry['average_elapsed_time'] = average_elapsed_time             # store the average elapsed time 
                                    results.append(result_entry)                                            # store all the data about the test

                                    # JUST FOR SEE THE PROGRESS OF THE GRID-SEARCH

                                    # 1. Update progress
                                    current_config += 1
                                    progress_percentage = (current_config / total_configs) * 100

                                    # display configuration being tested
                                    current_configuration = f"Config {current_config}/{total_configs} - " \
                                                            f"Population Size: {pop_size}, " \
                                                            f"Crossover Rate: {cross_rate}, " \
                                                            f"Mutation Rate: {mut_rate}, " \
                                                            f"Tournament Size: {tour_size}, " \
                                                            f"Linear Ranking Alpha: {alpha_linear}, " \
                                                            f"Exponential Ranking Alpha: {alpha_exponential}, " \
                                                            f"Lambda: {lambda_val}, " \
                                                            f"Mu: {mu_val}"

                                    print(f"Progress: {progress_percentage:.2f}% - {current_configuration}")

                                    # 2. Estimate time left
                                    remaining_configs = total_configs - current_config
                                    estimated_seconds_left = remaining_configs * average_elapsed_time
                                    estimated_hours_left, remainder = divmod(estimated_seconds_left, 3600)
                                    estimated_minutes_left, estimated_seconds_left = divmod(remainder, 60)
                                    print(f"Estimated time left: {int(estimated_hours_left)}h {int(estimated_minutes_left)}m {int(estimated_seconds_left)}s")

    # get the best configuration operators obtained so-far
    best_configuration_hyperparameters = max(results, key=lambda x: x['average_fitness'])
    print(f"Best Configuration (Hyperparameters): {best_configuration_hyperparameters}, Average Fitness: {best_configuration_hyperparameters['average_fitness']}")

    # save results to CSV
    save_results_to_csv(results, "results_hyperparameters.csv")

def save_results_to_csv(results, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in results:
            writer.writerow(row)

if __name__ == "__main__":
    instances = ["instance1.txt", "instance2.txt", "instance3.txt", "instance4.txt", "instance5.txt", "instance6.txt", "instance7.txt", "instance8.txt"]
    time_deadline = 180
    n_runs = 3  # Number of runs for each configuration

    # Perform grid search on operators
    best_operators_configuration = grid_search_operators(instances)

    # Perform grid search on hyperparameters with the best operators configuration
    grid_search_hyperparameters(best_operators_configuration, instances)