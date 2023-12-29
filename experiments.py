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
    'swap', 
    'insertion', 
    'scramble', 
    'inversion'
)
CROSSOVER_OPERATORS = (
    'cut_and_crossfill', 
    'pmx', 
    'edge_crossover'
)
SELECTION_METHODS = (
    'fitness_proportional', 
    'linear_ranking', 
    'exponential_ranking', 
    'tournament', 
    'uniform'
)
POPULATION_REPLACEMENT_STRATEGIES = (
    'generational', 
    'steady_state', 
    'replace_worst_genitor', 
    'elitism', 
    'round_robin', 
    'lambda_mu_replacement'
)

# define the hyperparameters to test 
# to be changed based on the result of the operators grid-search)
POPULATION_SIZES = (50, 100, 200)
CROSSOVER_RATES = (0.8, 0.9, 1.0)
MUTATION_RATES = (0.1, 0.2, 0.3)
TOURNAMENT_SIZE = (2, 4, 8)
LINEAR_RANKING_ALPHA = (1.5, 2.0, 2.5)
EXPONENTIAL_RANKING_ALPHA = (1.5, 2.0, 2.5)
LAMBDA_VALUE = (5, 10, 20)
MU_VALUE = (2, 5, 10)

# define a function for run a configuration of the genetic algorithm
def run_single_configuration(configuration, instance):
    total_fitness = 0                                       # initalize total fitness as 0
    elapsed_times = []
    # repeat the process for the predefined number of runs
    for _ in range(N_RUNS):
        ga = GA(TIME_DEADLINE, instance, **configuration)   # genetic algorithm
        start_time = time.time()                            # register starting time
        ga.run()                                            # run the algorithm
        total_fitness += ga.get_best_fitness()             # get the best fitness from the GA
        elapsed_time = time.time() - start_time             # get the elapsed time
        elapsed_times.append(elapsed_time)

    average_fitness = total_fitness / N_RUNS                # calculate the average fitness on the runs
    total_elapsed_time = sum(elapsed_times)
    average_elapsed_time = total_elapsed_time / N_RUNS

    return average_fitness, average_elapsed_time, total_elapsed_time    # return the average fitness and the elapsed time

# define a grid-search function to test the operators
def grid_search_operators(instances):
    # initialize an empty array for store the results
    results = []        
    # calculate the total configurations to test (JUST FOR SEE THE PROGRESS OF THE GRID-SEARCH)
    total_configs = len(MUTATION_OPERATORS) * len(CROSSOVER_OPERATORS) * len(SELECTION_METHODS) * len(POPULATION_REPLACEMENT_STRATEGIES)
    # set the counter of configurations tested to 0 (JUST FOR SEE THE PROGRESS OF THE GRID-SEARCH)
    current_config = 0

    # iterate all the possibile combinations of configurations
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

                    ranks_across_instances = []
                    total_fitness_across_instances = 0      # initialize the total fitness counter as 0
                    total_elapsed_time = 0                  # initialize the total elapsed time counter as 0

                    # test all the instances
                    for instance in instances:
                        # test the current configuration
                        average_fitness, average_elapsed_time, elapsed_time = run_single_configuration(  
                            configuration, 
                            instances[instance]
                        )

                        result_entry['fitness_' + str(instance)] = average_fitness
                        result_entry['time_' + str(instance)] = average_elapsed_time

                        rank = (average_fitness, average_elapsed_time)
                        ranks_across_instances.append(rank)

                        total_fitness_across_instances += average_fitness   # add the obtained fitness to the total fitness
                        total_elapsed_time += elapsed_time                  # add the elapsed time to the counter (JUST FOR SEE THE PROGRESS OF THE GRID-SEARCH)

                    # Sort by fitness and time in case of ties
                    sorted_ranks = sorted(ranks_across_instances, key=lambda x: (x[0], x[1]))

                    # Assign ranks
                    ranks_assigned = [(rank[0], rank[1], i + 1) for i, rank in enumerate(sorted_ranks)]

                    # Calculate average rank
                    average_rank = sum(rank[2] for rank in ranks_assigned) / len(ranks_assigned)

                    average_fitness_across_instances = total_fitness_across_instances / len(instances)  # calculate the average fitness
                    average_elapsed_time = total_elapsed_time / len(instances)                          # calculate the average elapsed time (JUST FOR SEE THE PROGRESS OF THE GRID-SEARCH)

                    result_entry = configuration.copy()                                     # copy the configuration just tested
                    result_entry['average_rank'] = average_rank                                       
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
    best_configuration_operators = min(results, key=lambda x: x['average_rank'])
    print(f"Best Configuration (Operators): {best_configuration_operators}, Average Rank: {best_configuration_operators['average_rank']}, Average Fitness: {best_configuration_operators['average_fitness']}")

    # save results to CSV
    save_results_to_csv(results, "results_operators.csv")

# define a grid-search function to test different combinations of hyperparameters
def grid_search_hyperparameters(instances):
    # create an empty list to store the results
    results = []
    # calculate the total configurations to test (JUST FOR SEE THE PROGRESS OF THE GRID-SEARCH)
    total_configs = len(POPULATION_SIZES) * len(CROSSOVER_RATES) * len(MUTATION_RATES) * len(TOURNAMENT_SIZE) * len(LINEAR_RANKING_ALPHA) * len(EXPONENTIAL_RANKING_ALPHA) * len(LAMBDA_VALUE) * len(MU_VALUE)
    # set the counter of configurations tested to 0 (JUST FOR SEE THE PROGRESS OF THE GRID-SEARCH)
    current_config = 0

    # iterate all the possibile combinations of hyperparameters
    for pop_size in POPULATION_SIZES:
        for cross_rate in CROSSOVER_RATES:
            for mut_rate in MUTATION_RATES:
                for tour_size in TOURNAMENT_SIZE:
                    for alpha_linear in LINEAR_RANKING_ALPHA:
                        for alpha_exponential in EXPONENTIAL_RANKING_ALPHA:
                            for lambda_val in LAMBDA_VALUE:
                                for mu_val in MU_VALUE:
                                    result_entry = {}
                                    # update the configuration with the hyperparameters
                                    configuration = {
                                        'population_size': pop_size,
                                        'crossover_rate': cross_rate,
                                        'mutation_rate': mut_rate,
                                        'tournament_size': tour_size,
                                        'linear_ranking_alpha': alpha_linear,
                                        'exponential_ranking_alpha': alpha_exponential,
                                        'lambda_value': lambda_val,
                                        'mu_value': mu_val
                                    }
                                
                                    ranks_across_instances = []
                                    total_fitness_across_instances = 0      # initialize the total fitness counter as 0
                                    total_elapsed_time = 0                  # initialize the total elapsed time counter as 0

                                    # test all the instances
                                    for instance in instances:
                                        # test the current configuration
                                        average_fitness, average_elapsed_time, elapsed_time = run_single_configuration(  
                                            configuration, 
                                            instances[instance]
                                        )

                                        result_entry['fitness_' + str(instance)] = average_fitness
                                        result_entry['time_' + str(instance)] = average_elapsed_time

                                        rank = (average_fitness, average_elapsed_time)
                                        ranks_across_instances.append(rank)

                                        total_fitness_across_instances += average_fitness   # add the obtained fitness to the total fitness
                                        total_elapsed_time += elapsed_time                  # add the elapsed time to the counter (JUST FOR SEE THE PROGRESS OF THE GRID-SEARCH)

                                    # Sort by fitness and time in case of ties
                                    sorted_ranks = sorted(ranks_across_instances, key=lambda x: (x[0], x[1]))

                                    # Assign ranks
                                    ranks_assigned = [(rank[0], rank[1], i + 1) for i, rank in enumerate(sorted_ranks)]

                                    # Calculate average rank
                                    average_rank = sum(rank[2] for rank in ranks_assigned) / len(ranks_assigned)

                                    average_fitness_across_instances = total_fitness_across_instances / len(instances)  # calculate the average fitness
                                    average_elapsed_time = total_elapsed_time / len(instances)                          # calculate the average elapsed time (JUST FOR SEE THE PROGRESS OF THE GRID-SEARCH)

                                    result_entry = configuration.copy()                                     # copy the configuration just tested
                                    result_entry['average_rank'] = average_rank                                       
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
    best_configuration_hyperparameters = min(results, key=lambda x: x['average_rank'])
    print(f"Best Configuration (Hyperparameters): {best_configuration_hyperparameters}, Average Fitness: {best_configuration_hyperparameters['average_fitness']}, Average Rank: {best_configuration_hyperparameters['average_rank']}")

    # save results to CSV
    save_results_to_csv(results, "results_hyperparameters.csv")

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
        grid_search_operators(instances)
        print("Operators Grid Search completed.")
    elif choice == '2':
        grid_search_hyperparameters(instances)
        print("Hyperparameters Grid Search completed.")
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()