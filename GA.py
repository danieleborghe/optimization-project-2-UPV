import random
import time
import numpy as np
class GA:

    ########################################
    ####### SELECTION OF POPULATION ########
    ########################################
    
    # ROULETTE WHEEL SELECTION METHOD
    def roulette_wheel_selection(self):
        # Calculate fitness values for each individual in the population
        fitness_values = np.array([self.fitness(individual) for individual in self.population])
        
        # Calculate total fitness of the population
        total_fitness = np.sum(fitness_values)
        
        # Calculate selection probabilities based on fitness values
        selection_probs = fitness_values / total_fitness

        # Calculate cumulative probabilities
        cumulative_probs = np.cumsum(selection_probs)
        
        # Generate random values for selection
        rand_vals = np.random.rand(self.population_size)

        selected_parents = []

        # Select individuals based on roulette wheel method
        for rand_val in rand_vals:
            index = np.searchsorted(cumulative_probs, rand_val, side='right')
            selected_parents.append(self.population[index])

        # Ensure the number of selected parents is even
        if len(selected_parents) % 2 != 0:
            selected_parents.pop()

        return selected_parents


    # LINEAR RANKING SELECTION METHOD
    def linear_ranking_selection(self):
        # Sort indices based on fitness values in descending order
        sorted_indices = np.argsort([self.fitness(individual) for individual in self.population])[::-1]
        
        # Calculate selection probabilities using linear ranking
        selection_probs = (
            (2 - self.linear_ranking_s) / self.population_size
            + (2 * np.arange(self.population_size) * (self.linear_ranking_s - 1))
            / (self.population_size * (self.population_size - 1))
        )

        selected_parents = []

        # Select individuals based on linear ranking method
        rand_vals = np.random.rand(self.population_size)
        cumulative_probs = np.cumsum(selection_probs)

        for rand_val in rand_vals:
            index = np.searchsorted(cumulative_probs, rand_val, side='right')
            selected_parents.append(self.population[sorted_indices[index]])

        # Ensure the number of selected parents is even
        if len(selected_parents) % 2 != 0:
            selected_parents.pop()

        return selected_parents
   
    # EXPONENTIAL SELECTION METHOD
    def exponential_ranking_selection(self):
        # Sort indices based on fitness values in descending order
        sorted_indices = np.argsort([self.fitness(individual) for individual in self.population])[::-1]
        
        # Calculate selection probabilities using exponential ranking
        c = self.exponential_ranking_c
        probs = np.maximum(
            0,
            (c - 1) / ((c ** self.population_size) - 1) * (c ** (np.arange(self.population_size)[::-1]))
        )

        selected_parents = []

        # Select individuals based on exponential ranking method
        rand_vals = np.random.rand(self.population_size)
        cumulative_probs = np.cumsum(probs)

        indices = np.searchsorted(cumulative_probs, rand_vals, side='right')
        
        selected_parents = [self.population[sorted_indices[i]] for i in indices]

        # Ensure the number of selected parents is even
        if len(selected_parents) % 2 != 0:
            selected_parents.pop()

        return selected_parents

    # TOURNAMENT SELECTION METHOD
    def tournament_selection(self):
        selected_parents = []
        
        # Calculate fitness values for each individual in the population
        fitness_values = {str(ind): self.fitness(ind) for ind in self.population}
        
        # Determine the number of individuals in each tournament
        k = int(self.tournament_size * len(self.population))

        # Create tournaments by randomly selecting individuals
        tournaments = [random.sample(self.population, k) for _ in range(self.population_size)]
        
        for tournament in tournaments:
            # Perform tournament selection
            if random.random() < self.tournament_probability:
                best_individual = max(tournament, key=lambda x: fitness_values[str(x)])
                selected_parents.append(best_individual)
            else:
                selected_parents.append(random.choice(tournament))

        # Ensure the number of selected parents is even
        if len(selected_parents) % 2 != 0:
            selected_parents.pop()

        return selected_parents
        
    # UNIFORM SELECTION METHOD
    def uniform_selection(self):
        selected_parents = random.choices(self.population, k=self.population_size)

        # ensure the number of selected parents is even
        if len(selected_parents) % 2 != 0:
            selected_parents.pop()

        return selected_parents

    ########################################
    ############ CROSSOVERS ################
    ########################################

    # CUT-AND-CROSSFILL CROSSOVER
    def cut_and_crossfill_crossover(self, parent1_orig, parent2_orig):
        # Extract integer values from parents
        parent1, parent2 = [value for value in parent1_orig if not isinstance(value, str)], [value for value in parent2_orig if not isinstance(value, str)]
        
        # Select a random position for crossover
        position = random.randint(1, len(parent1) - 2)

        # Ensure parents have the same length
        if len(parent1) != len(parent2):
            raise ValueError("Parents must have the same length")

        # Perform cut-and-crossfill crossover
        child1 = parent1[:position].copy()
        child2 = parent2[:position].copy()

        for value in parent2:
            if value not in child1:
                child1.append(value)

        for value in parent1:
            if value not in child2:
                child2.append(value)

        # Rearrange elements to match the original order
        child1_new = [child1.pop(0) if isinstance(el, int) else el for el in parent1_orig]
        child2_new = [child2.pop(0) if isinstance(el, int) else el for el in parent2_orig]

        return [child1_new, child2_new]
    
    
    # PARTIALLY MAPPED CROSSOVER (PMX)
    def partially_mapped_crossover(self, parent1_orig, parent2_orig):
        # Extract integer values from parents
        parent1, parent2 = [v for v in parent1_orig if not isinstance(v, str)], [v for v in parent2_orig if not isinstance(v, str)]
        
        # Determine the size of the parents
        size = len(parent1)
        
        # Randomly choose two cut points
        cut1, cut2 = sorted([random.randint(0, size), random.randint(0, size)])

        # Initialize the children with the genetic material between the cut points
        child1, child2 = parent1[cut1:cut2], parent2[cut1:cut2]

        # Perform partially mapped crossover
        for i in range(size):
            if cut1 <= i < cut2:
                continue
            gene1, gene2 = parent1[i], parent2[i]

            while gene1 in child2:
                gene1 = parent1[parent2.index(gene1)]

            while gene2 in child1:
                gene2 = parent2[parent1.index(gene2)]

            child1.append(gene2)
            child2.append(gene1)

        # Rearrange elements to match the original order
        child1_new = [child1.pop(0) if isinstance(el, int) else el for el in parent1_orig]
        child2_new = [child2.pop(0) if isinstance(el, int) else el for el in parent2_orig]

        return [child1_new, child2_new]
    
    # EDGE CROSSOVER
    def edge_crossover(self, parent1_orig, parent2_orig):
        # Extract integer values from parents
        parent1 = [value for value in parent1_orig if not isinstance(value, str)]
        parent2 = [value for value in parent2_orig if not isinstance(value, str)]

        # Build an adjacent table for each parent
        adjacent_table = {}

        for parent in [parent1, parent2]:
            for i, value in enumerate(parent):
                prev_value = parent[i - 1]
                next_value = parent[(i + 1) % len(parent)]
                if value not in adjacent_table:
                    adjacent_table[value] = set()
                adjacent_table[value].add(prev_value)
                adjacent_table[value].add(next_value)

        # Initialize the child with a random starting value
        child = set()
        X = random.choice(parent1)

        # Perform edge crossover
        while len(child) != len(parent1):
            child.add(X)
            adjacent_table[X] = set()

            common_adjacent_values = set.intersection(*[adjacent_table[val] for val in child])

            if common_adjacent_values:
                X = common_adjacent_values.pop()
            elif X in adjacent_table and adjacent_table[X]:
                X = min(adjacent_table[X], key=lambda val: len(adjacent_table[val]))
            else:
                remaining_values = set(parent1) - child
                X = random.choice(list(remaining_values)) if remaining_values else None

        # Rearrange elements to match the original order
        child = list(child)
        child_new = [child.pop(0) if isinstance(el, int) else el for el in parent1_orig]

        return [child_new]
    
    # SEQUENTIAL CONSTRUCTIVE CROSSOVER
    # Ahmed, Zakir H. ”Genetic algorithm for the traveling salesman problem using sequential constructive crossover operator.”, 2010
    def scx_crossover(self, parent1_orig, parent2_orig):
        parent1 = [value for value in parent1_orig if not isinstance(value, str)]
        parent2 = [value for value in parent2_orig if not isinstance(value, str)]

        distance_matrix = self.distance_matrix
        num_cities = len(parent1)
        offspring1 = [-1] * num_cities
        offspring2 = [-1] * num_cities
        
        # Step 1: Choose the city stored in the first gene of parent 1 as the start city
        start_city = parent1[0]
        offspring1[0] = start_city
        offspring2[0] = start_city
        
        # Step 2: Construct offspring
        for i in range(1, num_cities):
            last_city = offspring1[i - 1]
            
            # Find the two cities next to and in the right of the last visited city in parent 1
            idx_p1 = parent1.index(last_city)
            city_alpha = parent1[(idx_p1 + 1) % num_cities]
            city_beta = parent1[(idx_p1 + 2) % num_cities]
            
            # Check if the cities are visited
            if city_alpha not in offspring1 and city_beta not in offspring1:
                # Both cities are unvisited, compare distances
                dp_alpha = distance_matrix[last_city][city_alpha]
                dp_beta = distance_matrix[last_city][city_beta]
                
                if dp_alpha < dp_beta:
                    offspring1[i] = city_alpha
                else:
                    offspring1[i] = city_beta
            elif city_alpha not in offspring1:
                # City alpha is unvisited, select it
                offspring1[i] = city_alpha
            elif city_beta not in offspring1:
                # City beta is unvisited, select it
                offspring1[i] = city_beta
            else:
                # Both cities are visited, find the closest unvisited city in parent 1
                unvisited_cities_p1 = [city for city in parent1 if city not in offspring1]
                closest_city_p1 = min(unvisited_cities_p1, key=lambda x: distance_matrix[last_city][x])
                offspring1[i] = closest_city_p1
        
        # Repeat the process for offspring 2 using parent 2 as the template
        for i in range(1, num_cities):
            last_city = offspring2[i - 1]
            idx_p2 = parent2.index(last_city)
            city_alpha = parent2[(idx_p2 + 1) % num_cities]
            city_beta = parent2[(idx_p2 + 2) % num_cities]
            
            if city_alpha not in offspring2 and city_beta not in offspring2:
                dp_alpha = distance_matrix[last_city][city_alpha]
                dp_beta = distance_matrix[last_city][city_beta]
                
                if dp_alpha < dp_beta:
                    offspring2[i] = city_alpha
                else:
                    offspring2[i] = city_beta
            elif city_alpha not in offspring2:
                offspring2[i] = city_alpha
            elif city_beta not in offspring2:
                offspring2[i] = city_beta
            else:
                unvisited_cities_p2 = [city for city in parent2 if city not in offspring2]
                closest_city_p2 = min(unvisited_cities_p2, key=lambda x: distance_matrix[last_city][x])
                offspring2[i] = closest_city_p2

        child1_new = [offspring1.pop(0) if isinstance(el, int) else el for el in parent1_orig]
        child2_new = [offspring2.pop(0) if isinstance(el, int) else el for el in parent2_orig]
        
        return [child1_new, child2_new]

    ########################################
    ############ MUTATIONS #################
    ########################################

    # SWAP MUTATION
    def swap_mutation(self, parent):
        # Make a copy of the parent for later rearrangement
        parent_orig = parent.copy()
        # Extract integer values from the parent
        parent = [value for value in parent_orig if not isinstance(value, str)]

        # Create a child by copying the parent
        child = parent.copy()

        # Choose two distinct positions for mutation
        i = random.randint(1, len(child) - 1)
        j_candidates = [pos for pos in range(1, len(child) - 1) if pos != i]
        j = random.choice(j_candidates)

        # Swap the values at the chosen positions
        child[i], child[j] = child[j], child[i]

        # Rearrange elements to match the original order
        child_new = [child.pop(0) if isinstance(el, int) else el for el in parent_orig]

        return child_new

    # INSERTION MUTATION
    def insertion_mutation(self, parent):
        # Make a copy of the parent for later rearrangement
        parent_orig = parent.copy()
        # Extract integer values from the parent
        parent = [value for value in parent_orig if not isinstance(value, str)]

        # Create a child by copying the parent
        child = parent.copy()

        # Choose two distinct positions for mutation
        i = random.randint(1, len(child) - 1)
        j_candidates = [pos for pos in range(1, len(child) - 1) if pos != i]
        j = random.choice(j_candidates)

        # Move the element at position j to position i + 1
        moved_element = child.pop(j)
        child.insert(i + 1, moved_element)

        # Rearrange elements to match the original order
        child_new = [child.pop(0) if isinstance(el, int) else el for el in parent_orig]

        return child_new
    
    # SCRAMBLE MUTATION
    def scramble_mutation(self, parent):
        # Make a copy of the parent for later rearrangement
        parent_orig = parent.copy()
        # Extract integer values from the parent
        parent = [value for value in parent_orig if not isinstance(value, str)]

        # Create a child by copying the parent
        child = parent.copy()

        # Choose two distinct positions for mutation
        i = random.randint(1, len(child) - 1)
        j_candidates = [pos for pos in range(1, len(child) - 1) if pos != i]
        j = random.choice(j_candidates)

        # Scramble the subsequence between positions i and j
        subsequence = child[i:j + 1]
        random.shuffle(subsequence)
        child[i:j + 1] = subsequence

        # Rearrange elements to match the original order
        child_new = [child.pop(0) if isinstance(el, int) else el for el in parent_orig]

        return child_new
    
    # INVERSION MUTATION
    def inversion_mutation(self, parent):
        # Make a copy of the parent for later rearrangement
        parent_orig = parent.copy()
        # Extract integer values from the parent
        parent = [value for value in parent_orig if not isinstance(value, str)]

        # Create a child by copying the parent
        child = parent.copy()

        # Choose two distinct positions for mutation
        i = random.randint(1, len(child) - 1)
        j_candidates = [pos for pos in range(1, len(child) - 1) if pos != i]
        j = random.choice(j_candidates)

        # Invert the subsequence between positions i and j
        child[i:j + 1] = reversed(child[i:j + 1])

        # Rearrange elements to match the original order
        child_new = [child.pop(0) if isinstance(el, int) else el for el in parent_orig]

        return child_new
    
    # RANDOM MIXED MUTATION
    def mixed_mutation(self, parent):
        # Define the mutation operators
        mutations = [self.swap_mutation, self.inversion_mutation, self.insertion_mutation, self.scramble_mutation]
        # Randomly choose a function
        random_mutation = random.choice(mutations)
        # Apply the chosen function to the parent
        return random_mutation(parent)

    ########################################
    ####### POPULATION REPLACEMENT #########
    ########################################

    # GENERATIONAL REPLACEMENT
    def generational_replacement(self, population, offspring):
        # Update the population based on generational replacement
        if len(offspring) > len(population):
            new_population = sorted(offspring, key=self.fitness)[:len(population)]
        elif len(offspring) < len(population):
            sorted_population = sorted(population, key=self.fitness)
            new_population = offspring + sorted_population[:len(population) - len(offspring)]
        else:
            new_population = offspring.copy()

        return new_population

    # REPLACE WORST (GENITOR) REPLACEMENT
    def replace_worst_replacement(self, population, offspring):
        # Update the population by replacing the worst individuals with the offspring
        new_population = population.copy()

        if len(offspring) <= len(new_population):
            new_population.sort(key=self.fitness, reverse=True)
            new_population[:len(offspring)] = offspring
        else:
            new_population = sorted(offspring, key=self.fitness)[:len(population)]

        return new_population
    
    # ROUND-ROBIN REPLACEMENT
    def round_robin_replacement(self, population, offspring):
        # Update the population using round-robin replacement
        matches_percentage = self.round_robin_matches
        combined_population = population + offspring
        
        matches = int(matches_percentage * len(combined_population))
        
        fitness_values = {str(ind): self.fitness(ind) for ind in combined_population}
        
        tournament_samples = [random.sample(combined_population, matches) for _ in range(len(combined_population))]
        
        selected_population = []
        
        for tournament_sample in tournament_samples:
            winner = max(tournament_sample, key=lambda x: fitness_values[str(x)])
            
            selected_population.append(winner)
        
        selected_population.sort(key=lambda x: fitness_values[str(x)])
        return selected_population[:len(population)]
    
    # LAMBDA-MU REPLACEMENT
    def lambda_mu_replacement(self, population, offspring):
        # Update the population using lambda-mu replacement
        combined_population = population + offspring
        combined_population.sort(key=self.fitness)
        new_population = combined_population[:len(population)]
        return new_population
    
    ########################################
    ########### MAIN FUNCTIONS #############
    ########################################

    def read_problem_instance(self, problem_path):
        with open(problem_path, 'r') as file:
            lines = file.readlines()

            # extract the number of locations and vehicles from the txt
            num_locations = int(lines[0].split()[1])
            num_vehicles = int(lines[1].split()[1])

            # initialize a matrix to store distances between locations
            distance_matrix = []
            for line in lines[3:]:  # Skip the first three lines
                # transform values equal to 9999 to 0
                distances = [0 if value == "9999" else float(value) for value in line.split()]
                distance_matrix.append(distances)

        self.num_locations = num_locations
        self.num_vehicles = num_vehicles
        self.distance_matrix = distance_matrix
    
    ########################################
        
    #Generate a random route
    #4 vehicles, 10 locations: ['D1', 5, 9, 8, 'D2', 7, 'D3', 4, 2, 'D4', 6, 3] 
    #OBS! Note that we only vist 9 locations since the first one is the depot

    
    def generate_route(self):
        random.seed(self.random_seed)
        # generate a zeros list
        individual = [0 for _ in range (1, self.num_locations + self.num_vehicles)]

        # set the first element of the list as D
        individual[0] = "D"

        # choose random positions for the depot
        possibile_depot_positions_1 = range(2, len(individual)-2, 2)
        possibile_depot_positions_2 = range(3, len(individual)-1, 2)
        chosen_possible_positions = random.choice([possibile_depot_positions_1, possibile_depot_positions_2])
        depot_positions = random.sample(chosen_possible_positions, self.num_vehicles - 1)

        # add the routes' starting points
        for i in depot_positions:
            individual[i] = "D"
        
        # generate locations list
        locations = list(range(1, self.num_locations))
        # shuffle locations
        random.shuffle(locations)

        # add the locations to the list
        individual = [element if element != 0 else locations.pop(0) for element in individual]
        
        return individual

    ########################################

    def initialize_population(self):
        random.seed(self.random_seed)
        # generate a number of solutions as the population size
        population = [self.generate_route() for _ in range(self.population_size)]

        self.population = population
        #self.best_solution = self.get_best_solution()

    ########################################
        
    def fitness(self, route):
        # change depot representations to 0
        route = [0 if isinstance(element, str) else element for element in route]

        # initialize total distance as 0
        total_distance = 0

        # pre-calculate distance matrix for faster access
        distance_matrix = self.distance_matrix

        # cache the length of the route to avoid repeated calculations
        route_length = len(route)

        # calculate total distance
        for i in range(route_length - 1):
            from_loc, to_loc = route[i], route[i + 1]
            total_distance += distance_matrix[from_loc][to_loc]

        # add distance from last location back to the depot
        last_loc = route[-1]
        total_distance += distance_matrix[last_loc][0]

        return total_distance  # Return total distance as the fitness value
    
    ########################################

    def get_best_solution(self):
        best_solution = min(self.population, key=self.fitness)
        best_solution = [0 if isinstance(element, str) else element for element in best_solution]
        
        best_solution_transformed = []
        current_route = [0]

        for item in best_solution[1:]:
            current_route.append(item)
            if item == 0:
                best_solution_transformed.append(current_route)
                current_route = [0]

        current_route.append(0)
        best_solution_transformed.append(current_route)

        self.best_solution = best_solution_transformed

        return self.best_solution

    ########################################

    def get_best_fitness(self):
        # get the best fitness
        best_fitness = min(self.fitness(solution) for solution in self.population)

        return best_fitness
    
    ########################################

    def get_elapsed_time(self):
        # get the elapsed time
        return self.elapsed_time
    
    ########################################
    
    def run(self):
        current_generation = 0                      # set current generation to 0
        previous_fitness = float('inf')             # initialize previous fitness to positive infinity
        consecutive_no_improvement = 0              # initalize early stopping counter
        random.seed(self.random_seed)               # set random seed for reproducibility

        start_time = time.time()                    # get starting time

        self.read_problem_instance(self.problem_path)   # read the problem instance
        self.initialize_population()                    # initialize the population

        while time.time() - start_time < self.time_deadline and consecutive_no_improvement < self.early_stopping_limit:
            # Selection: Choose tours from the current population to act as parents for the next generation
            selected_parents = self.SELECTION_METHODS[self.selection_method]()

            offspring = []

            for i in range(0, len(selected_parents), 2):
                parent1_orig, parent2_orig = [selected_parents[i].copy(), selected_parents[i + 1].copy()]

                if random.random() < self.cross_rate:
                    # Crossover: Create new offspring from two parents. 
                    crossover_operator = self.CROSSOVER_OPERATORS[self.crossover_operator]
                    children_new = crossover_operator(parent1_orig, parent2_orig)
                    offspring.extend(children_new)
                else:
                    offspring.extend([parent1_orig, parent2_orig])
            # Mutation: Make small changes to the offspring
            mutation_operator = self.MUTATION_OPERATORS[self.mutation_operator]
            offspring = [mutation_operator(child) if random.random() < self.mut_rate else child for child in offspring]
            # Replacement: Create new population from the current population and the offspring
            replacement_strategy = self.POPULATION_REPLACEMENT_STRATEGIES[self.population_replacement_strategy]
            self.population = replacement_strategy(self.population, offspring)
            # Update current generation
            current_generation += 1
            
            # get best fitness
            best_fitness = self.get_best_fitness()

            # check if there is an improvement in the fitness
            if best_fitness < previous_fitness:
                previous_fitness = best_fitness
                consecutive_no_improvement = 0
            else:
                consecutive_no_improvement += 1
        
        self.elapsed_time = time.time() - start_time

        
        best_solution = self.get_best_solution()
        best_fitness = self.get_best_fitness()

        # Print the best solution found
        print(f"Best Solution (Generation {current_generation}):")
        print(f"Route: {best_solution}")
        print(f"Total Distance: {best_fitness}")

    #in kwargs we should receive population_size
    def __init__(self, time_deadline, problem_path, **kwargs):
        self.MUTATION_OPERATORS = {
            'swap'      :   self.swap_mutation,
            'insertion' :   self.insertion_mutation,
            'scramble'  :   self.scramble_mutation,
            'inversion' :   self.inversion_mutation,
            "mixed"     :   self.mixed_mutation,
        }

        self.CROSSOVER_OPERATORS = {
            'cut_and_crossfill' :   self.cut_and_crossfill_crossover,
            "pmx"               :   self.partially_mapped_crossover,
            "edge"              :   self.edge_crossover,
            "scx"               :   self.scx_crossover,
        }

        self.SELECTION_METHODS = {
            'roulette_wheel'        :   self.roulette_wheel_selection,
            'linear_ranking'        :   self.linear_ranking_selection,
            'exponential_ranking'   :   self.exponential_ranking_selection,
            'tournament'            :   self.tournament_selection,
            'uniform'               :   self.uniform_selection,
        }

        self.POPULATION_REPLACEMENT_STRATEGIES = {
            'generational'          :   self.generational_replacement, 
            'replace_worst_genitor' :   self.replace_worst_replacement,
            'round_robin'           :   self.round_robin_replacement,
            'lambda_mu'             :   self.lambda_mu_replacement,
        }

        self.problem_path   =   problem_path
        self.best_solution  =   None #Atributo para guardar la mejor solución encontrada
        self.time_deadline  =   time_deadline # Límite de tiempo (en segundos) para el cómputo del algoritmo genético

        self.population         =   None
        self.num_locations      =   None
        self.num_vehicles       =   None
        self.distance_matrix    =   None

        self.random_seed = kwargs.get('random_seed', 0)
        random.seed(self.random_seed)
        
        self.mutation_operator                  =   kwargs.get('mutation_operator',                 "mixed"             )
        self.crossover_operator                 =   kwargs.get('crossover_operator',                "cut_and_crossfill" )
        self.selection_method                   =   kwargs.get('selection_method',                  "roulette_wheel"    )
        self.population_replacement_strategy    =   kwargs.get('population_replacement_strategy',   "lambda_mu"         )

        self.population_size        =   kwargs.get('population_size',       400 )
        self.cross_rate             =   kwargs.get('cross_rate',            1.00)
        self.mut_rate               =   kwargs.get('mut_rate',              0.50)
        self.early_stopping_limit   =   kwargs.get("early_stopping_limit",  100 )

        self.round_robin_matches    =   kwargs.get("elite_percentage",          0.3)
        self.exponential_ranking_c  =   kwargs.get("exponential_ranking_c",     0.5)
        self.linear_ranking_s       =   kwargs.get("linear_ranking_s",          1.5)
        self.tournament_size        =   kwargs.get("tournament_size",           0.3)
        self.tournament_probability =   kwargs.get("tournament_probability",    0.5)

    ########################################
        
ga_instance = GA(time_deadline=180, problem_path='instances/instance01.txt')
ga_instance.run()

