import random
import time
import numpy as np
class GA:

    ########################################
    ####### SELECTION OF POPULATION ########
    ########################################
    
    def roulette_wheel_selection(self):
        fitness_values = np.array([self.fitness(individual) for individual in self.population])
        total_fitness = np.sum(fitness_values)
        selection_probs = fitness_values / total_fitness

        # Use NumPy's cumulative sum to avoid the inner loop
        cumulative_probs = np.cumsum(selection_probs)
        
        # Generate random values in one go
        rand_vals = np.random.rand(self.population_size)

        selected_parents = []

        for rand_val in rand_vals:
            # Use NumPy's searchsorted to find the index efficiently
            index = np.searchsorted(cumulative_probs, rand_val, side='right')
            selected_parents.append(self.population[index])

        # Ensure the number of selected parents is even
        if len(selected_parents) % 2 != 0:
            selected_parents.pop()

        return selected_parents

    def linear_ranking_selection(self):
        sorted_indices = np.argsort([self.fitness(individual) for individual in self.population])[::-1]
        selection_probs = (
            (2 - self.linear_ranking_s) / self.population_size
            + (2 * np.arange(self.population_size) * (self.linear_ranking_s - 1))
            / (self.population_size * (self.population_size - 1))
        )

        selected_parents = []

        rand_vals = np.random.rand(self.population_size)
        cumulative_probs = np.cumsum(selection_probs)

        for rand_val in rand_vals:
            index = np.searchsorted(cumulative_probs, rand_val, side='right')
            selected_parents.append(self.population[sorted_indices[index]])

        # Ensure the number of selected parents is even
        if len(selected_parents) % 2 != 0:
            selected_parents.pop()

        return selected_parents
   
    def exponential_ranking_selection(self):
        sorted_indices = np.argsort([self.fitness(individual) for individual in self.population])[::-1]
        
        c = self.exponential_ranking_c
        probs = np.maximum(
            0,
            (c - 1) / ((c ** self.population_size) - 1) * (c ** (np.arange(self.population_size)[::-1]))
        )

        selected_parents = []

        rand_vals = np.random.rand(self.population_size)
        cumulative_probs = np.cumsum(probs)

        indices = np.searchsorted(cumulative_probs, rand_vals, side='right')
        
        selected_parents = [self.population[sorted_indices[i]] for i in indices]

        # Ensure the number of selected parents is even
        if len(selected_parents) % 2 != 0:
            selected_parents.pop()

        return selected_parents

    def tournament_selection(self):
        selected_parents = []
        
        # Precalculate fitness values for all individuals in the population
        fitness_values = {str(ind): self.fitness(ind) for ind in self.population}
        k = int(self.tournament_size * len(self.population))
        # Precalculate tournaments
        tournaments = [random.sample(self.population, k) for _ in range(self.population_size)]
        
        for tournament in tournaments:
            if random.random() < self.tournament_probability:
                best_individual = max(tournament, key=lambda x: fitness_values[str(x)])
                selected_parents.append(best_individual)
            else:
                selected_parents.append(random.choice(tournament))

        # Ensure the number of selected parents is even
        if len(selected_parents) % 2 != 0:
            selected_parents.pop()

        return selected_parents
        
    def uniform_selection(self):
        selected_parents = random.choices(self.population, k=self.population_size)

        # Ensure the number of selected parents is even
        if len(selected_parents) % 2 != 0:
            selected_parents.pop()

        return selected_parents

    ########################################
    ############ CROSSOVERS ################
    ########################################

    # CUT-AND-CROSSFILL CROSSOVER
    def cut_and_crossfill_crossover(self, parent1_orig, parent2_orig):
        parent1, parent2 = [value for value in parent1_orig if not isinstance(value, str)], [value for value in parent2_orig if not isinstance(value, str)]
        position = random.randint(1, len(parent1) - 2)

        if len(parent1) != len(parent2):
            raise ValueError("Parents must have the same length")

        child1 = parent1[:position].copy()
        child2 = parent2[:position].copy()

        for value in parent2:
            if value not in child1:
                child1.append(value)

        for value in parent1:
            if value not in child2:
                child2.append(value)

        child1_new = [child1.pop(0) if isinstance(el, int) else el for el in parent1_orig]
        child2_new = [child2.pop(0) if isinstance(el, int) else el for el in parent2_orig]

        return [child1_new, child2_new]
    
    
    # PARTIALLY MAPPED CROSSOVER (PMX)
    def partially_mapped_crossover(self, parent1_orig, parent2_orig):
        parent1, parent2 = [v for v in parent1_orig if not isinstance(v, str)], [v for v in parent2_orig if not isinstance(v, str)]
        size = len(parent1)
        cut1, cut2 = sorted([random.randint(0, size), random.randint(0, size)])

        child1, child2 = parent1[cut1:cut2], parent2[cut1:cut2]

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

        child1_new = [child1.pop(0) if isinstance(el, int) else el for el in parent1_orig]
        child2_new = [child2.pop(0) if isinstance(el, int) else el for el in parent2_orig]

        return [child1_new, child2_new]
    
    # EDGE CROSSOVER
    def edge_crossover(self, parent1_orig, parent2_orig):
        parent1 = [value for value in parent1_orig if not isinstance(value, str)]
        parent2 = [value for value in parent2_orig if not isinstance(value, str)]

        adjacent_table = {}

        for parent in [parent1, parent2]:
            for i, value in enumerate(parent):
                prev_value = parent[i - 1]
                next_value = parent[(i + 1) % len(parent)]
                if value not in adjacent_table:
                    adjacent_table[value] = set()
                adjacent_table[value].add(prev_value)
                adjacent_table[value].add(next_value)

        child = set()
        X = random.choice(parent1)

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

        child = list(child)

        child_new = [child.pop(0) if isinstance(el, int) else el for el in parent1_orig]

        return [child_new]
    
    def mixed_crossover(self, parent1, parent2):
        # List of functions
        crossovers = [self.edge_crossover, self.partially_mapped_crossover, self.cut_and_crossfill_crossover]
        random_crossover = random.choice(crossovers)
        # Randomly choose a function
        return random_crossover(parent1, parent2)
    ########################################
    ############ MUTATIONS #################
    ########################################

    def swap_mutation(self, parent):
        parent_orig = parent.copy()
        parent = [value for value in parent_orig if not isinstance(value, str)]

        child = parent.copy()

        i = random.randint(1, len(child) - 1)
        j_candidates = [pos for pos in range(1, len(child) - 1) if pos != i]
        j = random.choice(j_candidates)

        child[i], child[j] = child[j], child[i]

        child_new = [child.pop(0) if isinstance(el, int) else el for el in parent_orig]

        return child_new

    # INSERTION MUTATION
    def insertion_mutation(self, parent):
        parent_orig = parent.copy()
        parent = [value for value in parent_orig if not isinstance(value, str)]

        child = parent.copy()

        i = random.randint(1, len(child) - 1)
        j_candidates = [pos for pos in range(1, len(child) - 1) if pos != i]
        j = random.choice(j_candidates)

        moved_element = child.pop(j)
        child.insert(i + 1, moved_element)

        child_new = [child.pop(0) if isinstance(el, int) else el for el in parent_orig]

        return child_new
    
    # SCRAMBLE MUTATION
    def scramble_mutation(self, parent):
        parent_orig = parent.copy()
        parent = [value for value in parent_orig if not isinstance(value, str)]

        child = parent.copy()

        i = random.randint(1, len(child) - 1)
        j_candidates = [pos for pos in range(1, len(child) - 1) if pos != i]
        j = random.choice(j_candidates)

        subsequence = child[i:j + 1]
        random.shuffle(subsequence)
        child[i:j + 1] = subsequence

        child_new = [child.pop(0) if isinstance(el, int) else el for el in parent_orig]

        return child_new
    
    def inversion_mutation(self, parent):
        parent_orig = parent.copy()
        parent = [value for value in parent_orig if not isinstance(value, str)]

        child = parent.copy()

        i = random.randint(1, len(child) - 1)
        j_candidates = [pos for pos in range(1, len(child) - 1) if pos != i]
        j = random.choice(j_candidates)

        child[i:j + 1] = reversed(child[i:j + 1])

        child_new = [child.pop(0) if isinstance(el, int) else el for el in parent_orig]

        return child_new
    
    def mixed_mutation(self, parent):
        # List of functions
        mutations = [self.swap_mutation, self.inversion_mutation, self.insertion_mutation, self.scramble_mutation]
        random_mutation = random.choice(mutations)
        # Randomly choose a function
        return random_mutation(parent)

    ########################################
    ####### POPULATION REPLACEMENT #########
    ########################################

    def generational_replacement(self, population, offspring):
        """
        Generational replacement strategy.

        -In this strategy, the entire population of the current
        generation is replaced by the new generation of individuals (offspring).
        """
        if len(offspring) > len(population):
            new_population = sorted(offspring, key=self.fitness)[:len(population)]
        elif len(offspring) < len(population):
            sorted_population = sorted(population, key=self.fitness)
            new_population = offspring + sorted_population[:len(population) - len(offspring)]
        else:
            new_population = offspring.copy()

        return new_population

    def replace_worst_replacement(self, population, offspring):
        """
        Replace worst (GENITOR) replacement strategy.

        - In this strategy, only the worst individuals
        in the population are replaced by the newly generated individuals.
        At each iteration, the least fit individuals are replaced by
        the newly generated individuals.
        """
        new_population = population.copy()

        # Replace the worst individuals with offspring
        if len(offspring) <= len(new_population):
            new_population.sort(key=self.fitness, reverse=True)
            new_population[:len(offspring)] = offspring
        else:
            new_population = sorted(offspring, key=self.fitness)[:len(population)]

        return new_population
    
    # ROUND-ROBIN REPLACEMENT
    def round_robin_replacement(self, population, offspring):
        matches_percentage = self.round_robin_matches
        # Combine the current population and offspring
        combined_population = population + offspring
        
        matches = int(matches_percentage * len(combined_population))
        
        # Precalculate fitness values for all individuals in the combined population
        fitness_values = {str(ind): self.fitness(ind) for ind in combined_population}
        
        # Sample all tournament individuals at once
        tournament_samples = [random.sample(combined_population, matches) for _ in range(len(combined_population))]
        
        # Perform tournament selection
        selected_population = []
        
        for tournament_sample in tournament_samples:
            # Find the individual with the maximum victories in the tournament
            winner = max(tournament_sample, key=lambda x: fitness_values[str(x)])
            
            # Add the winner to the selected population
            selected_population.append(winner)
        
        selected_population.sort(key=lambda x: fitness_values[str(x)])
        return selected_population[:len(population)]
    
    def lambda_mu_replacement(self, population, offspring):
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

            # Extract the number of locations and vehicles from the txt
            num_locations = int(lines[0].split()[1])
            num_vehicles = int(lines[1].split()[1])

            # Initialize a matrix to store distances between locations
            distance_matrix = []
            for line in lines[3:]:  # Skip the first three lines
                # Transform values equal to 9999 to 0
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
        individual = [0 for _ in range (1, self.num_locations + self.num_vehicles)]

        individual[0] = "D"

        possibile_depot_positions_1 = range(2, len(individual)-2, 2)
        possibile_depot_positions_2 = range(3, len(individual)-1, 2)

        chosen_possible_positions = random.choice([possibile_depot_positions_1, possibile_depot_positions_2])

        depot_positions = random.sample(chosen_possible_positions, self.num_vehicles - 1)

        for i in depot_positions:
            individual[i] = "D"
        
        # generate locations list
        locations = list(range(1, self.num_locations))
        # shuffle locations
        random.shuffle(locations)

        individual = [element if element != 0 else locations.pop(0) for element in individual]
        
        #print("individual", individual)
        return individual
        

    ########################################

    def initialize_population(self):
        population = [self.generate_route() for _ in range(self.population_size)]
        # Assert that all members in the population have the same length
        #assert all(len(route) == len(population[0]) for route in population), "All routes must have the same length"
        self.population = population
        self.best_solution = self.get_best_solution()

    ########################################
        
    def fitness(self, route):
        # Change depot representations to 0
        route = [0 if isinstance(element, str) else element for element in route]

        # initialize total distance as 0
        total_distance = 0

        # Pre-calculate distance matrix for faster access
        distance_matrix = self.distance_matrix

        # Cache the length of the route to avoid repeated calculations
        route_length = len(route)

        for i in range(route_length - 1):
            from_loc, to_loc = route[i], route[i + 1]
            total_distance += distance_matrix[from_loc][to_loc]

        # Add distance from last location back to the depot
        last_loc = route[-1]
        total_distance += distance_matrix[last_loc][0]

        return total_distance  # Return total distance as the fitness value

    ########################################

    def fitness_old(self, route):
        total_distance = 0
        result = []; chunk = []
        for item in route:
            if isinstance(item, str):
                if chunk:
                    chunk.append(0)
                    result.append(chunk)
                    chunk = []
                chunk.append(0)
            else:
                chunk.append(item)
        if chunk:
            chunk.append(0)
            result.append(chunk)
        
        for vehicle_route in result:
            vehicle_distance = 0
            for j in range(len(vehicle_route) - 1):
                from_loc = vehicle_route[j]
                to_loc = vehicle_route[j + 1]
                vehicle_distance += self.distance_matrix[from_loc - 1][to_loc - 1]
            total_distance += vehicle_distance
        return total_distance
    
    ########################################

    def get_best_solution(self):
        """
        Method to return the best solution found so far.
        """
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

        return best_solution_transformed

    ########################################

    def get_best_fitness(self):
        """
        Get the best fitness score found during the execution of the genetic algorithm.
        Returns:
            float: The best fitness score.
        """
        #print(len(self.population))
        best_fitness = min(self.fitness(solution) for solution in self.population)

        return best_fitness
    
    ########################################

    def get_elapsed_time(self):
        return self.elapsed_time
    
    ########################################
    
    def run(self):
        """
        Método que ejecuta el algoritmo genético. Debe crear la población inicial y
        ejecutar el bucle principal del algoritmo genético
        TODO: Se debe implementar aquí la lógica del algoritmo genético
        """
        
        self.read_problem_instance(self.problem_path)   # read the problem instance
        self.initialize_population()                    # initialize the population

        
        current_generation = 0                      # set current generation to 0
        previous_fitness = float('inf')             # initialize previous fitness to positive infinity
        consecutive_no_improvement = 0

        start_time = time.time()                    # get starting time
        #while time.time() - start_time < self.time_deadline:
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
            #print(best_fitness)

            # check if there is an improvement in the fitness
            if best_fitness < previous_fitness:
                previous_fitness = best_fitness
                consecutive_no_improvement = 0
            else:
                consecutive_no_improvement += 1
        
        self.elapsed_time = time.time() - start_time

        # Print the best solution found
        best_solution = self.get_best_solution()
        best_fitness = self.get_best_fitness()

        #print(f"Best Solution (Generation {current_generation}):")
        #print(f"Route: {best_solution}")
        #print(f"Total Distance: {best_fitness}")

    #in kwargs we should receive population_size
    def __init__(self, time_deadline, problem_path, **kwargs):
        """
        Inicializador de los objetos de la clase. Usar
        este método para hacer todo el trabajo previo y necesario
        para configurar el algoritmo genético
        Args:
            problem_path: Cadena de texto que determina la ruta en la que se encuentra la definición del problema
            time_deadline: Límite de tiempo que el algoritmo genético puede computar
        """
        self.MUTATION_OPERATORS = {
            'swap': self.swap_mutation,
            'insertion': self.insertion_mutation,
            'scramble': self.scramble_mutation,
            'inversion': self.inversion_mutation,
            "mixed": self.mixed_mutation,
        }

        self.CROSSOVER_OPERATORS = {
            'cut_and_crossfill': self.cut_and_crossfill_crossover,
            "pmx": self.partially_mapped_crossover,
            "edge": self.edge_crossover,
            "mixed": self.mixed_crossover,
        }

        self.SELECTION_METHODS = {
            'roulette_wheel': self.roulette_wheel_selection,
            'linear_ranking': self.linear_ranking_selection,
            'exponential_ranking': self.exponential_ranking_selection,
            'tournament': self.tournament_selection,
            'uniform': self.uniform_selection,
        }

        self.POPULATION_REPLACEMENT_STRATEGIES = {
            'generational': self.generational_replacement, 
            'replace_worst_genitor': self.replace_worst_replacement,
            'round_robin': self.round_robin_replacement,
            'lambda_mu': self.lambda_mu_replacement,
        }

        self.problem_path = problem_path
        self.best_solution = None #Atributo para guardar la mejor solución encontrada
        self.time_deadline = time_deadline # Límite de tiempo (en segundos) para el cómputo del algoritmo genético

        self.population = None
        self.num_locations = None
        self.num_vehicles = None
        self.distance_matrix = None

        self.random_seed = kwargs.get('random_seed', 0)
        random.seed(self.random_seed)
        
        self.mutation_operator = kwargs.get('mutation_operator', "swap")
        self.crossover_operator = kwargs.get('crossover_operator', "edge")
        self.selection_method = kwargs.get('selection_method', "linear_ranking")
        self.population_replacement_strategy = kwargs.get('population_replacement_strategy', "lambda_mu")

        self.population_size = kwargs.get('population_size', 100)
        self.cross_rate = kwargs.get('cross_rate', 0.80)
        self.mut_rate = kwargs.get('mut_rate', 0.30)
        self.early_stopping_limit = kwargs.get("early_stopping_limit", 50)

        self.elite_percentage = kwargs.get("elite_percentage", 0.2)
        self.round_robin_matches = kwargs.get("elite_percentage", 0.3)
        self.exponential_ranking_c = kwargs.get("exponential_ranking_c", 0.5)
        self.linear_ranking_s = kwargs.get("linear_ranking_s", 1.5)
        self.tournament_size = kwargs.get("tournament_size", 0.3)
        self.tournament_probability = kwargs.get("tournament_probability", 0.5)

    ########################################
        
#ga_instance = GA(time_deadline=180, problem_path='instances/instance01.txt')
#ga_instance.run()

