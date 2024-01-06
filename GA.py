import random
import time
import numpy as np
class GA:

    ########################################
    ####### SELECTION OF POPULATION ########
    ########################################
    
    def roulette_wheel_selection(self):
        total_fitness = sum(self.fitness(individual) for individual in self.population)
        selection_probs = [(self.fitness(individual) / total_fitness) for individual in self.population]
        selected_parents = []

        for _ in range(self.population_size):
            rand_val = random.random()  # Generate a random value between 0 and 1
            cumulative_prob = 0
            for i, prob in enumerate(selection_probs):
                cumulative_prob += prob
                if rand_val <= cumulative_prob:
                    selected_parents.append(self.population[i])
                    break
        
        # Ensure the number of selected parents is even
        if len(selected_parents) % 2 != 0:
            selected_parents.pop()

        return selected_parents

    def linear_ranking_selection(self):
        sorted_population = sorted(self.population, key=self.fitness, reverse=True)
        selection_probs = []

        for i, _ in enumerate(sorted_population):
            prob = (2 - self.linear_ranking_s) / self.population_size + (2 * i * (self.linear_ranking_s - 1)) / (self.population_size * (self.population_size - 1))
            selection_probs.append(prob)

        selected_parents = []
        for _ in range(self.population_size):
            rand_val = random.random()  # Generate a random value between 0 and 1
            cumulative_prob = 0
            for i, prob in enumerate(selection_probs):
                cumulative_prob += prob
                if rand_val <= cumulative_prob:
                    selected_parents.append(sorted_population[i])
                    break

        # Ensure the number of selected parents is even
        if len(selected_parents) % 2 != 0:
            selected_parents.pop()

        return selected_parents
   
    def exponential_ranking_selection(self):
        sorted_population = sorted(self.population, key=self.fitness, reverse=True)
        selection_probs = []

        for i in range(self.population_size):
            prob = max(0, (self.exponential_ranking_c - 1) / ((self.exponential_ranking_c ** self.population_size) - 1) * (self.exponential_ranking_c ** (self.population_size - i - 1)))
            selection_probs.append(prob)

        selected_parents = []
        for _ in range(self.population_size):
            rand_val = random.random()  # Generate a random value between 0 and 1
            cumulative_prob = 0
            for i, prob in enumerate(selection_probs):
                cumulative_prob += prob
                if rand_val <= cumulative_prob:
                    selected_parents.append(sorted_population[i])
                    break
        
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
    def cut_and_crossfill_crossover(self, parents):
        if len(parents) % 2 != 0:
            raise ValueError("Number of parents must be even for cut-and-crossfill crossover.")

        offspring = []

        for i in range(0, len(parents), 2):
            parent1_orig, parent2_orig = parents[i].copy(), parents[i + 1].copy()
            if random.random() < self.cross_rate:
                
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

                child1_new = []
                index = 0
                for el in parent1_orig:
                    if isinstance(el, int):
                        child1_new.append(child1[index])
                        index += 1
                    else:
                        child1_new.append(el)

                child2_new = []
                index = 0
                for el in parent2_orig:
                    if isinstance(el, int):
                        child2_new.append(child2[index])
                        index += 1
                    else:
                        child2_new.append(el)

                offspring.extend([child1_new, child2_new])
            else:
                offspring.extend([parent1_orig, parent2_orig])

        return offspring
    
    # PARTIALLY MAPPED CROSSOVER (PMX)
    def partially_mapped_crossover(self, parents):
        if len(parents) % 2 != 0:
            raise ValueError("Number of parents must be even for cut-and-crossfill crossover.")

        offspring = []

        for i in range(0, len(parents), 2):
            parent1_orig, parent2_orig = parents[i].copy(), parents[i + 1].copy()
            if random.random() < self.cross_rate:
                
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

                child1_new = []
                index = 0
                for el in parent1_orig:
                    if isinstance(el, int):
                        child1_new.append(child1[index])
                        index += 1
                    else:
                        child1_new.append(el)
                
                child2_new = []
                index = 0
                for el in parent2_orig:
                    if isinstance(el, int):
                        child2_new.append(child2[index])
                        index += 1
                    else:
                        child2_new.append(el)

                offspring.extend([child1_new, child2_new])
            else:
                offspring.extend([parent1_orig, parent2_orig])

        return offspring
    
    # EDGE CROSSOVER
    def edge_crossover(self, parents):
        """
        Perform Edge Crossover on two parents.

        Parameters:
        - parent1: List representing the first parent
        - parent2: List representing the second parent

        Returns:
        - child: List representing the child after crossover
        """
        if len(parents) % 2 != 0:
            raise ValueError("Number of parents must be even for cut-and-crossfill crossover.")

        offspring = []

        for i in range(0, len(parents), 2):
            parent1_orig, parent2_orig = parents[i].copy(), parents[i + 1].copy()

            if random.random() < self.cross_rate:

                parent1 = [value for value in parent1_orig if not isinstance(value, str)]
                parent2 = [value for value in parent2_orig if not isinstance(value, str)]

                # Step 1: Construct a table of values adjacent to each value of both parents
                adjacent_table = {}
                for parent in [parent1, parent2]:
                    for i in range(len(parent)):
                        value = parent[i]
                        if value not in adjacent_table:
                            adjacent_table[value] = set()

                        if i == 0:
                            adjacent_table[value].add(parent[i + 1])
                            adjacent_table[value].add(parent[-1])
                        elif i == len(parent) - 1:
                            adjacent_table[value].add(parent[i - 1])
                            adjacent_table[value].add(parent[0])
                        else:
                            adjacent_table[value].add(parent[i - 1])
                            adjacent_table[value].add(parent[i + 1])

                # Step 2: Child initialization
                child = set()

                # Step 3: X <- Initial element from one of the two parents
                X = random.choice(parent1)

                # Step 4: Main loop
                while len(child) != len(parent1):
                    # Step 4.1: Child <- Child ∪ X
                    child.add(X)

                    # Step 4.2: Remove references to X in adjacent lists
                    for adjacent_list in adjacent_table.values():
                        adjacent_list.discard(X)

                    # Step 4.3: Check for common two equal adjacent values
                    common_adjacent_values = set.intersection(*[adjacent_table[val] for val in parent1])

                    if common_adjacent_values:
                        X = common_adjacent_values.pop()
                    # Step 4.4: Choose element from adjacent list with the shortest adjacent list
                    elif X in adjacent_table and adjacent_table[X]:
                        X = min(adjacent_table[X], key=lambda val: len(adjacent_table[val]))
                    # Step 4.5: Choose random value not in child
                    else:  
                        remaining_values = set(parent1) - child
                        X = random.choice(list(remaining_values)) if remaining_values else None

                child = list(child)

                child_new = []
                index = 0
                for el in parent1_orig:
                    if isinstance(el, int):
                        child_new.append(child[index])
                        index += 1
                    else:
                        child_new.append(el)

                offspring.extend([child_new])
            else:
                offspring.extend([parent1_orig, parent2_orig])
            
        return offspring

    ########################################
    ############ MUTATIONS #################
    ########################################

    # SWAP MUTATION
    def swap_mutation(self, parent):
        if random.random() < self.mut_rate:
            parent_orig = parent.copy()
            parent = [value for value in parent_orig if not isinstance(value, str)]

            # Step 1: Copy parent into child
            child = parent.copy()
            #print("child", child)

            # Step 2: Randomly select position i ∈ {1, …, n}
            i = random.randint(1, len(child) - 1)

            # Step 3: Randomly select position j ∈ {1, …, i − 1, i + 1, …, n}
            j_candidates = list(range(1, i)) + list(range(i + 1, len(child) - 1))
            j = random.choice(j_candidates)

            # Step 4: Exchange content between positions i and j
            child[i], child[j] = child[j], child[i]


            child_new = []
            index = 0
            for el in parent_orig:
                if isinstance(el, int):
                    child_new.append(child[index])
                    index += 1
                else:
                    child_new.append(el)
            

            return child_new
        else:
            return parent

    # INSERTION MUTATION
    def insertion_mutation(self, parent):
        if random.random() < self.mut_rate:
            parent_orig = parent.copy()
            parent = [value for value in parent_orig if not isinstance(value, str)]
            # Step 1: Copy parent into child
            child = parent.copy()

            # Step 2: Select random position i ∈ {1, …, n}
            i = random.randint(1, len(child) - 1)

            # Step 3: Select random position j ∈ {1, …, i − 1, i + 1, …, n}
            j_candidates = list(range(1, i)) + list(range(i + 1, len(child) - 1))
            j = random.choice(j_candidates)

            # Step 4: Shift the content of j into i+1, moving to the right every value between i and j
            moved_element = child.pop(j)
            child.insert(i + 1, moved_element)

            child_new = []
            index = 0
            for el in parent_orig:
                if isinstance(el, int):
                    child_new.append(child[index])
                    index += 1
                else:
                    child_new.append(el)
            

            return child
        else:
            return parent
    
    # SCRAMBLE MUTATION
    def scramble_mutation(self, parent):
        if random.random() < self.mut_rate:
            parent_orig = parent.copy()
            parent = [value for value in parent_orig if not isinstance(value, str)]
            # Step 1: Copy parent into child
            child = parent.copy()

            # Step 2: Select random position i ∈ {1, …, n}
            i = random.randint(1, len(child) - 1)

            # Step 3: Select random position j ∈ {1, …, i − 1, i + 1, …, n}
            j_candidates = list(range(1, i)) + list(range(i + 1, len(child) - 1))
            j = random.choice(j_candidates)

            # Step 4: Shuffle content between positions i and j
            subsequence = child[i:j + 1]
            random.shuffle(subsequence)
            child[i:j + 1] = subsequence

            child_new = []
            index = 0
            for el in parent_orig:
                if isinstance(el, int):
                    child_new.append(child[index])
                    index += 1
                else:
                    child_new.append(el)
            

            return child
        else:
            return parent
    
    # INVERSION MUTATION
    def inversion_mutation(self, parent):
        if random.random() < self.mut_rate:
            parent_orig = parent.copy()
            parent = [value for value in parent_orig if not isinstance(value, str)]
            # Step 1: Copy parent into child
            child = parent.copy()

            # Step 2: Select random position i ∈ {1, …, n}
            i = random.randint(1, len(child) - 1)

            # Step 3: Select random position j ∈ {1, …, i − 1, i + 1, …, n}
            j_candidates = list(range(1, i)) + list(range(i + 1, len(child) - 1))
            j = random.choice(j_candidates)

            # Step 4: Invert order between positions i and j
            child[i:j + 1] = reversed(child[i:j + 1])

            child_new = []
            index = 0
            for el in parent_orig:
                if isinstance(el, int):
                    child_new.append(child[index])
                    index += 1
                else:
                    child_new.append(el)
            

            return child
        else:
            return parent

    ########################################
    ####### POPULATION REPLACEMENT #########
    ########################################

    #GENERATIONAL REPLACEMENT    
    def generational_replacement(self, population, offspring):
        """
        Generational replacement strategy.

        -In this strategy, the entire population of the current
        generation is replaced by the new generation of individuals (offspring).
        """
        # Combines parents and children to form the new population
        if len(offspring) > len(population):
            new_population = sorted(offspring, key=self.fitness)[:len(population)]
        elif len(offspring) < len(population):
            sorted_population = sorted(population, key=self.fitness)
            new_population = offspring.copy() + sorted_population
            new_population = new_population[:len(population)]
        else:
            new_population = offspring.copy()

        return new_population
       
    # STEADY STATE REPLACEMENT
    def steady_state_replacement(self, population, offspring):
        """
        Steady-state replacement strategy.

        - in this strategy, only a small subset of the population is replaced at each iteration,
        keeping most of the individuals from the previous generation.
        Implementation: In the steady-state strategy, at each iteration,
        some individuals are replaced by the new individuals, retaining part
        of the previous population.
        """
        # Choose one or more individuals from the current population to
        # replace with offspring
        replace_indices = random.sample(range(len(population)), len(offspring))

        for idx, offspring_individual in zip(replace_indices, offspring):
            population[idx] = offspring_individual

        return population

    # REPLACE (WORST)
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
            new_population[:len(offspring)] = offspring.copy()
        else:
            new_population = offspring.copy()
            new_population = sorted(new_population, key=self.fitness)[:len(population)]

        return new_population
   
    # ELITISM REPLACEMENT 
    def elitism_replacement(self, population, offspring):
        num_elites = int(self.elite_percentage * len(population))
        elites = sorted(population, key=self.fitness)[:num_elites]
        offspring_sorted = sorted(offspring, key=self.fitness)
        new_population = elites + offspring_sorted

        return new_population[:len(population)]
    
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
    
    # 𝝀-𝝁 REPLACEMENT
    def lambda_mu_replacement(self, population, offspring):
        combined_population = population.copy() + offspring.copy()
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

        for i in range(len(route)-1):
            from_loc = route[i]
            to_loc = route[i+1]
            total_distance += self.distance_matrix[from_loc][to_loc]
        
        # Add distance from last location back to the depot
        last_loc = route[-1]
        total_distance += self.distance_matrix[last_loc][0]
    
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
        best_fitness = min(self.fitness(solution) for solution in self.population)

        return best_fitness
    
    ########################################
    
    def run(self):
        """
        Método que ejecuta el algoritmo genético. Debe crear la población inicial y
        ejecutar el bucle principal del algoritmo genético
        TODO: Se debe implementar aquí la lógica del algoritmo genético
        """
        
        self.read_problem_instance(self.problem_path)   # read the problem instance
        self.initialize_population()                    # initialize the population

        start_time = time.time()                    # get starting time
        current_generation = 0                      # set current generation to 0
        previous_fitness = float('inf')             # initialize previous fitness to positive infinity
        consecutive_no_improvement = 0

        #while time.time() - start_time < self.time_deadline:
        while time.time() - start_time < self.time_deadline and consecutive_no_improvement < self.early_stopping_limit:
            # Selection: Choose tours from the current population to act as parents for the next generation
            selected_parents = self.SELECTION_METHODS[self.selection_method]()
            
            # Crossover: Create new offspring from two parents.
            crossover_operator = self.CROSSOVER_OPERATORS[self.crossover_operator]
            offspring = crossover_operator(selected_parents)

            # Mutation: Make small changes to the offspring
            mutation_operator = self.MUTATION_OPERATORS[self.mutation_operator]
            offspring = mutation_operator(offspring)

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
        }

        self.CROSSOVER_OPERATORS = {
            'cut_and_crossfill': self.cut_and_crossfill_crossover,
            "pmx": self.partially_mapped_crossover,
            "edge": self.edge_crossover,
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
            'steady_state': self.steady_state_replacement,
            'replace_worst_genitor': self.replace_worst_replacement,
            'elitism': self.elitism_replacement,
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
        self.selection_method = kwargs.get('selection_method', "tournament")
        self.population_replacement_strategy = kwargs.get('population_replacement_strategy', "round_robin")

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
        
ga_instance = GA(time_deadline=60, problem_path='instances/instance02.txt')
ga_instance.run()

