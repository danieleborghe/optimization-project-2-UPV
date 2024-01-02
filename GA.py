import random
from collections import defaultdict


class GA:
    
    ########################################
    ####### POPULATION REPLACEMENT #########
    ########################################

    #GENERATIONAL REPLACEMENT    
    def generational_replacement(self, parents, offspring):
        """
        Generational replacement strategy.

        -In this strategy, the entire population of the current
        generation is replaced by the new generation of individuals (offspring).
        """
        # Combines parents and children to form the new population
        new_population = parents + offspring
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
        #Combines parents and children to form the new population
        combined_population = population + offspring

        # Sort the combined population by fitness rating (best to worst) 
        combined_population.sort(key=lambda x: self.fitness(x))

        # Replace the worst individuals with offspring
        new_population = combined_population[:len(population)]

        return new_population
    
    # ELITISM REPLACEMENT 
    def elitism_replacement(self, parents, offspring, elite_percentage=0.1):
        num_elites = int(elite_percentage * len(parents))
        elites = sorted(parents, key=self.fitness)[:num_elites]
        new_population = elites + offspring

        return new_population
    
    # ROUND-ROBIN REPLACEMENT
    def round_robin_replacement(self, parents, offspring):
        new_population = parents.copy()
      
        for i in range(0, len(parents), len(offspring)):
          new_population[i:i+len(offspring)] = offspring

        return new_population
    
    # 𝝀-𝝁 REPLACEMENT
    def lambda_mu_replacement(self, parents, offspring, mu):
        combined_population = parents + offspring
        combined_population.sort(key=self.fitness)
        new_population = combined_population[:mu]

        return new_population

    ########################################
    ####### SELECTION OF POPULATION ########
    ########################################
    
    def roulette_wheel_selection(self):
        total_fitness = sum(individual.fitness for individual in self.population)
        selection_probs = [individual.fitness / total_fitness for individual in self.population]

        selected_parents = []
        for _ in range(self.population_size):
            rand_val = random.random()  # Generate a random value between 0 and 1
            cumulative_prob = 0
            for i, prob in enumerate(selection_probs):
                cumulative_prob += prob
                if rand_val <= cumulative_prob:
                    selected_parents.append(self.population[i])
                    break

        return selected_parents

    def linear_ranking_selection(self, s=1.5):
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        selection_probs = []

        for i, _ in enumerate(sorted_population):
            prob = (2 - s) / self.population_size + (2 * i * (s - 1)) / (self.population_size * (self.population_size - 1))
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

        return selected_parents
    
    def exponential_ranking_selection(self, c=0.6):
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        selection_probs = []

        for i, _ in enumerate(sorted_population):
            prob = (c - 1) / ((c ** self.population_size) - 1) * (c ** self.population_size - i - 1)
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

        return selected_parents

    def tournament_selection(self, k=2, p=0.9):
        selected_parents = []
        for _ in range(self.population_size):
            tournament = random.sample(self.population, k)  # Select k individuals randomly
            if random.random() < p:
                best_individual = max(tournament, key=lambda x: x.fitness)  # Select the best individual
                selected_parents.append(best_individual)
            else:
                selected_parents.append(random.choice(tournament))  # Randomly select from the tournament

        return selected_parents

    def uniform_selection(self):
        selected_parents = random.choices(self.population, k=self.population_size)
        return selected_parents

    ########################################
    ############ MUTATIONS #################
    ########################################

    # SWAP MUTATION
    def swap_mutation(self, parent):
        # Step 1: Copy parent into child
        child = parent.copy()

        # Step 2: Randomly select position i ∈ {1, …, n}
        i = random.randint(1, len(child) - 1)

        # Step 3: Randomly select position j ∈ {1, …, i − 1, i + 1, …, n}
        j_candidates = list(range(1, i)) + list(range(i + 1, len(child) - 1))
        j = random.choice(j_candidates)

        # Step 4: Exchange content between positions i and j
        child[i], child[j] = child[j], child[i]

        return child

    # INSERTION MUTATION
    def insertion_mutation(self, parent):
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

        return child
    
    # SCRAMBLE MUTATION
    def scramble_mutation(self, parent):
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

        return child
    
    # INVERSION MUTATION
    def inversion_mutation(self, parent):
        # Step 1: Copy parent into child
        child = parent.copy()

        # Step 2: Select random position i ∈ {1, …, n}
        i = random.randint(1, len(child) - 1)

        # Step 3: Select random position j ∈ {1, …, i − 1, i + 1, …, n}
        j_candidates = list(range(1, i)) + list(range(i + 1, len(child) - 1))
        j = random.choice(j_candidates)

        # Step 4: Invert order between positions i and j
        child[i:j + 1] = reversed(child[i:j + 1])

        return child
    
    ########################################
    ############ CROSSOVERS ################
    ########################################
    
    # CUT-AND-CROSSFILL CROSSOVER
    def cut_and_crossfill_crossover(self, parent1, parent2):
        # Step 1: Select random position i ∈ {1, …, n-2}
        i = random.randint(1, len(parent1) - 2)

        # Ensure the first element of the parents remains the first element of the children
        child1 = [parent1[0]] + parent1[1:i] + parent2[i:]
        child2 = [parent2[0]] + parent2[1:i] + parent1[i:]

        # Step 3: Scan from the second segment of parent 2 and fill with values not present in child 1
        for gene in parent2[i:] + parent2[1:i]:
            if gene not in child1:
                child1.append(gene)

        # Step 4: Do the same for parent 1 and child 2
        for gene in parent1[i:] + parent1[1:i]:
            if gene not in child2:
                child2.append(gene)

        return child1, child2
    
    # PARTIALLY MAPPED CROSSOVER (PMX)
    def partially_mapped_crossover(self, parent1, parent2):
        # Step 1: Select two random positions i, j ∈ {1, …, n-1}
        i, j = sorted(random.sample(range(1, len(parent1)), 2))

        # Step 2: Child1[i+1:j] <- Parent1[i+1:j]
        child1 = [parent1[0]] + parent1[1:i + 1] + parent2[i + 1:j] + parent1[j:]

        # Step 3: For each element e2 in the segment of the second parent
        for idx in range(i + 1, j):
            e2 = parent2[idx]

            # Step 3.1: If e2 not in first child
            if e2 not in child1:
                e1_idx = parent1.index(e2)

                # Step 3.1.1: Get element e1 that occupies its position in the first child
                e1 = child1[e1_idx]

                # Step 3.1.2: Try to add e2 in the position occupied by e1 in the second child
                while e1 in parent2[i + 1:j]:
                    e1_idx = parent1.index(e1)
                    e1 = child1[e1_idx]

                child1[e1_idx] = e2

            else:
                # Step 3.1.3: Otherwise, if element e3 occupies the position in the child
                e3_idx = parent2.index(e2)

                # Step 3.1.3.1: Put e2 in the position that e3 has in the second parent
                child1[e3_idx] = e2

        # Step 4: The rest of the positions can be filled from the second parent
        for idx in range(len(child1)):
            if idx < i or idx >= j:
                if parent2[idx] not in child1:
                    child1[idx] = parent2[idx]

        # Repeat the process for the second child
        child2 = self.partially_mapped_crossover(parent2, parent1)

        return child1, child2
    
    # EDGE CROSSOVER
    def edge_crossover(self, parent1, parent2):
        # Step 1: For each value, build a table of adjacent values from both parents
        adjacency_table = defaultdict(set)

        for p1, p2 in zip(parent1, parent2):
            adjacency_table[p1].add(p2)
            adjacency_table[p2].add(p1)

        # Step 2: Child <- parent1[0]
        child = [parent1[0]]

        # Step 3: X <- Initial element from one of the two parents (random)
        X = random.choice(parent1[1:])

        # Step 4: While |Child| != n
        while len(child) < len(parent1):
            # Step 4.1: Child <- Child ∪ X
            child.append(X)

            # Step 4.2: Remove all references to X in the adjacency table
            del adjacency_table[X]

            common_adjacent_values = set()
            shortest_adjacent_list = None

            # Step 4.3: If X has a common adjacent value in both parents
            if X in parent1 and X in parent2:
                common_adjacent_values = adjacency_table[parent1[parent1.index(X)]].intersection(adjacency_table[parent2[parent2.index(X)]])

            # Step 4.3.1: X <- Common adjacent value in both parents
            if common_adjacent_values:
                X = min(common_adjacent_values, key=lambda val: len(adjacency_table[val]))

            # Step 4.4: Otherwise, if X has a non-empty adjacent list
            elif adjacency_table[X]:
                # Step 4.4.1: X <- Element from the adjacent list with the shortest adjacent list
                X = min(adjacency_table[X], key=lambda val: len(adjacency_table[val]))

            # Step 4.5: Otherwise, if the adjacent list is empty
            else:
                # Step 4.5.1: X <- Random value not in child
                remaining_values = set(parent1) - set(child)
                X = random.choice(list(remaining_values))

        return child
    
    ########################################

    def get_best_fitness(self):
        """
        Get the best fitness score found during the execution of the genetic algorithm.
        Returns:
            float: The best fitness score.
        """
        if self.best_solution is not None:
            return self.best_solution['fitness']
        else:
            raise ValueError("No best fitness found. Run the genetic algorithm first.")
    
    ########################################

    def early_stopping(self, current_fitness):
        """
        Early stopping method to check for fitness score improvement.
        The algorithm will stop if there is no improvement for 3 consecutive rounds.
        Args:
            current_fitness: The current fitness score.
        Returns:
            bool: True if early stopping criteria met, False otherwise.
        """
        if self.best_solution is None or current_fitness < self.best_solution['fitness']:
            # Update the best solution if the current solution is better
            self.best_solution = {'fitness': current_fitness, 'population': self.population.copy()}
            self.consecutive_rounds_no_improvement = 0
        else:
            # Increment the counter if there's no improvement
            self.consecutive_rounds_no_improvement += 1

        # Check if early stopping criteria met
        return self.consecutive_rounds_no_improvement >= 3

    ########################################

    def read_problem_instance(self, problem_path):
        with open(problem_path, 'r') as file:
            lines = file.readlines()

            # Extract the number of locations and vehicles from the txt
            num_locations = int(lines[0].split()[1])
            num_vehicles = int(lines[1].split()[1])

            # Initialize a matrix to store distances between locations
            distances = []
            for line in lines[3:]: #Skip the first three lines
                distances.append(list(map(int, line.split())))
        
        self.num_locations = num_locations
        self.num_vehicles = num_vehicles
        self.distances = distances
        
    ########################################

    #Generate a random route
    #4 vehicles, 10 locations: ['D1', 5, 9, 8, 'D2', 1, 10, 7, 'D3', 4, 2, 'D4', 6, 3]
    def generate_route(self):
        locations = list(range(1, self.num_locations + 1))
        random.shuffle(locations)
        routes = []

        remaining_locs = self.num_locations
        for i in range(self.num_vehicles):
            routes.append(f"D{i + 1}")

            if i == self.num_vehicles - 1:
                num_locs_per_vehicle = remaining_locs
            else:
                num_locs_per_vehicle = random.randint(1, remaining_locs - (self.num_vehicles - i - 1))

            vehicle_locs = locations[:num_locs_per_vehicle]
            routes.extend(vehicle_locs)
            locations = locations[num_locs_per_vehicle:]
            remaining_locs -= num_locs_per_vehicle

        return routes

    ########################################

    def initialize_population(self):
        population = [self.generate_route() for _ in range(self.population_size)]
        
        self.population = population
    
    ########################################

    def fitness(self, route):
        total_distance = 0
        locations_per_vehicle = len(route) // self.num_vehicles
    
        for i in range(self.num_vehicles):
            start_idx = i * locations_per_vehicle
            end_idx = start_idx + locations_per_vehicle if i < self.num_vehicles - 1 else len(route)
            vehicle_route = route[start_idx:end_idx]
    
            # Skip 'D' markers while calculating distances for each vehicle
            filtered_route = [loc for loc in vehicle_route if isinstance(loc, int)]
    
            vehicle_distance = 0
            for j in range(len(filtered_route) - 1):
                from_loc = filtered_route[j]
                to_loc = filtered_route[j + 1]
    
                # Fetch the distance from the matrix based on 0-indexed locations
                vehicle_distance += self.distances[from_loc - 1][to_loc - 1]  
    
            total_distance += vehicle_distance
    
        return total_distance  # Return total distance as the fitness value

    ########################################

    def get_best_solution(self):
        result = []
        current_group = []
        if self.best_solution is not None:
            for item in self.best_solution:
                if isinstance(item, str):
                    if current_group:
                        current_group.append(1)
                        result.append(current_group)
                        current_group = []
                else:
                    current_group.append(item)

            if current_group:
                current_group.append(1)
                result.append(current_group)

            return result
        else:
             raise ValueError("No best solution found. Run the genetic algorithm first.")

    ########################################

    def run(self):
        """
        Método que ejecuta el algoritmo genético. Debe crear la población inicial y
        ejecutar el bucle principal del algoritmo genético
        TODO: Se debe implementar aquí la lógica del algoritmo genético
        """
        self.read_problem_instance(self.problem_path)
        self.initialize_population()
        
        pass

    ########################################

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
        self.problem_path = problem_path
        self.best_solution = None #Atributo para guardar la mejor solución encontrada
        self.time_deadline = time_deadline # Límite de tiempo (en segundos) para el cómputo del algoritmo genético
        
        self.random_seed = kwargs.get('random_seed', 0)
        random.seed(self.random_seed)
        
        self.mutation_operator = kwargs.get('mutation_operator', 100)
        self.crossover_operator = kwargs.get('crossover_operator', 100)
        self.selection_method = kwargs.get('selection_method', 100)
        self.population_replacement_strategy = kwargs.get('population_replacement_strategy', 100)

        self.population_size = kwargs.get('population_size', 100)
        self.cross_rate = kwargs.get('cross_rate', 100)
        self.mut_rate = kwargs.get('mut_rate', 100)

    ########################################