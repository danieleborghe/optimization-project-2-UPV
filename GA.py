import random
from collections import defaultdict
import time

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

        return selected_parents

    def linear_ranking_selection(self, s=1.5):
        sorted_population = sorted(self.population, key=lambda x: self.fitness(x), reverse=True)
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
        sorted_population = sorted(self.population, key=lambda x: self.fitness(x), reverse=True)
        selection_probs = []

        for i in range(self.population_size):
            prob = max(0, (c - 1) / ((c ** self.population_size) - 1) * (c ** (self.population_size - i - 1)))
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
                best_individual = max(tournament, key=lambda x: self.fitness(x))  # Select the best individual
                selected_parents.append(best_individual)
            else:
                selected_parents.append(random.choice(tournament))  # Randomly select from the tournament

        return selected_parents

    def uniform_selection(self):
        selected_parents = random.choices(self.population, k=self.population_size)
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
            parent1, parent2 = parents[i], parents[i + 1]
            # Step 1: Select random position i ∈ {1, …, n-2}
            crossover_point = random.randint(1, len(parent1) - 2)

            # Ensure the first element of the parents remains the first element of the children
            child1 = [parent1[0]] + parent1[1:crossover_point] + parent2[crossover_point:]
            child2 = [parent2[0]] + parent2[1:crossover_point] + parent1[crossover_point:]

            # Step 3: Scan from the second segment of parent 2 and fill with values not present in child 1
            for gene in parent2[crossover_point:] + parent2[1:crossover_point]:
                if gene not in child1:
                    child1.append(gene)

            # Step 4: Do the same for parent 1 and child 2
            for gene in parent1[crossover_point:] + parent1[1:crossover_point]:
                if gene not in child2:
                    child2.append(gene)

            offspring.extend([child1, child2])

        return offspring

    # PARTIALLY MAPPED CROSSOVER (PMX)
    def partially_mapped_crossover(self, selected_parents):
        offspring = []

        for parent1, parent2 in zip(selected_parents[::2], selected_parents[1::2]):
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
            child2 = [parent2[0]] + parent2[1:i + 1] + parent1[i + 1:j] + parent2[j:]

            for idx in range(i + 1, j):
                e2 = parent1[idx]
                if e2 not in child2:
                    e3_idx = parent2.index(e2)
                    e3 = child2[e3_idx]
                    while e3 in parent1[i + 1:j]:
                        e3_idx = parent2.index(e3)
                        e3 = child2[e3_idx]
                    child2[e3_idx] = e2
                else:
                    e1_idx = parent1.index(e2)
                    child2[e1_idx] = e2

            for idx in range(len(child2)):
                if idx < i or idx >= j:
                    if parent1[idx] not in child2:
                        child2[idx] = parent1[idx]

            offspring.extend([child1, child2])

        return offspring 
    
    # EDGE CROSSOVER
    def edge_crossover(self, selected_parents):
        offspring = []

        for parent1, parent2 in zip(selected_parents[::2], selected_parents[1::2]):
            # Step 1: For each value, build a table of adjacent values from both parents
            adjacency_table = defaultdict(set)

            for p1, p2 in zip(parent1, parent2):
                adjacency_table[p1].add(p2)
                adjacency_table[p2].add(p1)

            # Step 2: Child <- parent1[0]
            child1 = [parent1[0]]
            child2 = [parent2[0]]

            # Step 3: X <- Initial element from one of the two parents (random)
            remaining_values = set(parent1[1:])
            X = random.choice(list(remaining_values))

            # Step 4: While |Child1| != n
            while len(child1) < len(parent1):
                # Step 4.1: Child1 <- Child1 ∪ X
                child1.append(X)
                child2.append(X)

                # Step 4.2: Remove all references to X in the adjacency table
                del adjacency_table[X]

                common_adjacent_values = set()

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
                    # Step 4.5.1: X <- Random value not in child1
                    remaining_values = set(parent1) - set(child1)
                    if remaining_values:
                        X = random.choice(list(remaining_values))

            offspring.extend([child1, child2])

        return offspring

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
    def lambda_mu_replacement(self, parents, offspring, mu=5):
        combined_population = parents + offspring
        combined_population.sort(key=self.fitness)
        new_population = combined_population[:mu]

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
            for line in lines[3:]: #Skip the first three lines
                distance_matrix.append(list(map(int, line.split())))
        
        self.num_locations = num_locations
        self.num_vehicles = num_vehicles
        self.distance_matrix = distance_matrix
    
    ########################################
        
    #Generate a random route
    #4 vehicles, 10 locations: ['D1', 5, 9, 8, 'D2', 7, 'D3', 4, 2, 'D4', 6, 3] 
    #OBS! Note that we only vist 9 locations since the first one is the depot
    def generate_route(self):
        # generate locations list
        locations = list(range(1, self.num_locations))
        # shuffle locations
        random.shuffle(locations)
        # generate an empty list of routes
        routes = []

        # create a counter for the remaining locations
        remaining_locs = self.num_locations
        # iterate all vehicles
        for i in range(self.num_vehicles):
            # appen the depot (starting point) for each vehicle
            routes.append(f"D{i + 1}")

            # if we are at the last vehicle add all the remaining locations to it
            if i == self.num_vehicles - 1:
                num_locs_per_vehicle = remaining_locs
            else:
                num_locs_per_vehicle = random.randint(1, remaining_locs - (self.num_vehicles - i))

            # select the locations for that vehicle's route
            vehicle_locs = locations[:num_locs_per_vehicle]
            # add the route to the routes
            routes.extend(vehicle_locs)
            # update locations
            locations = locations[num_locs_per_vehicle:]
            # update number of remaining locations
            remaining_locs -= num_locs_per_vehicle

        return routes

    ########################################

    def initialize_population(self):
        population = [self.generate_route() for _ in range(self.population_size)]
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
        best_route = min(self.population, key=self.fitness)
        best_fitness = self.fitness(best_route)

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

        while time.time() - start_time < self.time_deadline and consecutive_no_improvement < 3:
            # Selection: Choose tours from the current population to act as parents for the next generation
            selected_parents = self.tournament_selection()

            # Crossover: Create new offspring from two parents.
            offspring = self.cut_and_crossfill_crossover(selected_parents)

            # Mutation: Make small changes to the offspring
            offspring = self.swap_mutation(offspring)

            # Replacement: Create new population from the current population and the offspring
            self.population = self.replace_worst_replacement(self.population, offspring)

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

        # Print the best solution found
        best_solution = self.get_best_solution()
        best_fitness = self.get_best_fitness()

        print(f"Best Solution (Generation {current_generation}):")
        print(f"Route: {best_solution}")
        print(f"Total Distance: {best_fitness}")

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
        self.population = None
        self.population_size = kwargs.get('population_size', 100)
        self.num_locations = None
        self.num_vehicles = None
        self.distance_matrix = None

        self.random_seed = kwargs.get('random_seed', 0)
        random.seed(self.random_seed)
        
        self.mutation_operator = kwargs.get('mutation_operator', 100)
        self.crossover_operator = kwargs.get('crossover_operator', 100)
        self.selection_method = kwargs.get('selection_method', 100)
        self.population_replacement_strategy = kwargs.get('population_replacement_strategy', 100)

        self.cross_rate = kwargs.get('cross_rate', 100)
        self.mut_rate = kwargs.get('mut_rate', 100)

########################################
        
ga_instance = GA(time_deadline=5, problem_path='instances/instance02.txt', population_size=100)
ga_instance.run()

