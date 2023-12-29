import random
from collections import defaultdict

class GA:

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
    def cut_and_crossfill_crossover(parent1, parent2):
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
    def partially_mapped_crossover(parent1, parent2):
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
        child2 = partially_mapped_crossover(parent2, parent1)

        return child1, child2
    
    # EDGE CROSSOVER
    def edge_crossover(parent1, parent2):
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

    def read_problem_instance(self, problem_path):
        """
        TODO: Implementar método para leer una instancia del problema
        y ajustar los atributos internos del objeto necesarios
        """
        pass

    ########################################

    def get_best_solution(self):
        """
        Método para devolver la mejor solución encontrada hasta
        el momento
        """
        #TODO
        pass

    ########################################

    def run(self):
        """
        Método que ejecuta el algoritmo genético. Debe crear la población inicial y
        ejecutar el bucle principal del algoritmo genético
        TODO: Se debe implementar aquí la lógica del algoritmo genético
        """
        self.read_problem_instance(self.problem_path)
        pass

    ########################################

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
        #TODO: Completar método para configurar el algoritmo genético (e.g., seleccionar cruce, mutación, etc.)

    ########################################


    

