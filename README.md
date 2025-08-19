# Optimizing Genetic Algorithms for the Multiple Traveling Salesman Problem

This repository contains the code and findings of a research project focused on solving the **Multiple Traveling Salesman Problem (MTSP)** by applying and optimizing a **Genetic Algorithm (GA)**.

The project was developed as part of the Bachelor's Degree in Data Science at the **Universitat Politècnica de València**.

[![Read the Report](https://img.shields.io/badge/Read_the_Full-Report-blue?style=for-the-badge&logo=adobeacrobatreader)](Optimization_project_2_report.pdf)

---

## 📝 Table of Contents

- [The Problem: Multiple Traveling Salesman Problem (MTSP)](#-the-problem-multiple-traveling-salesman-problem-mtsp)
- [Our Solution: A Genetic Algorithm Approach](#-our-solution-a-genetic-algorithm-approach)
- [Key Innovation: Random-Mixed Mutation](#-key-innovation-random-mixed-mutation)
- [Technical Stack & Methodologies](#-technical-stack--methodologies)
- [Algorithm Details](#-algorithm-details)
- [Systematic Optimization: A Two-Phase Grid Search](#-systematic-optimization-a-two-phase-grid-search)
- [Key Findings](#-key-findings)
- [Repository Structure](#-repository-structure)
- [How to Run the Experiments](#-how-to-run-the-experiments)
- [Future Work](#-future-work)
- [Authors](#-authors)

---

## 🎯 The Problem: Multiple Traveling Salesman Problem (MTSP)

The MTSP is a well-known extension of the classic Traveling Salesman Problem (TSP). Instead of a single agent, the MTSP involves a fleet of "salesmen" (or vehicles) who must start from a common depot to visit a set of cities.

The core constraints are:
- Each city must be visited exactly once by a single salesman.
- Every salesman starts from and returns to the same depot.
- The objective is to **minimize the total distance** traveled by the entire fleet.

This problem has countless real-world applications, from logistics and transportation planning to scheduling industrial tasks and routing drone fleets.

---

## 💡 Our Solution: A Genetic Algorithm Approach

To tackle the complexity of the MTSP, we designed and implemented a **Genetic Algorithm (GA)**. GAs are metaheuristics inspired by the process of natural selection and are particularly effective for complex optimization problems.

Our approach is grounded in an intensive experimental phase to identify the most performant combination of genetic operators and hyperparameters, aiming to effectively balance **exploration** (searching for new solutions) and **exploitation** (refining existing good solutions).

---

## 🚀 Key Innovation: Random-Mixed Mutation

The main innovation of this project is the introduction of a hybrid mutation strategy, which we call **Random-Mixed Mutation (RMM)**. Instead of relying on a single mutation operator, our algorithm leverages a combination of four well-established techniques, randomly choosing one at each mutation step:

- **Swap Mutation**: Swaps the positions of two genes (cities) in the chromosome.
- **Insertion Mutation**: Moves a gene to a new position.
- **Inversion Mutation**: Reverses the order of a subset of genes.
- **Scramble Mutation**: Randomly shuffles a subset of genes.

This strategy proved crucial for increasing the genetic diversity of the population and significantly improving the algorithm's performance by preventing premature convergence to local optima.

---

## 💻 Technical Stack & Methodologies

- **Language**: **Python 3.x**
- **Core Libraries**:
  - **NumPy**: For efficient numerical operations and handling of the distance matrix.
  - **Pandas**: For managing, processing, and analyzing the experimental results stored in CSV files.
  - **Matplotlib** & **Seaborn**: For data visualization and plotting the performance graphs included in the report.
  - **Jupyter Notebook**: For interactive analysis and exploration of the results.
- **Development Methodology**:
  - **Object-Oriented Programming (OOP)**: The Genetic Algorithm and its components (operators, population, etc.) are encapsulated in classes for modularity and reusability.
  - **Systematic Experimentation**: A rigorous, two-stage **Grid Search** was employed to test all operator and hyperparameter combinations.
  - **Statistical Ranking**: To compare the performance of different configurations, we used a ranking method inspired by the non-parametric statistical tests proposed by **Derrac et al. (2011)**.

---

## 🔧 Algorithm Details

The proposed GA is composed of the following core components:

- **Chromosome Representation**: A solution (a set of routes) is encoded as a single list of cities. A special symbol, `D`, represents the depot, marking the start and end of each salesman's tour.

- **Initial Population Generation**: The population is generated randomly but with a specific constraint to avoid creating empty routes (consecutive `D` symbols).

- **Selection Method**: **Fitness-Proportional Selection (Roulette Wheel)**. Parents are selected with a probability proportional to their fitness (which is inversely proportional to the total distance).

- **Crossover Operator**: **Cut-and-Crossfill Crossover (CCX)**. This operator generates offspring by combining segments from parent chromosomes while ensuring that each child remains a valid permutation (no duplicate cities).

- **Replacement Strategy**: **Lambda-Mu Replacement**. The new generation is formed by selecting the best `μ` individuals from a combined pool of the current population (`μ` individuals) and the newly generated offspring (`λ` individuals).

---

## 🔬 Systematic Optimization: A Two-Phase Grid Search

To find the optimal configuration, we conducted an exhaustive Grid Search process, testing a total of **775 combinations** across 8 different problem instances.

1.  **Phase 1: Operator Grid Search**: We tested hundreds of combinations of selection, crossover, mutation, and replacement operators to identify the most performant base architecture for the GA.

2.  **Phase 2: Hyperparameter Grid Search**: Once the best operator set was identified, we performed a second search to fine-tune the key hyperparameters:
    - **Population Size**: `[25, 50, 100, 200, 400]`
    - **Crossover Rate**: `[0.6, 0.7, 0.8, 0.9, 1.0]`
    - **Mutation Rate**: `[0.1, 0.2, 0.3, 0.4, 0.5]`
    - **Early-Stopping Limit**: `[25, 50, 100]` generations without improvement.

---

## 📈 Key Findings

Our experiments led to the following key conclusions:

- **Optimal Configuration**: The most effective combination of operators was found to be: **Fitness Proportional Selection (RWS)**, **Cut-and-Crossfill Crossover (CCX)**, **Random-Mixed Mutation (RMM)**, and **Lambda-Mu Replacement (LMR)**.

- **Impact of Population and Mutation**: The algorithm's performance significantly improves with a **very large initial population** (400 individuals) and a **high mutation rate** (0.5). This highlights the importance of a strong exploratory drive to effectively solve the MTSP.

- **Superior Replacement Strategy**: The **Lambda-Mu (LMR)** strategy proved to be markedly superior, demonstrating robust and consistent behavior when combined with all tested mutation operators.

---

## 📂 Repository Structure

```

.
├── GA.py                                \# Main implementation of the Genetic Algorithm
├── experiments.py                       \# Script to launch the Grid Search experiments
├── instances/                           \# Directory with the 8 MTSP problem instances
│   ├── instance01.txt
│   └── ...
├── results/                             \# CSV files with the raw results from the experiments
│   ├── results\_operators.csv
│   └── results\_hyperparameters.csv
├── results\_analysis\_operators.ipynb     \# Jupyter Notebook for analyzing operator results
├── results\_analysis\_hyperparameters.ipynb \# Jupyter Notebook for analyzing hyperparameter results
├── Optimization\_project\_2\_report.pdf    \# The final, detailed project report
├── figures/                             \# Figures and plots generated from the analysis
└── README.md                            \# This file

````

---

## 🚀 How to Run the Experiments

To replicate the experiments, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/danieleborghe/optimization-project-2-upv.git](https://github.com/danieleborghe/optimization-project-2-upv.git)
    cd optimization-project-2-upv
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install numpy pandas matplotlib seaborn
    ```

4.  **Run the experiment script:**
    The `experiments.py` script is configured to run the Grid Search. You can modify the parameters within the script to test specific configurations.
    ```bash
    python experiments.py
    ```
    The results will be saved as CSV files in the `results/` directory.

---

## 🔮 Future Work

Despite the promising results, there are several areas for future improvement:

- **Smarter Initialization**: Implement more sophisticated initialization strategies (e.g., based on the "Nearest Neighbor" heuristic) to improve the quality of the initial population.
- **Adaptive Operator Selection**: Develop mechanisms that allow the algorithm to dynamically adapt the choice of operators or their parameters during runtime.
- **Expanded Hyperparameter Search**: Test even larger values for population size (>400) and mutation rate (>0.5) to investigate the potential for further performance gains.

---

## 👥 Authors

- **Eva Cantín Larumbe**
- **Eva Teisberg**
- **Mikel Baraza Vidal**
- **Francesco Pio Capoccello**
- **Daniele Borghesi**
