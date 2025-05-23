# Code and Class Structure Documentation

This document provides a detailed explanation of the code organization and class structure used in the Fantasy League Team Optimization project.

## 1. Project Architecture Overview

The Fantasy League Team Optimization project follows a modular architecture with clear separation of concerns:

```
CIFO-24-25/
├── Core Components
│   ├── solution.py           # Solution representation
│   ├── evolution.py          # Evolutionary algorithms
│   ├── operators.py          # Genetic operators
│   ├── fitness_counter.py    # Fitness evaluation counter
│
├── Execution and Analysis
│   ├── CIFO_Complete_Pipeline_Final.py    # Complete pipeline
│   └── players.csv                        # Player dataset
│
└── Documentation
    ├── README.md                       # Project overview
    ├── OPERATORS_DOCUMENTATION.md      # Operators documentation
    ├── REPRESENTATION_DOCUMENTATION.md # Representation documentation
    └── CODE_STRUCTURE_DOCUMENTATION.md # This document
```

## 2. Core Components

### 2.1 Solution Module (`solution.py`)

The `solution.py` module defines the solution representation for the Fantasy League Team Optimization problem.

#### 2.1.1 `LeagueSolution` Class

```python
class LeagueSolution:
    def __init__(self, players, repr=None, num_teams=5, team_size=7, max_budget=750):
        self.players = players
        self.num_teams = num_teams
        self.team_size = team_size
        self.max_budget = max_budget
        self.fitness_counter = None
        
        # Initialize representation
        if repr is None:
            # Random initialization
            self.repr = [random.randint(0, num_teams - 1) for _ in range(len(players))]
        else:
            # Use provided representation
            self.repr = repr
```

**Responsibilities**:
- Represent a solution to the Fantasy League Team Optimization problem
- Provide methods for solution evaluation and manipulation
- Calculate fitness (standard deviation of average skills)
- Check solution validity (position and budget constraints)
- Extract team compositions from the representation

**Key Methods**:
- `fitness()`: Calculate fitness of the solution
- `is_valid()`: Check if the solution respects all constraints
- `get_teams()`: Extract teams from the representation
- `get_team_stats()`: Calculate statistics for each team
- `set_fitness_counter()`: Set fitness counter for tracking evaluations

### 2.2 Evolution Module (`evolution.py`)

The `evolution.py` module implements various evolutionary algorithms for optimization.

#### 2.2.1 `GeneticAlgorithm` Class

```python
class GeneticAlgorithm:
    def __init__(self, selection_operator, crossover_operator, mutation_operator, 
                 population_size=100, elitism_rate=0.1, mutation_rate=None):
        self.selection_operator = selection_operator
        self.crossover_operator = crossover_operator
        self.mutation_operator = mutation_operator
        self.population_size = population_size
        self.elitism_rate = elitism_rate
        self.mutation_rate = mutation_rate
```

**Responsibilities**:
- Implement the genetic algorithm framework
- Manage population evolution over generations
- Apply selection, crossover, and mutation operators
- Track best solution and convergence history

**Key Methods**:
- `initialize_population()`: Create initial population
- `evolve()`: Evolve population for one generation
- `run()`: Run the genetic algorithm for multiple generations

#### 2.2.2 `HillClimbing` Class

```python
class HillClimbing:
    def __init__(self, max_iterations=1000):
        self.max_iterations = max_iterations
```

**Responsibilities**:
- Implement the hill climbing algorithm
- Generate and evaluate neighbors
- Accept improving moves
- Track best solution and convergence history

**Key Methods**:
- `run()`: Run the hill climbing algorithm

#### 2.2.3 `SimulatedAnnealing` Class

```python
class SimulatedAnnealing:
    def __init__(self, initial_temperature=100, cooling_rate=0.95, max_iterations=1000):
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
```

**Responsibilities**:
- Implement the simulated annealing algorithm
- Generate and evaluate neighbors
- Accept moves based on temperature and fitness difference
- Track best solution and convergence history

**Key Methods**:
- `run()`: Run the simulated annealing algorithm

### 2.3 Operators Module (`operators.py`)

The `operators.py` module defines genetic operators for selection, crossover, and mutation.

#### 2.3.1 Selection Operators

```python
def selection_tournament(fitness_values, tournament_size=3):
    """Tournament selection: randomly select tournament_size individuals and choose the best."""
    # Implementation details...

def selection_ranking(fitness_values):
    """Rank-based selection: selection probability based on rank rather than absolute fitness."""
    # Implementation details...

def selection_boltzmann(fitness_values, temperature=1.0):
    """Boltzmann selection: selection probability based on Boltzmann distribution."""
    # Implementation details...
```

**Responsibilities**:
- Implement various selection methods
- Select individuals from the population based on fitness
- Control selection pressure

#### 2.3.2 Crossover Operators

```python
def crossover_one_point(parent1, parent2):
    """One-point crossover: creates a child by taking portions from each parent."""
    # Implementation details...

def crossover_two_point(parent1, parent2):
    """Two-point crossover: creates a child by taking portions from each parent."""
    # Implementation details...

def crossover_uniform(parent1, parent2, swap_probability=0.5):
    """Uniform crossover: creates a child by randomly selecting genes from either parent."""
    # Implementation details...
```

**Responsibilities**:
- Implement various crossover methods
- Combine genetic material from two parents
- Create offspring solutions

#### 2.3.3 Mutation Operators

```python
def mutate_swap(solution, mutation_rate=None):
    """Swap mutation: randomly changes team assignments."""
    # Implementation details...

def mutate_scramble(solution, mutation_rate=0.1):
    """Scramble mutation: randomly selects a subsequence and shuffles it."""
    # Implementation details...
```

**Responsibilities**:
- Implement various mutation methods
- Introduce small random changes to solutions
- Maintain genetic diversity

### 2.4 Fitness Counter Module (`fitness_counter.py`)

The `fitness_counter.py` module implements a counter for tracking fitness evaluations.

#### 2.4.1 `FitnessCounter` Class

```python
class FitnessCounter:
    def __init__(self):
        self.count = 0
    
    def increment(self):
        self.count += 1
    
    def get_count(self):
        return self.count
    
    def reset(self):
        self.count = 0
```

**Responsibilities**:
- Track the number of fitness evaluations
- Provide methods for incrementing, getting, and resetting the count

## 3. Execution and Analysis

### 3.1 Complete Pipeline (`CIFO_Complete_Pipeline_Final.py`)

The `CIFO_Complete_Pipeline_Final.py` file integrates all components into a complete pipeline for execution and analysis.

#### 3.1.1 Configuration

```python
EXPERIMENT_CONFIG = {
    # General parameters
    'seed': 42,                    # Seed for reproducibility
    'num_runs': 30,                # Number of runs for each algorithm
    'max_evaluations': 10000,      # Maximum number of function evaluations
    'population_size': 100,        # Population size for genetic algorithms
    'max_generations': 100,        # Maximum number of generations for genetic algorithms
    
    # Execution parameters
    'execution_mode': ExecutionMode.MULTI_PROCESSOR,  # Parallel execution by default
    'num_processes': max(1, multiprocessing.cpu_count() - 1),  # Use all but one CPU core
    
    # Statistical analysis parameters
    'alpha': 0.05,                 # Significance level for statistical tests
    'post_hoc_method': 'tukey',    # Post-hoc test method ('tukey' or 'dunn')
    
    # Visualization parameters
    'figure_size': (14, 10),       # Default figure size
    'save_figures': False,         # Save figures to files
    'figure_format': 'png',        # Format for saving figures
    
    # Data storage parameters
    'save_results': True,          # Save results to files
    'save_history': True,          # Save convergence history
    'save_statistics': True,       # Save statistical test results
    'results_dir': 'experiment_results',  # Directory for storing results
    
    # Execution parameters
    'verbose': True,               # Show detailed progress
    'load_existing': False,        # Load existing results (if available)
}
```

**Responsibilities**:
- Define configuration parameters for experiments
- Control execution mode (sequential or parallel)
- Configure statistical analysis parameters
- Set visualization and data storage options

#### 3.1.2 Algorithm Configurations

```python
configs = {
    "HC_Standard": {
        "algorithm": "hill_climbing",
        "max_iterations": 10000,
    },
    "SA_Standard": {
        "algorithm": "simulated_annealing",
        "initial_temperature": 100,
        "cooling_rate": 0.95,
        "max_iterations": 10000,
    },
    "GA_Tournament_OnePoint": {
        "algorithm": "genetic_algorithm",
        "selection": "tournament",
        "crossover": "one_point",
        "mutation": "swap",
        "population_size": 100,
        "elitism_rate": 0.1,
        "tournament_size": 3,
        "max_generations": 100,
    },
    # More configurations...
}
```

**Responsibilities**:
- Define algorithm configurations for experiments
- Specify algorithm type and parameters
- Enable comparison of different algorithms and parameter settings

#### 3.1.3 Execution Functions

```python
def run_experiment(config, players_list, max_evaluations=10000):
    """Run a single experiment with the specified configuration."""
    # Implementation details...

def run_multiple_experiments(configs, players_list, num_runs=30, max_evaluations=10000):
    """Run multiple experiments for each configuration."""
    # Implementation details...

def run_parallel_experiments(configs, players_list, num_runs=30, max_evaluations=10000, num_processes=None):
    """Run multiple experiments in parallel."""
    # Implementation details...
```

**Responsibilities**:
- Execute experiments with specified configurations
- Run multiple repetitions for statistical significance
- Support parallel execution for faster results
- Track and store results and convergence history

#### 3.1.4 Analysis Functions

```python
def analyze_results(results_df):
    """Analyze experiment results and perform statistical tests."""
    # Implementation details...

def plot_performance_comparison(results_df):
    """Plot performance comparison across all configurations."""
    # Implementation details...

def plot_convergence_curves(history_data, title="Convergence Curves by Run"):
    """Plot convergence curves for all configurations."""
    # Implementation details...
```

**Responsibilities**:
- Analyze experiment results
- Perform statistical tests for algorithm comparison
- Create visualizations of performance and convergence
- Display the optimal team solution

## 4. Class Relationships and Interactions

### 4.1 Class Diagram

```
+-------------------+     +-------------------+     +-------------------+
|  LeagueSolution   |     |  FitnessCounter   |     |  GeneticAlgorithm |
+-------------------+     +-------------------+     +-------------------+
| - players         |     | - count           |     | - selection_op    |
| - repr            |     +-------------------+     | - crossover_op    |
| - num_teams       |     | + increment()     |     | - mutation_op     |
| - team_size       |     | + get_count()     |     +-------------------+
| - max_budget      |     | + reset()         |     | + initialize_pop()|
| - fitness_counter |     +-------------------+     | + evolve()        |
+-------------------+           ^                   | + run()           |
| + fitness()       |           |                   +-------------------+
| + is_valid()      |           |                           ^
| + get_teams()     |           |                           |
| + get_team_stats()|           |                   +-------+-------+
+-------------------+           |                   |               |
        ^                       |                   |               |
        |                       |                   v               v
+-------+-------+     +---------+---------+    +-------+     +------------+
| HillClimbing  |     | SimulatedAnnealing|    |  GA1  |     |    GA2     |
+---------------+     +-------------------+    +-------+     +------------+
| + run()       |     | + run()           |
+---------------+     +-------------------+
```

### 4.2 Key Interactions

1. **Solution and Fitness Counter**:
   - `LeagueSolution` uses `FitnessCounter` to track fitness evaluations
   - `set_fitness_counter()` method establishes this relationship

2. **Algorithms and Solution**:
   - All algorithms (`GeneticAlgorithm`, `HillClimbing`, `SimulatedAnnealing`) operate on `LeagueSolution` objects
   - Algorithms create, evaluate, and modify solutions during optimization

3. **Genetic Algorithm and Operators**:
   - `GeneticAlgorithm` uses selection, crossover, and mutation operators from `operators.py`
   - Operators are passed as parameters to the `GeneticAlgorithm` constructor

4. **Pipeline and Components**:
   - `CIFO_Complete_Pipeline_Final.py` integrates all components
   - Creates and configures algorithm instances
   - Executes experiments and analyzes results

## 5. Main Workflow

### 5.1 Execution Workflow

1. **Configuration**:
   - Define experiment configuration (`EXPERIMENT_CONFIG`)
   - Define algorithm configurations (`configs`)

2. **Data Loading**:
   - Load player data from CSV file
   - Create player list for solution initialization

3. **Experiment Execution**:
   - For each algorithm configuration:
     - For each run (1 to `num_runs`):
       - Initialize fitness counter
       - Create initial solution
       - Run optimization algorithm
       - Record best solution, fitness, and evaluations
       - Track convergence history

4. **Results Storage**:
   - Save results to CSV file
   - Save convergence history to NPY file

### 5.2 Analysis Workflow

1. **Data Loading**:
   - Load results from CSV file
   - Load convergence history from NPY file

2. **Statistical Analysis**:
   - Check normality of data (Shapiro-Wilk test)
   - Check homogeneity of variances (Levene's test)
   - Perform ANOVA or Kruskal-Wallis test
   - Perform post-hoc tests (Tukey HSD or Dunn's test)

3. **Visualization**:
   - Plot performance comparison (bar charts)
   - Plot convergence curves (line charts)
   - Plot statistical significance (boxplots)

4. **Solution Display**:
   - Display the optimal team solution
   - Show team compositions and statistics

## 6. Algorithm Implementation Details

### 6.1 Hill Climbing

```python
def run_hill_climbing(solution, max_iterations=1000):
    """
    Run hill climbing algorithm.
    
    Args:
        solution: Initial solution
        max_iterations: Maximum number of iterations
        
    Returns:
        tuple: (best_solution, best_fitness, fitness_history)
    """
    best_solution = deepcopy(solution)
    best_fitness = solution.fitness()
    
    fitness_history = [best_fitness]
    
    for i in range(max_iterations):
        # Generate neighbor
        neighbor = deepcopy(best_solution)
        idx = random.randint(0, len(neighbor.repr) - 1)
        neighbor.repr[idx] = random.randint(0, neighbor.num_teams - 1)
        
        neighbor_fitness = neighbor.fitness()
        
        # Accept if better
        if neighbor_fitness < best_fitness:
            best_solution = neighbor
            best_fitness = neighbor_fitness
            fitness_history.append(best_fitness)
        else:
            fitness_history.append(best_fitness)
    
    return best_solution, best_fitness, fitness_history
```

### 6.2 Simulated Annealing

```python
def run_simulated_annealing(solution, initial_temperature=100, cooling_rate=0.95, max_iterations=1000):
    """
    Run simulated annealing algorithm.
    
    Args:
        solution: Initial solution
        initial_temperature: Initial temperature
        cooling_rate: Cooling rate
        max_iterations: Maximum number of iterations
        
    Returns:
        tuple: (best_solution, best_fitness, fitness_history)
    """
    current_solution = deepcopy(solution)
    current_fitness = solution.fitness()
    
    best_solution = deepcopy(current_solution)
    best_fitness = current_fitness
    
    temperature = initial_temperature
    fitness_history = [current_fitness]
    
    for i in range(max_iterations):
        # Generate neighbor
        neighbor = deepcopy(current_solution)
        idx = random.randint(0, len(neighbor.repr) - 1)
        neighbor.repr[idx] = random.randint(0, neighbor.num_teams - 1)
        
        neighbor_fitness = neighbor.fitness()
        
        # Calculate acceptance probability
        if neighbor_fitness < current_fitness:
            acceptance_probability = 1.0
        else:
            acceptance_probability = math.exp((current_fitness - neighbor_fitness) / temperature)
        
        # Accept or reject
        if random.random() < acceptance_probability:
            current_solution = neighbor
            current_fitness = neighbor_fitness
            
            # Update best solution
            if current_fitness < best_fitness:
                best_solution = deepcopy(current_solution)
                best_fitness = current_fitness
        
        # Cool down
        temperature *= cooling_rate
        
        # Record history
        fitness_history.append(best_fitness)
    
    return best_solution, best_fitness, fitness_history
```

### 6.3 Genetic Algorithm

```python
def run_genetic_algorithm(solution_class, players, selection_func, crossover_func, mutation_func, 
                         population_size=100, elitism_rate=0.1, tournament_size=3, 
                         max_generations=100, mutation_rate=None):
    """
    Run genetic algorithm.
    
    Args:
        solution_class: Solution class
        players: List of player dictionaries
        selection_func: Selection function
        crossover_func: Crossover function
        mutation_func: Mutation function
        population_size: Population size
        elitism_rate: Elitism rate
        tournament_size: Tournament size (for tournament selection)
        max_generations: Maximum number of generations
        mutation_rate: Mutation rate
        
    Returns:
        tuple: (best_solution, best_fitness, fitness_history)
    """
    # Initialize population
    population = []
    for _ in range(population_size):
        solution = solution_class(players=players)
        population.append(solution)
    
    # Evaluate initial population
    fitness_values = [solution.fitness() for solution in population]
    
    # Find best solution
    best_idx = fitness_values.index(min(fitness_values))
    best_solution = deepcopy(population[best_idx])
    best_fitness = fitness_values[best_idx]
    
    fitness_history = [best_fitness]
    
    # Evolution loop
    for generation in range(max_generations):
        # Elitism: keep best individuals
        elitism_count = int(population_size * elitism_rate)
        elite_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])[:elitism_count]
        elite = [deepcopy(population[i]) for i in elite_indices]
        
        # Create new population
        new_population = []
        
        # Add elite individuals
        new_population.extend(elite)
        
        # Fill the rest with offspring
        while len(new_population) < population_size:
            # Selection
            if selection_func == selection_tournament:
                parent1_idx = selection_func(fitness_values, tournament_size)
                parent2_idx = selection_func(fitness_values, tournament_size)
            else:
                parent1_idx = selection_func(fitness_values)
                parent2_idx = selection_func(fitness_values)
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover
            child = crossover_func(parent1, parent2)
            
            # Mutation
            child = mutation_func(child, mutation_rate)
            
            new_population.append(child)
        
        # Replace population
        population = new_population
        
        # Evaluate new population
        fitness_values = [solution.fitness() for solution in population]
        
        # Update best solution
        current_best_idx = fitness_values.index(min(fitness_values))
        current_best_fitness = fitness_values[current_best_idx]
        
        if current_best_fitness < best_fitness:
            best_solution = deepcopy(population[current_best_idx])
            best_fitness = current_best_fitness
        
        # Record history
        fitness_history.append(best_fitness)
    
    return best_solution, best_fitness, fitness_history
```

## 7. Advanced Algorithm Implementations

### 7.1 GA Memetic

```python
def run_memetic_algorithm(solution_class, players, selection_func, crossover_func, mutation_func, 
                         population_size=100, elitism_rate=0.1, tournament_size=3, 
                         max_generations=100, mutation_rate=None, 
                         local_search_prob=0.1, local_search_iterations=10):
    """
    Run memetic algorithm (GA with local search).
    
    Args:
        solution_class: Solution class
        players: List of player dictionaries
        selection_func: Selection function
        crossover_func: Crossover function
        mutation_func: Mutation function
        population_size: Population size
        elitism_rate: Elitism rate
        tournament_size: Tournament size (for tournament selection)
        max_generations: Maximum number of generations
        mutation_rate: Mutation rate
        local_search_prob: Probability of applying local search
        local_search_iterations: Number of local search iterations
        
    Returns:
        tuple: (best_solution, best_fitness, fitness_history)
    """
    # Similar to run_genetic_algorithm, but with local search
    # ...
    
    # Local search function
    def local_search(solution, iterations):
        """
        Local search to improve a solution through hill climbing.
        
        Args:
            solution: Solution to improve
            iterations: Number of local search iterations
            
        Returns:
            Improved solution
        """
        best_sol = deepcopy(solution)
        best_fitness = solution.fitness()
        
        for _ in range(iterations):
            # Generate neighbor
            neighbor = deepcopy(best_sol)
            idx = random.randint(0, len(neighbor.repr) - 1)
            neighbor.repr[idx] = random.randint(0, neighbor.num_teams - 1)
            
            neighbor_fitness = neighbor.fitness()
            
            # Accept if better
            if neighbor_fitness < best_fitness:
                best_sol = neighbor
                best_fitness = neighbor_fitness
        
        return best_sol
    
    # Apply local search to individuals with probability local_search_prob
    # ...
    
    return best_solution, best_fitness, fitness_history
```

### 7.2 GA Island Model

```python
def run_island_model_ga(solution_class, players, selection_func, crossover_func, mutation_func, 
                       population_size=100, elitism_rate=0.1, tournament_size=3, 
                       max_generations=100, mutation_rate=None, 
                       num_islands=5, migration_interval=10, migration_size=5):
    """
    Run island model genetic algorithm.
    
    Args:
        solution_class: Solution class
        players: List of player dictionaries
        selection_func: Selection function
        crossover_func: Crossover function
        mutation_func: Mutation function
        population_size: Population size
        elitism_rate: Elitism rate
        tournament_size: Tournament size (for tournament selection)
        max_generations: Maximum number of generations
        mutation_rate: Mutation rate
        num_islands: Number of islands
        migration_interval: Number of generations between migrations
        migration_size: Number of individuals to migrate
        
    Returns:
        tuple: (best_solution, best_fitness, fitness_history)
    """
    # Initialize islands
    islands = []
    island_fitness = []
    
    for _ in range(num_islands):
        # Initialize population for each island
        island = []
        for _ in range(population_size // num_islands):
            solution = solution_class(players=players)
            island.append(solution)
        
        islands.append(island)
        island_fitness.append([solution.fitness() for solution in island])
    
    # Find best solution across all islands
    best_solution = None
    best_fitness = float('inf')
    
    for i in range(num_islands):
        island_best_idx = island_fitness[i].index(min(island_fitness[i]))
        if island_fitness[i][island_best_idx] < best_fitness:
            best_solution = deepcopy(islands[i][island_best_idx])
            best_fitness = island_fitness[i][island_best_idx]
    
    fitness_history = [best_fitness]
    
    # Evolution loop
    for generation in range(max_generations):
        # Evolve each island independently
        for i in range(num_islands):
            # Evolution logic similar to run_genetic_algorithm
            # ...
        
        # Migration between islands
        if generation % migration_interval == 0 and generation > 0:
            for i in range(num_islands):
                # Select migrants (best individuals)
                migrant_indices = np.argsort(island_fitness[i])[:migration_size]
                migrants = [deepcopy(islands[i][idx]) for idx in migrant_indices]
                
                # Send to next island (ring topology)
                next_island = (i + 1) % num_islands
                
                # Replace worst individuals in next island
                worst_indices = np.argsort(island_fitness[next_island])[-migration_size:]
                for j, idx in enumerate(worst_indices):
                    islands[next_island][idx] = migrants[j]
                
                # Re-evaluate fitness of next island
                island_fitness[next_island] = [solution.fitness() for solution in islands[next_island]]
        
        # Update best solution across all islands
        for i in range(num_islands):
            island_best_idx = island_fitness[i].index(min(island_fitness[i]))
            if island_fitness[i][island_best_idx] < best_fitness:
                best_solution = deepcopy(islands[i][island_best_idx])
                best_fitness = island_fitness[i][island_best_idx]
        
        # Record history
        fitness_history.append(best_fitness)
    
    return best_solution, best_fitness, fitness_history
```

## 8. Parallel Execution

### 8.1 Execution Mode Enum

```python
class ExecutionMode(Enum):
    SINGLE_PROCESSOR = 1
    MULTI_PROCESSOR = 2
```

### 8.2 Parallel Execution Implementation

```python
def run_parallel_experiments(configs, players_list, num_runs=30, max_evaluations=10000, num_processes=None):
    """
    Run multiple experiments in parallel.
    
    Args:
        configs: Dictionary of algorithm configurations
        players_list: List of player dictionaries
        num_runs: Number of runs for each configuration
        max_evaluations: Maximum number of function evaluations
        num_processes: Number of processes to use
        
    Returns:
        tuple: (results_df, history_data)
    """
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    # Create experiment tasks
    tasks = []
    for config_name, config in configs.items():
        for run in range(num_runs):
            tasks.append((config_name, config, players_list, max_evaluations, run))
    
    # Run experiments in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(run_single_experiment, tasks)
    
    # Process results
    all_results = []
    history_data = {}
    
    for config_name, run, best_fitness, evaluations, runtime, history in results:
        all_results.append({
            'Configuration': config_name,
            'Run': run,
            'Best Fitness': best_fitness,
            'Evaluations': evaluations,
            'Runtime (s)': runtime
        })
        
        # Store history data
        if config_name not in history_data:
            history_data[config_name] = {}
        
        history_data[config_name][run] = history
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    return results_df, history_data
```

## 9. Data Storage and Loading

### 9.1 Saving Results

```python
def save_results(results_df, history_data, experiment_dir):
    """
    Save experiment results to files.
    
    Args:
        results_df: DataFrame with experiment results
        history_data: Dictionary with convergence history
        experiment_dir: Directory to save results
    """
    # Create directory if it doesn't exist
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save results CSV
    results_path = os.path.join(experiment_dir, "results.csv")
    results_df.to_csv(results_path, index=False)
    
    # Save history data
    history_path = os.path.join(experiment_dir, "history_data.npy")
    np.save(history_path, history_data)
    
    print(f"Results saved to: {experiment_dir}")
```

### 9.2 Loading Results

```python
def load_results(experiment_dir):
    """
    Load experiment results from files.
    
    Args:
        experiment_dir: Directory with results
        
    Returns:
        tuple: (results_df, history_data)
    """
    # Load results CSV
    results_path = os.path.join(experiment_dir, "results.csv")
    results_df = pd.read_csv(results_path)
    
    # Load history data
    history_path = os.path.join(experiment_dir, "history_data.npy")
    history_data = np.load(history_path, allow_pickle=True).item()
    
    return results_df, history_data
```

## 10. Statistical Analysis

### 10.1 Normality Testing

```python
def test_normality(results_df):
    """
    Test normality of fitness values for each configuration.
    
    Args:
        results_df: DataFrame with experiment results
        
    Returns:
        dict: Dictionary with normality test results
    """
    normality_results = {}
    
    for config_name in results_df['Configuration'].unique():
        config_results = results_df[results_df['Configuration'] == config_name]
        fitness_values = config_results['Best Fitness'].values
        
        # Shapiro-Wilk test
        statistic, p_value = stats.shapiro(fitness_values)
        
        normality_results[config_name] = {
            'statistic': statistic,
            'p_value': p_value,
            'normal': p_value > EXPERIMENT_CONFIG['alpha']
        }
    
    return normality_results
```

### 10.2 ANOVA and Post-hoc Tests

```python
def perform_anova(results_df):
    """
    Perform ANOVA test to compare configurations.
    
    Args:
        results_df: DataFrame with experiment results
        
    Returns:
        dict: Dictionary with ANOVA test results
    """
    # Group data by configuration
    groups = []
    group_names = []
    
    for config_name in results_df['Configuration'].unique():
        config_results = results_df[results_df['Configuration'] == config_name]
        groups.append(config_results['Best Fitness'].values)
        group_names.append(config_name)
    
    # Perform ANOVA
    statistic, p_value = stats.f_oneway(*groups)
    
    # Perform post-hoc test if ANOVA is significant
    post_hoc_results = None
    
    if p_value < EXPERIMENT_CONFIG['alpha']:
        if EXPERIMENT_CONFIG['post_hoc_method'] == 'tukey':
            # Tukey HSD test
            post_hoc_results = stats.tukey_hsd(*groups)
        else:
            # Default to Tukey HSD
            post_hoc_results = stats.tukey_hsd(*groups)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < EXPERIMENT_CONFIG['alpha'],
        'post_hoc': post_hoc_results,
        'group_names': group_names
    }
```

## 11. Conclusion

The Fantasy League Team Optimization project follows a modular architecture with clear separation of concerns. The core components (`solution.py`, `evolution.py`, `operators.py`, `fitness_counter.py`) provide the foundation for the optimization algorithms, while the complete pipeline (`CIFO_Complete_Pipeline_Final.py`) integrates these components for execution and analysis.

The project demonstrates good software engineering practices:
1. **Modularity**: Clear separation of concerns
2. **Extensibility**: Easy to add new algorithms and operators
3. **Configurability**: Centralized configuration for experiments
4. **Parallelization**: Support for parallel execution
5. **Data Management**: Comprehensive data storage and loading
6. **Analysis**: Robust statistical analysis and visualization

This architecture allows for efficient experimentation and comparison of different optimization algorithms for the Fantasy League Team Optimization problem.
