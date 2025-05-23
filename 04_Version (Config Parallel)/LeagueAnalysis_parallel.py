# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: cloudspace
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Sports League Optimization Analysis
#
# Este notebook analisa diferentes algoritmos de otimização para o problema da Liga Esportiva.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
from copy import deepcopy
import multiprocessing
from multiprocessing import Process, Queue
import os
import sys
import warnings
from collections import defaultdict

# Importações explícitas das classes necessárias para execução paralela
from solution import LeagueSolution, LeagueHillClimbingSolution, LeagueSASolution
from evolution import hill_climbing, simulated_annealing, run_genetic_algorithm, run_hybrid_ga
from operators import (
    mutate_swap, mutate_swap_constrained, mutate_team_shift,
    mutate_targeted_player_exchange, mutate_shuffle_within_team_constrained,
    crossover_one_point, crossover_uniform,
    selection_tournament, selection_ranking
)

# Implementação da função mutate_scramble que está faltando
def mutate_scramble(solution, mutation_rate=None):
    """
    Scramble mutation: randomly shuffles a subsequence of the solution.
    
    Args:
        solution (LeagueSolution): The solution to mutate
        mutation_rate (float, optional): Probability of mutation. Defaults to 0.1 if None.
        
    Returns:
        LeagueSolution: A new solution with the mutation applied
    """
    if mutation_rate is None:
        mutation_rate = 0.1
        
    # Create a copy of the solution
    mutated = deepcopy(solution)
    
    # Apply mutation with probability mutation_rate
    if random.random() < mutation_rate:
        # Select random subsequence
        length = len(mutated.repr)
        start = random.randint(0, length - 2)
        end = random.randint(start + 1, length - 1)
        
        # Scramble the subsequence
        subsequence = mutated.repr[start:end+1]
        random.shuffle(subsequence)
        mutated.repr[start:end+1] = subsequence
        
    return mutated

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Configure plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# Suppress warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. Configuração

# %%
# Experiment configuration
EXPERIMENT_CONFIG = {
    'parallel': True,           # Enable/disable parallel execution
    'num_runs': 30,             # Number of runs per configuration
    'num_processes': None,      # Number of parallel processes (None = automatic)
    'max_evaluations': None,    # Maximum fitness evaluations (None = unlimited)
    'save_results': True,       # Save results to file
    'results_file': 'experiment_results.csv',  # Results file name
    'verbose': True,            # Show detailed progress
    'safe_exp_max': 700,        # Maximum value for safe exponential function
    'show_best_solution': True  # Show details of the best solution found
}

# Define algorithm configurations to test
ALGORITHM_CONFIGS = [
    {
        "name": "Hill Climbing",
        "algorithm": "hill_climbing",
        "max_iterations": 1000
    },
    {
        "name": "Intensive Hill Climbing",
        "algorithm": "hill_climbing",
        "max_iterations": 5000,
        "max_neighbors": 100
    },
    {
        "name": "Simulated Annealing",
        "algorithm": "simulated_annealing",
        "initial_temperature": 100,
        "cooling_rate": 0.95,
        "max_iterations": 1000
    },
    {
        "name": "Enhanced Simulated Annealing",
        "algorithm": "simulated_annealing",
        "initial_temperature": 1000,
        "cooling_rate": 0.99,
        "max_iterations": 5000
    },
    {
        "name": "Genetic Algorithm (Tournament)",
        "algorithm": "genetic_algorithm",
        "selection": "tournament",
        "crossover": "uniform",
        "mutation": "swap",
        "population_size": 100,
        "elitism_rate": 0.1,
        "tournament_size": 3,
        "max_generations": 50,
        "mutation_rate": 0.1
    },
    {
        "name": "Genetic Algorithm (Ranking)",
        "algorithm": "genetic_algorithm",
        "selection": "ranking",
        "crossover": "uniform",
        "mutation": "swap",
        "population_size": 100,
        "elitism_rate": 0.1,
        "max_generations": 50,
        "mutation_rate": 0.1
    },
    {
        "name": "Hybrid GA",
        "algorithm": "hybrid_ga",
        "selection": "tournament",
        "crossover": "uniform",
        "mutation": "swap",
        "population_size": 50,
        "elitism_rate": 0.1,
        "tournament_size": 3,
        "max_generations": 40,
        "mutation_rate": 0.1,
        "local_search_rate": 0.2
    },
    {
        "name": "Optimized Hybrid GA",
        "algorithm": "hybrid_ga",
        "selection": "tournament",
        "crossover": "uniform",
        "mutation": "swap_constrained",
        "population_size": 100,
        "elitism_rate": 0.2,
        "tournament_size": 5,
        "max_generations": 50,
        "mutation_rate": 0.15,
        "local_search_rate": 0.3
    }
]

# %% [markdown]
# ## 2. Funções Auxiliares

# %%
# Safe exponential function to prevent overflow
# Importante: Não substituir np.exp globalmente para evitar recursão infinita
def safe_exp(x, max_value=EXPERIMENT_CONFIG['safe_exp_max']):
    """
    Compute exponential function safely to prevent overflow.
    
    Args:
        x: Input value
        max_value: Maximum absolute value to allow before clipping
        
    Returns:
        Exponential of clipped input value
    """
    return np.exp(np.clip(x, -max_value, max_value))

# Fitness counter class
class FitnessCounter:
    """
    Class to count fitness evaluations.
    """
    def __init__(self):
        self.count = 0
        self.original_fitness = None
    
    def start_counting(self):
        """Start counting fitness evaluations."""
        self.original_fitness = LeagueSolution.fitness
        self.count = 0
        
        # Use a wrapper function that correctly handles the 'self' parameter
        def counting_wrapper(instance):
            self.count += 1
            return self.original_fitness(instance)
        
        # Replace the fitness method with our wrapper
        LeagueSolution.fitness = counting_wrapper
    
    def stop_counting(self):
        """Stop counting and restore original fitness function."""
        # Restore original fitness function
        if self.original_fitness:
            LeagueSolution.fitness = self.original_fitness
        
        return self.count

# Function to run a single experiment
def run_experiment(config, players_list, max_evaluations=None):
    """
    Run a single experiment with the specified configuration.
    
    Args:
        config: Algorithm configuration dictionary
        players_list: List of player dictionaries
        max_evaluations: Maximum number of fitness evaluations (None = unlimited)
        
    Returns:
        Tuple of (best_solution, best_fitness, evaluations, runtime, history)
    """
    # Start timing
    start_time = time.time()
    
    # Create fitness counter
    fitness_counter = FitnessCounter()
    
    # Start counting fitness evaluations
    fitness_counter.start_counting()
    
    # Run the appropriate algorithm
    if config["algorithm"] == "hill_climbing":
        max_neighbors = config.get("max_neighbors", 10)
        best_solution, best_fitness, history = hill_climbing(
            solution_class=LeagueSolution,
            players=players_list,
            max_iterations=config["max_iterations"],
            max_neighbors=max_neighbors,
            max_evaluations=max_evaluations
        )
        
    elif config["algorithm"] == "simulated_annealing":
        best_solution, best_fitness, history = simulated_annealing(
            solution_class=LeagueSolution,
            players=players_list,
            initial_temperature=config["initial_temperature"],
            cooling_rate=config["cooling_rate"],
            max_iterations=config["max_iterations"],
            max_evaluations=max_evaluations
        )
        
    elif config["algorithm"] == "genetic_algorithm":
        # Select operators
        if config["selection"] == "tournament":
            selection_func = selection_tournament
        elif config["selection"] == "ranking":
            selection_func = selection_ranking
        else:
            selection_func = selection_tournament
            
        if config["crossover"] == "uniform":
            crossover_func = crossover_uniform
        elif config["crossover"] == "one_point":
            crossover_func = crossover_one_point
        else:
            crossover_func = crossover_uniform
            
        if config["mutation"] == "swap":
            mutation_func = mutate_swap
        elif config["mutation"] == "swap_constrained":
            mutation_func = mutate_swap_constrained
        elif config["mutation"] == "scramble":
            mutation_func = mutate_scramble
        else:
            mutation_func = mutate_swap
            
        best_solution, best_fitness, history = run_genetic_algorithm(
            solution_class=LeagueSolution,
            players=players_list,
            selection_func=selection_func,
            crossover_func=crossover_func,
            mutation_func=mutation_func,
            population_size=config["population_size"],
            elitism_rate=config["elitism_rate"],
            tournament_size=config.get("tournament_size", 3),
            max_generations=config["max_generations"],
            mutation_rate=config.get("mutation_rate", 0.1),
            max_evaluations=max_evaluations
        )
        
    elif config["algorithm"] == "hybrid_ga":
        # Select operators
        if config["selection"] == "tournament":
            selection_func = selection_tournament
        elif config["selection"] == "ranking":
            selection_func = selection_ranking
        else:
            selection_func = selection_tournament
            
        if config["crossover"] == "uniform":
            crossover_func = crossover_uniform
        elif config["crossover"] == "one_point":
            crossover_func = crossover_one_point
        else:
            crossover_func = crossover_uniform
            
        if config["mutation"] == "swap":
            mutation_func = mutate_swap
        elif config["mutation"] == "swap_constrained":
            mutation_func = mutate_swap_constrained
        elif config["mutation"] == "scramble":
            mutation_func = mutate_scramble
        else:
            mutation_func = mutate_swap
            
        best_solution, best_fitness, history = run_hybrid_ga(
            solution_class=LeagueHillClimbingSolution,
            players=players_list,
            selection_func=selection_func,
            crossover_func=crossover_func,
            mutation_func=mutation_func,
            population_size=config["population_size"],
            elitism_rate=config["elitism_rate"],
            tournament_size=config.get("tournament_size", 3),
            max_generations=config["max_generations"],
            mutation_rate=config.get("mutation_rate", 0.1),
            local_search_rate=config.get("local_search_rate", 0.1),
            max_evaluations=max_evaluations
        )
    else:
        raise ValueError(f"Unknown algorithm: {config['algorithm']}")
    
    # Stop counting fitness evaluations
    evaluations = fitness_counter.stop_counting()
    
    # Calculate runtime
    runtime = time.time() - start_time
    
    return best_solution, best_fitness, evaluations, runtime, history

# Function to run multiple experiments sequentially
def run_multiple_experiments(configs, players_list, num_runs=3, max_evaluations=None):
    """
    Run multiple experiments sequentially.
    
    Args:
        configs: List of algorithm configurations
        players_list: List of player dictionaries
        num_runs: Number of runs per configuration
        max_evaluations: Maximum number of fitness evaluations per run
        
    Returns:
        Tuple of (results_df, history_data)
    """
    all_results = []
    history_data = defaultdict(dict)
    
    for config_idx, config in enumerate(configs):
        config_name = config["name"]
        
        if EXPERIMENT_CONFIG['verbose']:
            print(f"Running {config_name}...")
        
        for run in range(num_runs):
            if EXPERIMENT_CONFIG['verbose']:
                print(f"  Run {run+1}/{num_runs}...")
            
            # Run experiment
            result = run_experiment(
                config=config,
                players_list=players_list,
                max_evaluations=max_evaluations
            )
            
            # Extract values from result
            best_solution, best_fitness, evaluations, runtime, history = result
            
            # Store results
            all_results.append({
                'Configuration': config_name,
                'Run': run,
                'Best Fitness': best_fitness,
                'Iterations': len(history),
                'Function Evaluations': evaluations,
                'Runtime (s)': runtime
            })
            
            # Store history
            history_data[config_name][run] = history
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    return results_df, history_data

# Function for worker process in parallel execution
def run_worker(config, players_list, run, max_evaluations, result_queue, error_queue):
    """
    Worker function for parallel execution.
    
    Args:
        config: Algorithm configuration
        players_list: List of player dictionaries
        run: Run number
        max_evaluations: Maximum number of fitness evaluations
        result_queue: Queue for results
        error_queue: Queue for errors
    """
    try:
        # Set random seed based on run number to ensure different results
        random.seed(42 + run)
        np.random.seed(42 + run)
        
        config_name = config["name"]
        
        # Print start message
        print(f"Process started for {config_name}, Run {run+1}")
        
        # Run experiment
        result = run_experiment(
            config=config,
            players_list=players_list,
            max_evaluations=max_evaluations
        )
        
        # Extract values from result
        best_solution, best_fitness, evaluations, runtime, history = result
        
        # Put result in queue
        result_queue.put({
            'Configuration': config_name,
            'Run': run,
            'Best Fitness': best_fitness,
            'Iterations': len(history),
            'Function Evaluations': evaluations,
            'Runtime (s)': runtime,
            'History': history
        })
        
        # Print completion message
        print(f"Process completed for {config_name}, Run {run+1}")
        
    except Exception as e:
        # Put error in queue
        error_queue.put({
            'Configuration': config["name"],
            'Run': run,
            'Error': str(e),
            'Traceback': str(sys.exc_info())
        })
        print(f"Error in {config['name']}, Run {run+1}: {str(e)}")

# Function to run multiple experiments in parallel
def run_parallel_experiments(configs, players_list, num_runs=3, max_evaluations=None, num_processes=None):
    """
    Run multiple experiments in parallel.
    
    Args:
        configs: List of algorithm configurations
        players_list: List of player dictionaries
        num_runs: Number of runs per configuration
        max_evaluations: Maximum number of fitness evaluations per run
        num_processes: Number of parallel processes (None = automatic)
        
    Returns:
        Tuple of (results_df, history_data)
    """
    # Determine number of processes
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    
    if EXPERIMENT_CONFIG['verbose']:
        print(f"Running experiments in parallel with {num_processes} processes...")
    
    # Create queues for results and errors
    result_queue = multiprocessing.Queue()
    error_queue = multiprocessing.Queue()
    
    # Create list of all experiments to run
    experiments = []
    for config in configs:
        for run in range(num_runs):
            experiments.append((config, run))
    
    # Create and start processes
    processes = []
    for config, run in experiments:
        p = multiprocessing.Process(
            target=run_worker,
            args=(config, players_list, run, max_evaluations, result_queue, error_queue)
        )
        processes.append(p)
        p.start()
        
        # Limit number of concurrent processes
        if len(processes) >= num_processes:
            # Wait for a process to finish
            processes[0].join()
            processes.pop(0)
    
    # Wait for remaining processes to finish
    for p in processes:
        p.join()
    
    # Collect results
    all_results = []
    history_data = defaultdict(dict)
    
    # Get results from queue
    while not result_queue.empty():
        result = result_queue.get()
        history = result.pop('History')
        all_results.append(result)
        history_data[result['Configuration']][result['Run']] = history
    
    # Check for errors
    errors = []
    while not error_queue.empty():
        errors.append(error_queue.get())
    
    if errors:
        print(f"Encountered {len(errors)} errors:")
        for error in errors:
            print(f"  {error['Configuration']}, Run {error['Run']+1}: {error['Error']}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    return results_df, history_data

# Function to run experiments (sequential or parallel)
def run_experiments(configs, players_list, experiment_config=None):
    """
    Run experiments with the specified configurations.
    
    Args:
        configs: List of algorithm configurations
        players_list: List of player dictionaries
        experiment_config: Experiment configuration dictionary
        
    Returns:
        Tuple of (results_df, history_data)
    """
    if experiment_config is None:
        experiment_config = EXPERIMENT_CONFIG
    
    # Extract configuration
    parallel = experiment_config.get('parallel', False)
    num_runs = experiment_config.get('num_runs', 3)
    num_processes = experiment_config.get('num_processes', None)
    max_evaluations = experiment_config.get('max_evaluations', None)
    
    # Run experiments
    if parallel:
        results_df, history_data = run_parallel_experiments(
            configs, 
            players_list, 
            num_runs=num_runs, 
            max_evaluations=max_evaluations,
            num_processes=num_processes
        )
    else:
        # Run sequentially
        results_df, history_data = run_multiple_experiments(
            configs, 
            players_list, 
            num_runs=num_runs, 
            max_evaluations=max_evaluations
        )
    
    # Save results
    if experiment_config.get('save_results', False):
        results_file = experiment_config.get('results_file', 'experiment_results.csv')
        results_df.to_csv(results_file, index=False)
        print(f"Results saved to {results_file}")
    
    return results_df, history_data

# %% [markdown]
# ## 3. Carregamento de Dados

# %%
# Função para carregar dados dos jogadores
def load_players_data(file_paths=None):
    """
    Load players data from CSV file.
    
    Args:
        file_paths: List of possible file paths to try
        
    Returns:
        List of player dictionaries
    """
    if file_paths is None:
        file_paths = [
            "players.csv",
            "data/players.csv",
            "../data/players.csv",
            "/home/ubuntu/CIFO-24-25/data/players.csv",
            "/home/ubuntu/CIFO/data/players.csv"
        ]
    
    # Try each path until one works
    for path in file_paths:
        try:
            print(f"Trying to load players data from {path}...")
            players_df = pd.read_csv(path)
            print(f"Successfully loaded players data from {path}")
            
            # Normalize column names if needed
            if "Salary (€M)" in players_df.columns and "Salary" not in players_df.columns:
                # Create a copy of the Salary column with the simpler name
                players_df["Salary"] = players_df["Salary (€M)"]
                print("Normalized 'Salary (€M)' column to 'Salary'")
            
            # Convert DataFrame to list of dictionaries
            players_list = players_df.to_dict('records')
            return players_list
        except FileNotFoundError:
            print(f"File not found: {path}")
    
    raise FileNotFoundError("Could not find players data file in any of the specified paths")

# %% [markdown]
# ## 4. Execução dos Experimentos

# %%
# Carregar dados dos jogadores
try:
    players_list = load_players_data()
    print(f"Loaded {len(players_list)} players")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please provide the correct path to the players.csv file")
    players_list = None

# %%
# Executar experimentos se os dados foram carregados com sucesso
if players_list:
    # Run experiments
    results_df, history_data = run_experiments(
        ALGORITHM_CONFIGS, 
        players_list, 
        experiment_config=EXPERIMENT_CONFIG
    )
    
    # Display results
    print("\nExperiment Results:")
    print(results_df)
else:
    print("Cannot run experiments without player data")

# %% [markdown]
# ## 5. Análise de Resultados

# %%
# Função para calcular estatísticas de resumo
def calculate_summary_statistics(results_df):
    """
    Calculate summary statistics for each configuration.
    
    Args:
        results_df: DataFrame with experiment results
        
    Returns:
        Dictionary of DataFrames with statistics for each metric
    """
    # Group by configuration
    grouped = results_df.groupby('Configuration')
    
    # Calculate statistics for each metric
    metrics = {
        'Best Fitness': 'Fitness Statistics:',
        'Iterations': 'Iterations Statistics:',
        'Function Evaluations': 'Function Evaluations Statistics:',
        'Runtime (s)': 'Runtime Statistics (seconds):'
    }
    
    stats_dfs = {}
    
    for metric, title in metrics.items():
        # Calculate statistics
        metric_stats = grouped[metric].agg(['mean', 'std', 'min', 'max'])
        
        # Format values for display
        metric_stats = metric_stats.map(lambda x: f"{x:.6f}" if not pd.isna(x) and not np.isinf(x) else "N/A")
        
        stats_dfs[metric] = (title, metric_stats)
    
    return stats_dfs

# %%
if 'results_df' in locals():
    # Calculate summary statistics
    stats_dfs = calculate_summary_statistics(results_df)
    
    # Display statistics
    for metric, (title, df) in stats_dfs.items():
        print(f"\n{title}")
        print(df)

# %%
# Função para plotar gráficos de convergência
def plot_convergence_graphs(history_data, num_runs_to_show=3):
    """
    Plot convergence graphs for each algorithm.
    
    Args:
        history_data: Dictionary of history data
        num_runs_to_show: Number of runs to show in the plot
    """
    # Create figure
    fig, axes = plt.subplots(len(history_data), 1, figsize=(12, 4 * len(history_data)))
    
    if len(history_data) == 1:
        axes = [axes]
    
    # Plot each algorithm
    for i, (config_name, runs) in enumerate(history_data.items()):
        ax = axes[i]
        
        # Plot each run
        for run in range(min(num_runs_to_show, len(runs))):
            history = runs[run]
            ax.plot(history, label=f'Run {run+1}')
        
        # Add labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Fitness (lower is better)')
        ax.set_title(f'Convergence for {config_name}')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

# %%
if 'history_data' in locals():
    # Plot convergence graphs
    plot_convergence_graphs(history_data)

# %%
# Função para plotar comparação de algoritmos
def plot_algorithm_comparison(results_df):
    """
    Plot comparison of algorithms.
    
    Args:
        results_df: DataFrame with experiment results
    """
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Metrics to plot
    metrics = [
        ('Best Fitness', 'Fitness (lower is better)'),
        ('Iterations', 'Number of Iterations'),
        ('Function Evaluations', 'Number of Function Evaluations'),
        ('Runtime (s)', 'Runtime (seconds)')
    ]
    
    # Plot each metric
    for i, (metric, ylabel) in enumerate(metrics):
        # Create boxplot with scatterplot overlay
        ax = axes[i]
        
        # Boxplot
        sns.boxplot(x='Configuration', y=metric, data=results_df, ax=ax)
        
        # Scatterplot overlay
        sns.stripplot(x='Configuration', y=metric, data=results_df, 
                     color='black', alpha=0.5, jitter=True, ax=ax)
        
        # Rotate x-axis labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add labels and title
        ax.set_xlabel('Algorithm')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Comparison of {metric} by Algorithm')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

# %%
if 'results_df' in locals():
    # Plot algorithm comparison
    plot_algorithm_comparison(results_df)

# %% [markdown]
# ## 6. Visualização da Melhor Solução

# %%
# Função para encontrar a melhor solução global
def find_best_solution(results_df, history_data, players_list):
    """
    Find the best solution across all experiments.
    
    Args:
        results_df: DataFrame with experiment results
        history_data: Dictionary of history data
        players_list: List of player dictionaries
        
    Returns:
        Tuple of (best_config, best_run, best_fitness)
    """
    if len(results_df) == 0:
        return None, None, None
    
    # Find row with minimum fitness
    best_idx = results_df['Best Fitness'].idxmin()
    best_row = results_df.loc[best_idx]
    
    best_config = best_row['Configuration']
    best_run = best_row['Run']
    best_fitness = best_row['Best Fitness']
    
    print(f"Best solution found by {best_config}, Run {best_run+1}")
    print(f"Fitness: {best_fitness:.6f}")
    print(f"Iterations: {best_row['Iterations']}")
    print(f"Function Evaluations: {best_row['Function Evaluations']}")
    print(f"Runtime: {best_row['Runtime (s)']:.6f} seconds")
    
    return best_config, best_run, best_fitness

# %%
# Função para visualizar a melhor solução
def visualize_best_solution(best_config, best_run, players_list, algorithm_configs):
    """
    Visualize the best solution.
    
    Args:
        best_config: Name of the best configuration
        best_run: Run number of the best solution
        players_list: List of player dictionaries
        algorithm_configs: List of algorithm configurations
    """
    if best_config is None or best_run is None:
        print("No best solution found")
        return
    
    # Find the configuration
    config = next((c for c in algorithm_configs if c["name"] == best_config), None)
    if config is None:
        print(f"Configuration {best_config} not found")
        return
    
    # Re-run the experiment to get the best solution
    best_solution, best_fitness, _, _, _ = run_experiment(config, players_list)
    
    # Get teams
    teams = best_solution.get_teams()
    team_stats = best_solution.get_team_stats()
    
    # Display team statistics
    print("\nTeam Statistics:")
    for i, stats in enumerate(team_stats):
        print(f"\nTeam {i+1}:")
        print(f"  Average Skill: {stats['avg_skill']:.2f}")
        print(f"  Total Salary: {stats['total_salary']:.2f}")
        print(f"  Positions: {stats['positions']}")
        
        # Display players
        print("  Players:")
        for p in stats['players']:
            print(f"    {p['Name']} - {p['Position']} - Skill: {p['Skill']} - Salary: {p.get('Salary', p.get('Salary (€M)', 'N/A'))}")
    
    # Plot team skills
    plt.figure(figsize=(10, 6))
    team_skills = [stats['avg_skill'] for stats in team_stats]
    plt.bar(range(1, len(team_skills) + 1), team_skills)
    plt.axhline(y=np.mean(team_skills), color='r', linestyle='-', label=f'Mean: {np.mean(team_skills):.2f}')
    plt.xlabel('Team')
    plt.ylabel('Average Skill')
    plt.title('Team Balance - Average Skill by Team')
    plt.xticks(range(1, len(team_skills) + 1))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# %%
if 'results_df' in locals() and 'history_data' in locals() and players_list:
    # Find and visualize best solution
    best_config, best_run, best_fitness = find_best_solution(results_df, history_data, players_list)
    
    if EXPERIMENT_CONFIG['show_best_solution'] and best_config is not None:
        visualize_best_solution(best_config, best_run, players_list, ALGORITHM_CONFIGS)

# %%
