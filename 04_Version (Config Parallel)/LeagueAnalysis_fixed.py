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
# This notebook analyzes different optimization algorithms for the Sports League problem.

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
from evolution import hill_climbing, simulated_annealing
from operators import (
    mutate_swap, mutate_swap_constrained, mutate_scramble,
    uniform_crossover, one_point_crossover,
    tournament_selection, ranking_selection
)

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Configure plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# Suppress warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## 1. Configuration

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
# ## 2. Helper Functions

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
            selection_func = tournament_selection
        elif config["selection"] == "ranking":
            selection_func = ranking_selection
        else:
            selection_func = tournament_selection
            
        if config["crossover"] == "uniform":
            crossover_func = uniform_crossover
        elif config["crossover"] == "one_point":
            crossover_func = one_point_crossover
        else:
            crossover_func = uniform_crossover
            
        if config["mutation"] == "swap":
            mutation_func = mutate_swap
        elif config["mutation"] == "swap_constrained":
            mutation_func = mutate_swap_constrained
        elif config["mutation"] == "scramble":
            mutation_func = mutate_scramble
        else:
            mutation_func = mutate_swap
            
        # Importar run_genetic_algorithm aqui para garantir que está disponível
        from evolution import run_genetic_algorithm
        
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
            selection_func = tournament_selection
        elif config["selection"] == "ranking":
            selection_func = ranking_selection
        else:
            selection_func = tournament_selection
            
        if config["crossover"] == "uniform":
            crossover_func = uniform_crossover
        elif config["crossover"] == "one_point":
            crossover_func = one_point_crossover
        else:
            crossover_func = uniform_crossover
            
        if config["mutation"] == "swap":
            mutation_func = mutate_swap
        elif config["mutation"] == "swap_constrained":
            mutation_func = mutate_swap_constrained
        elif config["mutation"] == "scramble":
            mutation_func = mutate_scramble
        else:
            mutation_func = mutate_swap
        
        # Importar run_hybrid_ga aqui para garantir que está disponível
        from evolution import run_hybrid_ga
            
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
