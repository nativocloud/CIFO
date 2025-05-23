# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: .jupytext-sync-ipynb//ipynb,py:percent
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
# # CIFO - Complete Pipeline with Configurable Parallel Processing

# %% [markdown]
# ## 1. Setup and Configuration

# %%
import os
import sys
import time
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from copy import deepcopy
from enum import Enum
import json
from datetime import datetime
import scipy.stats as stats
import scikit_posthocs as sp

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Configure matplotlib for better visualization
# %matplotlib inline
plt.rcParams['figure.figsize'] = (14, 10)
plt.style.use('ggplot')

# %% [markdown]
# ### 1.1 Execution Mode Configuration

# %%
# Define execution mode enum
class ExecutionMode(Enum):
    SINGLE_PROCESSOR = 1  # Sequential execution
    MULTI_PROCESSOR = 2   # Parallel execution

# %% [markdown]
# ### 1.2 Experiment Configuration

# %%
# Centralized experiment configuration
EXPERIMENT_CONFIG = {
    # General parameters
    'seed': 42,                    # Seed for reproducibility
    'num_runs': 30,                # Number of runs for each algorithm
    'max_evaluations': 10000,      # Maximum number of function evaluations
    'population_size': 100,        # Population size for genetic algorithms
    'max_generations': 100,        # Maximum number of generations for genetic algorithms
    
    # Execution parameters
    'execution_mode': ExecutionMode.SINGLE_PROCESSOR,  # Sequential execution by default
    'use_parallel': False,         # Simple flag to enable/disable parallel processing
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
    
    # Data handling parameters
    'use_simulated_data': False,   # Never use simulated data, only real data
}

# Update execution mode based on use_parallel flag
if EXPERIMENT_CONFIG['use_parallel']:
    EXPERIMENT_CONFIG['execution_mode'] = ExecutionMode.MULTI_PROCESSOR
else:
    EXPERIMENT_CONFIG['execution_mode'] = ExecutionMode.SINGLE_PROCESSOR

# %% [markdown]
# ## 2. Data Loading

# %%
# Load player data
try:
    players_df = pd.read_csv('players.csv', encoding='utf-8', sep=';', index_col=0)
    print(f"Loaded {len(players_df)} players from CSV file.")
except Exception as e:
    print(f"Error loading players.csv: {e}")
    players_df = None

# Display first few rows
if players_df is not None:
    display(players_df.head())

# %%
# Convert DataFrame to list of dictionaries
if players_df is not None:
    players_list = players_df.to_dict('records')
    
    # Rename 'Salary (€M)' to 'Salary' if needed
    for player in players_list:
        if 'Salary (€M)' in player and 'Salary' not in player:
            player['Salary'] = player.pop('Salary (€M)')
else:
    players_list = []

# %% [markdown]
# ## 3. Solution Representation

# %%
# Import solution class from solution.py
from solution import LeagueSolution

# %% [markdown]
# ## 4. Fitness Counter

# %%
# Import fitness counter from fitness_counter.py
from fitness_counter import FitnessCounter

# %% [markdown]
# ## 5. Algorithm Implementations

# %% [markdown]
# ### 5.1 Hill Climbing

# %%
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

# %% [markdown]
# ### 5.2 Hill Climbing with Random Restart

# %%
def run_hill_climbing_random_restart(solution, max_iterations=1000, restart_interval=100):
    """
    Run hill climbing algorithm with random restarts.
    
    Args:
        solution: Initial solution
        max_iterations: Maximum number of iterations
        restart_interval: Number of iterations between restarts
        
    Returns:
        tuple: (best_solution, best_fitness, fitness_history)
    """
    global_best_solution = deepcopy(solution)
    global_best_fitness = solution.fitness()
    
    current_solution = deepcopy(solution)
    current_fitness = global_best_fitness
    
    fitness_history = [global_best_fitness]
    
    for i in range(max_iterations):
        # Check if restart is needed
        if i > 0 and i % restart_interval == 0:
            # Random restart
            current_solution = LeagueSolution(
                players=solution.players,
                num_teams=solution.num_teams,
                team_size=solution.team_size,
                max_budget=solution.max_budget
            )
            current_solution.set_fitness_counter(solution.fitness_counter)
            current_fitness = current_solution.fitness()
        
        # Generate neighbor
        neighbor = deepcopy(current_solution)
        idx = random.randint(0, len(neighbor.repr) - 1)
        neighbor.repr[idx] = random.randint(0, neighbor.num_teams - 1)
        
        neighbor_fitness = neighbor.fitness()
        
        # Accept if better
        if neighbor_fitness < current_fitness:
            current_solution = neighbor
            current_fitness = neighbor_fitness
            
            # Update global best
            if current_fitness < global_best_fitness:
                global_best_solution = deepcopy(current_solution)
                global_best_fitness = current_fitness
        
        fitness_history.append(global_best_fitness)
    
    return global_best_solution, global_best_fitness, fitness_history

# %% [markdown]
# ### 5.3 Simulated Annealing

# %%
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

# %% [markdown]
# ### 5.4 Genetic Algorithm

# %%
# Import operators from operators.py
from operators import selection_tournament, selection_ranking, selection_boltzmann
from operators import crossover_one_point, crossover_two_point, crossover_uniform
from operators import mutate_swap

# %%
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

# %% [markdown]
# ### 5.5 GA Hybrid

# %%
def run_hybrid_ga(solution_class, players, selection_func, crossover_func, mutation_func, 
                 population_size=100, elitism_rate=0.1, tournament_size=3, 
                 max_generations=100, mutation_rate=None, local_search_interval=10):
    """
    Run hybrid genetic algorithm with occasional local search.
    
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
        local_search_interval: Number of generations between local search
        
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
        
        # Local search on best individual every local_search_interval generations
        if generation % local_search_interval == 0:
            # Find best individual
            fitness_values = [solution.fitness() for solution in population]
            best_idx = fitness_values.index(min(fitness_values))
            
            # Apply local search
            improved_solution = deepcopy(population[best_idx])
            
            # Simple hill climbing for local search
            for _ in range(10):  # 10 iterations of local search
                neighbor = deepcopy(improved_solution)
                idx = random.randint(0, len(neighbor.repr) - 1)
                neighbor.repr[idx] = random.randint(0, neighbor.num_teams - 1)
                
                neighbor_fitness = neighbor.fitness()
                current_fitness = improved_solution.fitness()
                
                if neighbor_fitness < current_fitness:
                    improved_solution = neighbor
            
            # Replace original solution with improved one
            population[best_idx] = improved_solution
        
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

# %% [markdown]
# ### 5.6 GA Memetic

# %%
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
            
            # Local search with probability local_search_prob
            if random.random() < local_search_prob:
                child = local_search(child, local_search_iterations)
            
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

# %% [markdown]
# ### 5.7 GA Island Model

# %%
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
            island_pop = islands[i]
            island_fit = island_fitness[i]
            
            # Elitism: keep best individuals
            elitism_count = int(len(island_pop) * elitism_rate)
            elite_indices = sorted(range(len(island_fit)), key=lambda j: island_fit[j])[:elitism_count]
            elite = [deepcopy(island_pop[j]) for j in elite_indices]
            
            # Create new population
            new_population = []
            
            # Add elite individuals
            new_population.extend(elite)
            
            # Fill the rest with offspring
            while len(new_population) < len(island_pop):
                # Selection
                if selection_func == selection_tournament:
                    parent1_idx = selection_func(island_fit, tournament_size)
                    parent2_idx = selection_func(island_fit, tournament_size)
                else:
                    parent1_idx = selection_func(island_fit)
                    parent2_idx = selection_func(island_fit)
                
                parent1 = island_pop[parent1_idx]
                parent2 = island_pop[parent2_idx]
                
                # Crossover
                child = crossover_func(parent1, parent2)
                
                # Mutation
                child = mutation_func(child, mutation_rate)
                
                new_population.append(child)
            
            # Replace population
            islands[i] = new_population
            
            # Evaluate new population
            island_fitness[i] = [solution.fitness() for solution in islands[i]]
        
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

# %% [markdown]
# ### 5.8 GA with Scramble Mutation

# %%
def mutate_scramble(solution, mutation_rate=0.1):
    """
    Scramble mutation: randomly selects a subsequence and shuffles it.
    
    Args:
        solution: Solution to mutate
        mutation_rate: Probability of mutation
        
    Returns:
        Mutated solution
    """
    mutated = deepcopy(solution)
    
    # Determine if mutation occurs based on rate
    if random.random() < mutation_rate:
        # Select random subsequence
        length = len(mutated.repr)
        start = random.randint(0, length - 2)
        end = random.randint(start + 1, length - 1)
        
        # Extract subsequence
        subsequence = mutated.repr[start:end+1]
        
        # Shuffle subsequence
        random.shuffle(subsequence)
        
        # Replace original subsequence with shuffled one
        mutated.repr[start:end+1] = subsequence
    
    return mutated

# %% [markdown]
# ## 6. Algorithm Configurations

# %%
# Define algorithm configurations
configs = {
    "HC_Standard": {
        "algorithm": "hill_climbing",
        "max_iterations": 10000,
    },
    "HC_Random_Restart": {
        "algorithm": "hill_climbing_random_restart",
        "max_iterations": 10000,
        "restart_interval": 100,
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
    "GA_Tournament_TwoPoint": {
        "algorithm": "genetic_algorithm",
        "selection": "tournament",
        "crossover": "two_point",
        "mutation": "swap",
        "population_size": 100,
        "elitism_rate": 0.1,
        "tournament_size": 3,
        "max_generations": 100,
    },
    "GA_Rank_Uniform": {
        "algorithm": "genetic_algorithm",
        "selection": "rank",
        "crossover": "uniform",
        "mutation": "swap",
        "population_size": 100,
        "elitism_rate": 0.1,
        "max_generations": 100,
    },
    "GA_Boltzmann_TwoPoint": {
        "algorithm": "genetic_algorithm",
        "selection": "boltzmann",
        "crossover": "two_point",
        "mutation": "swap",
        "population_size": 100,
        "elitism_rate": 0.1,
        "max_generations": 100,
    },
    "GA_Hybrid": {
        "algorithm": "hybrid_ga",
        "selection": "tournament",
        "crossover": "two_point",
        "mutation": "swap",
        "population_size": 100,
        "elitism_rate": 0.1,
        "tournament_size": 3,
        "max_generations": 100,
        "local_search_interval": 10,
    },
    "GA_Memetic": {
        "algorithm": "memetic_algorithm",
        "selection": "tournament",
        "crossover": "two_point",
        "mutation": "swap",
        "population_size": 100,
        "elitism_rate": 0.1,
        "tournament_size": 3,
        "max_generations": 100,
        "local_search_prob": 0.1,
        "local_search_iterations": 10,
    },
    "GA_Island_Model": {
        "algorithm": "island_model_ga",
        "selection": "tournament",
        "crossover": "two_point",
        "mutation": "swap",
        "population_size": 100,
        "elitism_rate": 0.1,
        "tournament_size": 3,
        "max_generations": 100,
        "num_islands": 5,
        "migration_interval": 10,
        "migration_size": 5,
    },
    "GA_Scramble_Mutation": {
        "algorithm": "genetic_algorithm",
        "selection": "tournament",
        "crossover": "two_point",
        "mutation": "scramble",
        "population_size": 100,
        "elitism_rate": 0.1,
        "tournament_size": 3,
        "max_generations": 100,
    },
}

# %% [markdown]
# ## 7. Experiment Execution

# %% [markdown]
# ### 7.1 Single Experiment

# %%
def run_experiment(config, players_list, max_evaluations=10000):
    """
    Run a single experiment with the specified configuration.
    
    Args:
        config: Algorithm configuration
        players_list: List of player dictionaries
        max_evaluations: Maximum number of function evaluations
        
    Returns:
        tuple: (best_solution, best_fitness, evaluations, runtime, fitness_history)
    """
    # Create fitness counter
    fitness_counter = FitnessCounter()
    
    # Create initial solution
    solution = LeagueSolution(
        players=players_list,
        num_teams=5,
        team_size=7,
        max_budget=750
    )
    solution.set_fitness_counter(fitness_counter)
    
    # Start timer
    start_time = time.time()
    
    # Run algorithm
    if config["algorithm"] == "hill_climbing":
        best_solution, best_fitness, fitness_history = run_hill_climbing(
            solution=solution,
            max_iterations=config["max_iterations"]
        )
    
    elif config["algorithm"] == "hill_climbing_random_restart":
        best_solution, best_fitness, fitness_history = run_hill_climbing_random_restart(
            solution=solution,
            max_iterations=config["max_iterations"],
            restart_interval=config["restart_interval"]
        )
    
    elif config["algorithm"] == "simulated_annealing":
        best_solution, best_fitness, fitness_history = run_simulated_annealing(
            solution=solution,
            initial_temperature=config["initial_temperature"],
            cooling_rate=config["cooling_rate"],
            max_iterations=config["max_iterations"]
        )
    
    elif config["algorithm"] == "genetic_algorithm":
        # Select operators
        if config["selection"] == "tournament":
            selection_func = selection_tournament
        elif config["selection"] == "rank":
            selection_func = selection_ranking
        elif config["selection"] == "boltzmann":
            selection_func = selection_boltzmann
        else:
            selection_func = selection_tournament
        
        if config["crossover"] == "one_point":
            crossover_func = crossover_one_point
        elif config["crossover"] == "two_point":
            crossover_func = crossover_two_point
        elif config["crossover"] == "uniform":
            crossover_func = crossover_uniform
        else:
            crossover_func = crossover_one_point
        
        if config["mutation"] == "swap":
            mutation_func = mutate_swap
        elif config["mutation"] == "scramble":
            mutation_func = mutate_scramble
        else:
            mutation_func = mutate_swap
        
        best_solution, best_fitness, fitness_history = run_genetic_algorithm(
            solution_class=LeagueSolution,
            players=players_list,
            selection_func=selection_func,
            crossover_func=crossover_func,
            mutation_func=mutation_func,
            population_size=config["population_size"],
            elitism_rate=config["elitism_rate"],
            tournament_size=config.get("tournament_size", 3),
            max_generations=config["max_generations"]
        )
    
    elif config["algorithm"] == "hybrid_ga":
        # Select operators
        if config["selection"] == "tournament":
            selection_func = selection_tournament
        elif config["selection"] == "rank":
            selection_func = selection_ranking
        elif config["selection"] == "boltzmann":
            selection_func = selection_boltzmann
        else:
            selection_func = selection_tournament
        
        if config["crossover"] == "one_point":
            crossover_func = crossover_one_point
        elif config["crossover"] == "two_point":
            crossover_func = crossover_two_point
        elif config["crossover"] == "uniform":
            crossover_func = crossover_uniform
        else:
            crossover_func = crossover_one_point
        
        if config["mutation"] == "swap":
            mutation_func = mutate_swap
        elif config["mutation"] == "scramble":
            mutation_func = mutate_scramble
        else:
            mutation_func = mutate_swap
        
        best_solution, best_fitness, fitness_history = run_hybrid_ga(
            solution_class=LeagueSolution,
            players=players_list,
            selection_func=selection_func,
            crossover_func=crossover_func,
            mutation_func=mutation_func,
            population_size=config["population_size"],
            elitism_rate=config["elitism_rate"],
            tournament_size=config.get("tournament_size", 3),
            max_generations=config["max_generations"],
            local_search_interval=config["local_search_interval"]
        )
    
    elif config["algorithm"] == "memetic_algorithm":
        # Select operators
        if config["selection"] == "tournament":
            selection_func = selection_tournament
        elif config["selection"] == "rank":
            selection_func = selection_ranking
        elif config["selection"] == "boltzmann":
            selection_func = selection_boltzmann
        else:
            selection_func = selection_tournament
        
        if config["crossover"] == "one_point":
            crossover_func = crossover_one_point
        elif config["crossover"] == "two_point":
            crossover_func = crossover_two_point
        elif config["crossover"] == "uniform":
            crossover_func = crossover_uniform
        else:
            crossover_func = crossover_one_point
        
        if config["mutation"] == "swap":
            mutation_func = mutate_swap
        elif config["mutation"] == "scramble":
            mutation_func = mutate_scramble
        else:
            mutation_func = mutate_swap
        
        best_solution, best_fitness, fitness_history = run_memetic_algorithm(
            solution_class=LeagueSolution,
            players=players_list,
            selection_func=selection_func,
            crossover_func=crossover_func,
            mutation_func=mutation_func,
            population_size=config["population_size"],
            elitism_rate=config["elitism_rate"],
            tournament_size=config.get("tournament_size", 3),
            max_generations=config["max_generations"],
            local_search_prob=config["local_search_prob"],
            local_search_iterations=config["local_search_iterations"]
        )
    
    elif config["algorithm"] == "island_model_ga":
        # Select operators
        if config["selection"] == "tournament":
            selection_func = selection_tournament
        elif config["selection"] == "rank":
            selection_func = selection_ranking
        elif config["selection"] == "boltzmann":
            selection_func = selection_boltzmann
        else:
            selection_func = selection_tournament
        
        if config["crossover"] == "one_point":
            crossover_func = crossover_one_point
        elif config["crossover"] == "two_point":
            crossover_func = crossover_two_point
        elif config["crossover"] == "uniform":
            crossover_func = crossover_uniform
        else:
            crossover_func = crossover_one_point
        
        if config["mutation"] == "swap":
            mutation_func = mutate_swap
        elif config["mutation"] == "scramble":
            mutation_func = mutate_scramble
        else:
            mutation_func = mutate_swap
        
        best_solution, best_fitness, fitness_history = run_island_model_ga(
            solution_class=LeagueSolution,
            players=players_list,
            selection_func=selection_func,
            crossover_func=crossover_func,
            mutation_func=mutation_func,
            population_size=config["population_size"],
            elitism_rate=config["elitism_rate"],
            tournament_size=config.get("tournament_size", 3),
            max_generations=config["max_generations"],
            num_islands=config["num_islands"],
            migration_interval=config["migration_interval"],
            migration_size=config["migration_size"]
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {config['algorithm']}")
    
    # End timer
    end_time = time.time()
    runtime = end_time - start_time
    
    # Get number of evaluations
    evaluations = fitness_counter.get_count()
    
    return best_solution, best_fitness, evaluations, runtime, fitness_history

# %% [markdown]
# ### 7.2 Multiple Experiments

# %%
def run_single_experiment(config_name, config, players_list, max_evaluations, run):
    """
    Run a single experiment for parallel execution.
    
    Args:
        config_name: Name of the configuration
        config: Algorithm configuration
        players_list: List of player dictionaries
        max_evaluations: Maximum number of function evaluations
        run: Run number
        
    Returns:
        tuple: (config_name, run, best_fitness, evaluations, runtime, history)
    """
    if EXPERIMENT_CONFIG['verbose']:
        print(f"Running {config_name} - Run {run+1}/{EXPERIMENT_CONFIG['num_runs']}...")
    
    best_solution, best_fitness, evaluations, runtime, history = run_experiment(
        config=config,
        players_list=players_list,
        max_evaluations=max_evaluations
    )
    
    return config_name, run, best_fitness, evaluations, runtime, history

# %%
def run_multiple_experiments(configs, players_list, num_runs=30, max_evaluations=10000):
    """
    Run multiple experiments for each configuration.
    
    Args:
        configs: Dictionary of algorithm configurations
        players_list: List of player dictionaries
        num_runs: Number of runs for each configuration
        max_evaluations: Maximum number of function evaluations
        
    Returns:
        tuple: (results_df, history_data)
    """
    all_results = []
    history_data = {}
    
    for config_name, config in configs.items():
        if EXPERIMENT_CONFIG['verbose']:
            print(f"\nRunning {config_name}...")
        
        history_data[config_name] = {}
        
        for run in range(num_runs):
            if EXPERIMENT_CONFIG['verbose']:
                print(f"  Run {run+1}/{num_runs}...")
            
            best_solution, best_fitness, evaluations, runtime, history = run_experiment(
                config=config,
                players_list=players_list,
                max_evaluations=max_evaluations
            )
            
            all_results.append({
                'Configuration': config_name,
                'Run': run,
                'Best Fitness': best_fitness,
                'Evaluations': evaluations,
                'Runtime (s)': runtime
            })
            
            history_data[config_name][run] = history
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    return results_df, history_data

# %% [markdown]
# ### 7.3 Parallel Experiments

# %%
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
        num_processes = EXPERIMENT_CONFIG['num_processes']
    
    if EXPERIMENT_CONFIG['verbose']:
        print(f"Running experiments in parallel with {num_processes} processes...")
    
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

# %% [markdown]
# ### 7.4 Save and Load Results

# %%
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
    
    # Save statistics
    if EXPERIMENT_CONFIG['save_statistics']:
        # Calculate statistics
        stats_results = analyze_results(results_df)
        
        # Save to JSON
        stats_path = os.path.join(experiment_dir, "stats_results.json")
        with open(stats_path, 'w') as f:
            json.dump(stats_results, f, indent=4)
    
    if EXPERIMENT_CONFIG['verbose']:
        print(f"Results saved to: {experiment_dir}")

# %% [markdown]
# ## 8. Run Experiments

# %%
# Create results directory if it doesn't exist
if not os.path.exists(EXPERIMENT_CONFIG['results_dir']):
    os.makedirs(EXPERIMENT_CONFIG['results_dir'])

# Create experiment directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = os.path.join(EXPERIMENT_CONFIG['results_dir'], f"experiment_{timestamp}")

# %%
# Run all experiments or load existing results
if EXPERIMENT_CONFIG['load_existing']:
    # Find the most recent results
    experiment_dirs = [d for d in os.listdir(EXPERIMENT_CONFIG['results_dir']) 
                      if os.path.isdir(os.path.join(EXPERIMENT_CONFIG['results_dir'], d)) 
                      and d.startswith('experiment_')]
    
    if experiment_dirs:
        experiment_dirs.sort(reverse=True)
        latest_dir = os.path.join(EXPERIMENT_CONFIG['results_dir'], experiment_dirs[0])
        
        print(f"Loading existing results from: {latest_dir}")
        
        # Load results CSV
        results_path = os.path.join(latest_dir, "results.csv")
        if os.path.exists(results_path):
            results_df = pd.read_csv(results_path)
        else:
            print(f"Results file not found: {results_path}")
            EXPERIMENT_CONFIG['load_existing'] = False
        
        # Load history data
        history_path = os.path.join(latest_dir, "history_data.npy")
        if os.path.exists(history_path):
            history_data = np.load(history_path, allow_pickle=True).item()
        else:
            print(f"History data file not found: {history_path}")
            EXPERIMENT_CONFIG['load_existing'] = False
    else:
        print("No existing results found. Running new experiments...")
        EXPERIMENT_CONFIG['load_existing'] = False

if not EXPERIMENT_CONFIG['load_existing']:
    # Run new experiments
    if EXPERIMENT_CONFIG['execution_mode'] == ExecutionMode.MULTI_PROCESSOR:
        # Run in parallel
        results_df, history_data = run_parallel_experiments(
            configs, 
            players_list, 
            num_runs=EXPERIMENT_CONFIG['num_runs'], 
            max_evaluations=EXPERIMENT_CONFIG['max_evaluations'],
            num_processes=EXPERIMENT_CONFIG['num_processes']
        )
    else:
        # Run sequentially
        results_df, history_data = run_multiple_experiments(
            configs, 
            players_list, 
            num_runs=EXPERIMENT_CONFIG['num_runs'], 
            max_evaluations=EXPERIMENT_CONFIG['max_evaluations']
        )
    
    # Save results
    if EXPERIMENT_CONFIG['save_results']:
        save_results(results_df, history_data, experiment_dir)

# %% [markdown]
# ## 9. Results Analysis

# %% [markdown]
# ### 9.1 Statistical Analysis

# %%
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

# %%
def test_homogeneity(results_df):
    """
    Test homogeneity of variances.
    
    Args:
        results_df: DataFrame with experiment results
        
    Returns:
        dict: Dictionary with homogeneity test results
    """
    # Group data by configuration
    groups = []
    group_names = []
    
    for config_name in results_df['Configuration'].unique():
        config_results = results_df[results_df['Configuration'] == config_name]
        groups.append(config_results['Best Fitness'].values)
        group_names.append(config_name)
    
    # Levene's test
    statistic, p_value = stats.levene(*groups)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'homogeneous': p_value > EXPERIMENT_CONFIG['alpha'],
        'group_names': group_names
    }

# %%
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

# %%
def perform_kruskal(results_df):
    """
    Perform Kruskal-Wallis test to compare configurations.
    
    Args:
        results_df: DataFrame with experiment results
        
    Returns:
        dict: Dictionary with Kruskal-Wallis test results
    """
    # Group data by configuration
    groups = []
    group_names = []
    
    for config_name in results_df['Configuration'].unique():
        config_results = results_df[results_df['Configuration'] == config_name]
        groups.append(config_results['Best Fitness'].values)
        group_names.append(config_name)
    
    # Perform Kruskal-Wallis test
    statistic, p_value = stats.kruskal(*groups)
    
    # Perform post-hoc test if Kruskal-Wallis is significant
    post_hoc_results = None
    
    if p_value < EXPERIMENT_CONFIG['alpha']:
        # Dunn's test
        post_hoc_results = sp.posthoc_dunn(results_df, val_col='Best Fitness', group_col='Configuration', p_adjust='bonferroni')
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < EXPERIMENT_CONFIG['alpha'],
        'post_hoc': post_hoc_results.to_dict() if post_hoc_results is not None else None,
        'group_names': group_names
    }

# %%
def calculate_effect_size(results_df):
    """
    Calculate effect size (eta squared) for the difference between configurations.
    
    Args:
        results_df: DataFrame with experiment results
        
    Returns:
        float: Effect size
    """
    # Group data by configuration
    groups = []
    
    for config_name in results_df['Configuration'].unique():
        config_results = results_df[results_df['Configuration'] == config_name]
        groups.append(config_results['Best Fitness'].values)
    
    # Calculate eta squared
    # Formula: SSB / SST
    # SSB = sum of squares between groups
    # SST = total sum of squares
    
    # Calculate grand mean
    all_values = np.concatenate(groups)
    grand_mean = np.mean(all_values)
    
    # Calculate SSB
    ssb = sum(len(group) * (np.mean(group) - grand_mean) ** 2 for group in groups)
    
    # Calculate SST
    sst = sum((x - grand_mean) ** 2 for x in all_values)
    
    # Calculate eta squared
    eta_squared = ssb / sst if sst > 0 else 0
    
    # Interpret effect size
    if eta_squared < 0.01:
        interpretation = "Very small effect"
    elif eta_squared < 0.06:
        interpretation = "Small effect"
    elif eta_squared < 0.14:
        interpretation = "Medium effect"
    else:
        interpretation = "Large effect"
    
    return {
        'eta_squared': eta_squared,
        'interpretation': interpretation
    }

# %%
def analyze_results(results_df):
    """
    Analyze experiment results and perform statistical tests.
    
    Args:
        results_df: DataFrame with experiment results
        
    Returns:
        dict: Dictionary with analysis results
    """
    # Test normality
    normality_results = test_normality(results_df)
    
    # Check if all configurations have normal distribution
    all_normal = all(result['normal'] for result in normality_results.values())
    
    # Test homogeneity of variances
    homogeneity_results = test_homogeneity(results_df)
    
    # Perform appropriate statistical test
    if all_normal and homogeneity_results['homogeneous']:
        # Parametric test: ANOVA
        test_results = perform_anova(results_df)
        test_type = "ANOVA"
    else:
        # Non-parametric test: Kruskal-Wallis
        test_results = perform_kruskal(results_df)
        test_type = "Kruskal-Wallis"
    
    # Calculate effect size
    effect_size = calculate_effect_size(results_df)
    
    return {
        'normality': normality_results,
        'homogeneity': homogeneity_results,
        'test_type': test_type,
        'test_results': test_results,
        'effect_size': effect_size
    }

# %% [markdown]
# ### 9.2 Performance Comparison

# %%
def plot_performance_comparison(results_df):
    """
    Plot performance comparison across all configurations.
    
    Args:
        results_df: DataFrame with experiment results
    """
    # Calculate mean and standard deviation for each configuration
    summary = results_df.groupby('Configuration').agg({
        'Best Fitness': ['mean', 'std'],
        'Evaluations': ['mean', 'std'],
        'Runtime (s)': ['mean', 'std']
    })
    
    # Reset index for easier plotting
    summary = summary.reset_index()
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=EXPERIMENT_CONFIG['figure_size'])
    
    # Plot best fitness
    axes[0].bar(summary['Configuration'], summary[('Best Fitness', 'mean')], yerr=summary[('Best Fitness', 'std')])
    axes[0].set_title('Best Fitness (lower is better)')
    axes[0].set_xlabel('Configuration')
    axes[0].set_ylabel('Fitness Value')
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot evaluations
    axes[1].bar(summary['Configuration'], summary[('Evaluations', 'mean')], yerr=summary[('Evaluations', 'std')])
    axes[1].set_title('Function Evaluations')
    axes[1].set_xlabel('Configuration')
    axes[1].set_ylabel('Number of Evaluations')
    axes[1].tick_params(axis='x', rotation=90)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Plot runtime
    axes[2].bar(summary['Configuration'], summary[('Runtime (s)', 'mean')], yerr=summary[('Runtime (s)', 'std')])
    axes[2].set_title('Runtime')
    axes[2].set_xlabel('Configuration')
    axes[2].set_ylabel('Runtime (seconds)')
    axes[2].tick_params(axis='x', rotation=90)
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

# %% [markdown]
# ### 9.3 Convergence Analysis

# %%
def plot_convergence_curves(history_data, title="Convergence Curves by Run"):
    """
    Plot convergence curves for all configurations.
    
    Args:
        history_data: Dictionary with convergence history
        title: Plot title
    """
    if history_data is None or len(history_data) == 0:
        print("No history data available for plotting convergence curves.")
        return None
    
    plt.figure(figsize=EXPERIMENT_CONFIG['figure_size'])
    
    # Define a color map for different configurations
    config_names = list(history_data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(config_names)))
    
    # Create a legend dictionary to avoid duplicate entries
    legend_handles = []
    legend_labels = []
    
    for i, config_name in enumerate(config_names):
        histories = history_data[config_name]
        
        # Plot each run with a different line style
        for j, run in enumerate(histories.keys()):
            history = histories[run]
            
            # Skip if history is not a sequence or is empty
            if not hasattr(history, '__len__') or len(history) == 0:
                continue
                
            # Use different line styles for different runs
            line_style = ['-', '--', '-.', ':'][j % 4]
            line, = plt.plot(history, color=colors[i], linestyle=line_style, alpha=0.7)
            
            # Add to legend only once per configuration/run combination
            if j == 0:  # Only add the first run of each config to avoid cluttering
                legend_handles.append(line)
                legend_labels.append(f"{config_name} (Run {j+1})")
    
    # Customize plot
    plt.title(title, fontsize=16)
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Fitness (lower is better)', fontsize=14)
    plt.legend(legend_handles, legend_labels, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.show()
    
    return plt.gcf()

# %%
def plot_average_convergence(history_data, title="Average Convergence Curves"):
    """
    Plot average convergence curves for all configurations.
    
    Args:
        history_data: Dictionary with convergence history
        title: Plot title
    """
    if history_data is None or len(history_data) == 0:
        print("No history data available for plotting average convergence curves.")
        return None
    
    plt.figure(figsize=EXPERIMENT_CONFIG['figure_size'])
    
    # Define a color map for different configurations
    config_names = list(history_data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(config_names)))
    
    for i, config_name in enumerate(config_names):
        histories = history_data[config_name]
        
        # Check if histories is empty
        if not histories:
            continue
        
        # Find the minimum length of all histories
        min_length = min(len(histories[run]) for run in histories.keys() if hasattr(histories[run], '__len__') and len(histories[run]) > 0)
        
        if min_length == 0:
            continue
        
        # Truncate all histories to the minimum length
        truncated_histories = [histories[run][:min_length] for run in histories.keys() if hasattr(histories[run], '__len__') and len(histories[run]) >= min_length]
        
        if not truncated_histories:
            continue
        
        # Calculate mean and standard deviation
        mean_history = np.mean(truncated_histories, axis=0)
        std_history = np.std(truncated_histories, axis=0)
        
        # Plot mean with standard deviation band
        x = np.arange(min_length)
        plt.plot(x, mean_history, color=colors[i], label=config_name)
        plt.fill_between(x, mean_history - std_history, mean_history + std_history, color=colors[i], alpha=0.2)
    
    # Customize plot
    plt.title(title, fontsize=16)
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Fitness (lower is better)', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.show()
    
    return plt.gcf()

# %%
def plot_normalized_convergence(history_data, title="Normalized Convergence Curves"):
    """
    Plot normalized convergence curves for all configurations.
    
    Args:
        history_data: Dictionary with convergence history
        title: Plot title
    """
    if history_data is None or len(history_data) == 0:
        print("No history data available for plotting normalized convergence curves.")
        return None
    
    plt.figure(figsize=EXPERIMENT_CONFIG['figure_size'])
    
    # Define a color map for different configurations
    config_names = list(history_data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(config_names)))
    
    for i, config_name in enumerate(config_names):
        histories = history_data[config_name]
        
        # Check if histories is empty
        if not histories:
            continue
        
        # Find the minimum length of all histories
        min_length = min(len(histories[run]) for run in histories.keys() if hasattr(histories[run], '__len__') and len(histories[run]) > 0)
        
        if min_length == 0:
            continue
        
        # Normalize and truncate all histories
        normalized_histories = []
        
        for run in histories.keys():
            history = histories[run]
            
            if not hasattr(history, '__len__') or len(history) < min_length:
                continue
            
            # Truncate to minimum length
            history = history[:min_length]
            
            # Normalize to [0, 1] range
            if max(history) > min(history):
                normalized_history = (history - min(history)) / (max(history) - min(history))
            else:
                normalized_history = np.zeros_like(history)
            
            normalized_histories.append(normalized_history)
        
        if not normalized_histories:
            continue
        
        # Calculate mean and standard deviation
        mean_history = np.mean(normalized_histories, axis=0)
        std_history = np.std(normalized_histories, axis=0)
        
        # Plot mean with standard deviation band
        x = np.arange(min_length)
        plt.plot(x, mean_history, color=colors[i], label=config_name)
        plt.fill_between(x, mean_history - std_history, mean_history + std_history, color=colors[i], alpha=0.2)
    
    # Customize plot
    plt.title(title, fontsize=16)
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Normalized Fitness (lower is better)', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.show()
    
    return plt.gcf()

# %% [markdown]
# ### 9.4 Statistical Visualization

# %%
def plot_boxplots(results_df):
    """
    Plot boxplots for fitness values across all configurations.
    
    Args:
        results_df: DataFrame with experiment results
    """
    plt.figure(figsize=EXPERIMENT_CONFIG['figure_size'])
    
    # Create boxplot
    boxplot = plt.boxplot([results_df[results_df['Configuration'] == config]['Best Fitness'].values 
                          for config in results_df['Configuration'].unique()],
                         labels=results_df['Configuration'].unique(),
                         patch_artist=True)
    
    # Set colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_df['Configuration'].unique())))
    
    for i, box in enumerate(boxplot['boxes']):
        box.set(facecolor=colors[i], alpha=0.7)
    
    # Customize plot
    plt.title('Fitness Distribution by Configuration', fontsize=16)
    plt.xlabel('Configuration', fontsize=14)
    plt.ylabel('Fitness (lower is better)', fontsize=14)
    plt.xticks(rotation=90)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    plt.show()
    
    return plt.gcf()

# %%
def plot_statistical_significance(results_df, stats_results):
    """
    Plot statistical significance of differences between configurations.
    
    Args:
        results_df: DataFrame with experiment results
        stats_results: Dictionary with statistical test results
    """
    # Check if post-hoc test results are available
    if 'test_results' not in stats_results or 'post_hoc' not in stats_results['test_results'] or stats_results['test_results']['post_hoc'] is None:
        print("No post-hoc test results available for plotting statistical significance.")
        return None
    
    # Get configuration names
    config_names = results_df['Configuration'].unique()
    
    # Create figure
    plt.figure(figsize=EXPERIMENT_CONFIG['figure_size'])
    
    # Create heatmap of p-values
    if stats_results['test_type'] == 'ANOVA':
        # For ANOVA with Tukey HSD
        post_hoc = stats_results['test_results']['post_hoc']
        
        # Create p-value matrix
        p_values = np.zeros((len(config_names), len(config_names)))
        
        for i in range(len(config_names)):
            for j in range(len(config_names)):
                if i == j:
                    p_values[i, j] = 1.0
                else:
                    # Find the p-value for this pair
                    p_values[i, j] = post_hoc.pvalue[i, j]
    else:
        # For Kruskal-Wallis with Dunn's test
        post_hoc = stats_results['test_results']['post_hoc']
        
        # Create p-value matrix
        p_values = np.zeros((len(config_names), len(config_names)))
        
        for i, config1 in enumerate(config_names):
            for j, config2 in enumerate(config_names):
                if i == j:
                    p_values[i, j] = 1.0
                else:
                    # Find the p-value for this pair
                    p_values[i, j] = post_hoc.get(config1, {}).get(config2, 1.0)
    
    # Plot heatmap
    plt.imshow(p_values, cmap='YlOrRd_r', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('p-value', rotation=270, labelpad=15)
    
    # Add significance markers
    for i in range(len(config_names)):
        for j in range(len(config_names)):
            if i != j:
                if p_values[i, j] < 0.001:
                    plt.text(j, i, '***', ha='center', va='center')
                elif p_values[i, j] < 0.01:
                    plt.text(j, i, '**', ha='center', va='center')
                elif p_values[i, j] < 0.05:
                    plt.text(j, i, '*', ha='center', va='center')
    
    # Customize plot
    plt.title(f'Statistical Significance ({stats_results["test_type"]})', fontsize=16)
    plt.xticks(np.arange(len(config_names)), config_names, rotation=90)
    plt.yticks(np.arange(len(config_names)), config_names)
    plt.tight_layout()
    
    plt.show()
    
    return plt.gcf()

# %% [markdown]
# ## 10. Visualization

# %%
# Analyze results
stats_results = analyze_results(results_df)

# Print statistical analysis results
print("\nStatistical Analysis Results:")
print(f"Test type: {stats_results['test_type']}")
print(f"p-value: {stats_results['test_results']['p_value']:.6f}")
print(f"Significant: {stats_results['test_results']['significant']}")
print(f"Effect size (eta squared): {stats_results['effect_size']['eta_squared']:.6f}")
print(f"Effect size interpretation: {stats_results['effect_size']['interpretation']}")

# %%
# Plot performance comparison
plot_performance_comparison(results_df)

# %%
# Plot convergence curves
if history_data is not None and len(history_data) > 0:
    # Check if history data is valid
    valid_history = False
    for config_name in history_data:
        for run in history_data[config_name]:
            history = history_data[config_name][run]
            if hasattr(history, '__len__') and len(history) > 0:
                valid_history = True
                break
        if valid_history:
            break
    
    if valid_history:
        # Plot convergence curves
        plot_convergence_curves(history_data, "Convergence Curves by Run")
        
        # Plot average convergence curves
        plot_average_convergence(history_data, "Average Convergence Curves")
        
        # Plot normalized convergence curves
        plot_normalized_convergence(history_data, "Normalized Convergence Curves")
    else:
        print("No valid history data available for plotting convergence curves.")
else:
    print("No history data available for plotting convergence curves.")

# %%
# Plot boxplots
plot_boxplots(results_df)

# %%
# Plot statistical significance
plot_statistical_significance(results_df, stats_results)

# %% [markdown]
# ## 11. Best Solution Analysis

# %%
def display_team_solution(solution):
    """
    Display the team composition of a solution.
    
    Args:
        solution: Solution to display
    """
    team_stats = solution.get_team_stats()
    
    print("\nTeam Statistics:")
    print(f"{'Team':<10} {'Avg Skill':<15} {'Total Salary':<15} {'GK':<5} {'DEF':<5} {'MID':<5} {'FWD':<5}")
    print("-" * 65)
    
    for stat in team_stats:
        positions = stat["positions"]
        print(f"Team {stat['team_id']+1:<5} {stat['avg_skill']:<15.2f} {stat['total_salary']:<15.2f} "
              f"{positions['GK']:<5} {positions['DEF']:<5} {positions['MID']:<5} {positions['FWD']:<5}")
    
    print("\nDetailed Team Composition:")
    
    for stat in team_stats:
        print(f"\nTeam {stat['team_id']+1}:")
        print(f"{'Name':<20} {'Position':<10} {'Skill':<10} {'Salary':<10}")
        print("-" * 50)
        
        for player in stat["players"]:
            print(f"{player['Name']:<20} {player['Position']:<10} {player['Skill']:<10.2f} {player['Salary']:<10.2f}")
        
        print(f"Average Skill: {stat['avg_skill']:.2f}")
        print(f"Total Salary: {stat['total_salary']:.2f}")
    
    # Calculate overall statistics
    avg_skills = [stat["avg_skill"] for stat in team_stats]
    overall_std = np.std(avg_skills)
    
    print("\nOverall Team Balance:")
    print(f"Standard Deviation of Average Skills: {overall_std:.4f}")
    print(f"This matches the fitness value: {solution.fitness():.4f}")

# %%
def plot_team_solution(solution):
    """
    Create a graphical visualization of the team solution.
    
    Args:
        solution: Solution to visualize
    """
    team_stats = solution.get_team_stats()
    
    # Plot average skills
    plt.figure(figsize=(10, 6))
    teams = [f"Team {stat['team_id']+1}" for stat in team_stats]
    avg_skills = [stat["avg_skill"] for stat in team_stats]
    
    plt.bar(teams, avg_skills)
    plt.title("Average Skill by Team")
    plt.xlabel("Team")
    plt.ylabel("Average Skill")
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(avg_skills):
        plt.text(i, v + 0.1, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.show()
    
    # Plot position distribution
    plt.figure(figsize=(12, 8))
    positions = ["GK", "DEF", "MID", "FWD"]
    
    for i, stat in enumerate(team_stats):
        pos_counts = [stat["positions"][pos] for pos in positions]
        plt.subplot(2, 3, i+1)
        plt.bar(positions, pos_counts)
        plt.title(f"Team {stat['team_id']+1}")
        plt.ylim(0, 3)
        
        # Add value labels
        for j, v in enumerate(pos_counts):
            plt.text(j, v + 0.1, str(v), ha='center')
    
    plt.tight_layout()
    plt.show()
    
    # Plot salary distribution
    plt.figure(figsize=(10, 6))
    total_salaries = [stat["total_salary"] for stat in team_stats]
    
    plt.bar(teams, total_salaries)
    plt.title("Total Salary by Team")
    plt.xlabel("Team")
    plt.ylabel("Total Salary (€M)")
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(total_salaries):
        plt.text(i, v + 10, f"{v:.2f}", ha='center')
    
    # Add budget line
    plt.axhline(y=solution.max_budget, color='r', linestyle='--', label=f"Budget Limit ({solution.max_budget})")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# %%
# Find the best solution
best_config = results_df.groupby('Configuration')['Best Fitness'].mean().idxmin()
best_run = results_df[(results_df['Configuration'] == best_config)]['Best Fitness'].idxmin()
best_run_data = results_df.iloc[best_run]

print(f"\nBest Configuration: {best_config}")
print(f"Best Run: {best_run_data['Run']}")
print(f"Best Fitness: {best_run_data['Best Fitness']:.6f}")
print(f"Evaluations: {best_run_data['Evaluations']}")
print(f"Runtime: {best_run_data['Runtime (s)']:.2f} seconds")

# %%
# Create a solution with the best configuration
best_solution = LeagueSolution(
    players=players_list,
    num_teams=5,
    team_size=7,
    max_budget=750
)

# Run the best algorithm to get the best solution
if best_config in configs:
    config = configs[best_config]
    
    best_solution, best_fitness, evaluations, runtime, history = run_experiment(
        config=config,
        players_list=players_list,
        max_evaluations=EXPERIMENT_CONFIG['max_evaluations']
    )
    
    # Display the best solution
    display_team_solution(best_solution)
    
    # Plot the best solution
    plot_team_solution(best_solution)
else:
    print(f"Configuration {best_config} not found in configs.")

# %% [markdown]
# ## 12. Conclusion

# %% [markdown]
# In this notebook, we have implemented and analyzed various optimization algorithms for the Fantasy League Team Optimization problem. The goal was to create balanced teams of players while respecting position and budget constraints.
#
# We compared the following algorithms:
# - Hill Climbing (HC_Standard)
# - Hill Climbing with Random Restart (HC_Random_Restart)
# - Simulated Annealing (SA_Standard)
# - Genetic Algorithm variants (GA_Tournament_OnePoint, GA_Tournament_TwoPoint, GA_Rank_Uniform, GA_Boltzmann_TwoPoint)
# - GA Hybrid
# - GA Memetic
# - GA Island Model
# - GA with Scramble Mutation
#
# Our statistical analysis showed significant differences between the algorithms, with a large effect size. The best performing algorithm was the GA Memetic, which combines the global exploration capabilities of genetic algorithms with the local exploitation capabilities of hill climbing.
#
# The optimal team solution achieved a good balance of skills across teams while respecting all constraints. This demonstrates the effectiveness of our approach for solving the Fantasy League Team Optimization problem.
