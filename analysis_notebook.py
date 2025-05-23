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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Sports League Optimization: Comparative Analysis of Algorithms
#
# This notebook presents a comprehensive analysis of different optimization algorithms applied to the Sports League problem. We compare Hill Climbing, Simulated Annealing, and Genetic Algorithm approaches, analyzing their performance across multiple metrics.
#
# ## Table of Contents
# 1. [Problem Definition](#1-problem-definition)
# 2. [Experimental Setup](#2-experimental-setup)
# 3. [Algorithm Implementations](#3-algorithm-implementations)
# 4. [Performance Comparison](#4-performance-comparison)
# 5. [Statistical Analysis](#5-statistical-analysis)
# 6. [Conclusion](#6-conclusion)

# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy import stats
import random
from copy import deepcopy
import os
import concurrent.futures
from datetime import datetime

# Import our custom modules
from solution import LeagueSolution, LeagueHillClimbingSolution, LeagueSASolution
from evolution import (
    hill_climbing, 
    simulated_annealing, 
    genetic_algorithm,
    # Mutation operators
    mutate_swap,
    mutate_swap_constrained,
    mutate_team_shift,
    mutate_targeted_player_exchange,
    mutate_shuffle_within_team_constrained,
    # Crossover operators
    crossover_one_point,
    crossover_one_point_prefer_valid,
    crossover_uniform,
    crossover_uniform_prefer_valid,
    # Selection operators
    selection_tournament,
    selection_tournament_variable_k,
    selection_ranking,
    selection_boltzmann
)

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# %% [markdown]
# ## 1. Problem Definition
#
# ### 1.1 Sports League Problem
#
# The Sports League problem involves assigning players to teams while satisfying specific constraints and optimizing for team balance. The goal is to create teams with similar average skill levels.
#
# **Formal Definition:**
# - We have 35 players with different positions (GK, DEF, MID, FWD) and skill levels
# - We need to assign these players to 5 teams (7 players per team)
# - Each team must have exactly 1 GK, 2 DEF, 2 MID, and 2 FWD
# - Each team's total salary must not exceed 750M €
# - The objective is to minimize the standard deviation of average team skills
#
# ### 1.2 Solution Representation
#
# We represent a solution as a list of team assignments for each player. For example, if `solution.repr[0] = 2`, it means player 0 is assigned to team 2.
#
# **Search Space Size:**
# - For 35 players and 5 teams, the theoretical search space is 5^35
# - With constraints, the actual feasible search space is much smaller, but still extremely large
#
# ### 1.3 Fitness Function
#
# The fitness function calculates the standard deviation of the average skill levels across all teams. A lower value indicates more balanced teams, which is our optimization goal.
#
# For invalid solutions (those violating constraints), we return infinity to ensure they are never selected.

# %%
# Load player data
players_df = pd.read_csv("players.csv", sep=";")
# Rename the salary column to match the code expectations
players_df = players_df.rename(columns={'Salary (€M)': 'Salary'})
players_data = players_df.to_dict(orient="records")

# Display the player data
players_df

# %% [markdown]
# ### 1.4 Data Analysis
#
# Let's analyze the player data to understand the distribution of skills, positions, and salaries.

# %%
# Analyze player positions
position_counts = players_df['Position'].value_counts()
print("Position distribution:")
print(position_counts)

# Analyze skill distribution by position
plt.figure(figsize=(12, 6))
sns.boxplot(x='Position', y='Skill', data=players_df)
plt.title('Skill Distribution by Position')
plt.grid(True)
plt.show()

# Analyze salary distribution by position
plt.figure(figsize=(12, 6))
sns.boxplot(x='Position', y='Salary', data=players_df)
plt.title('Salary Distribution by Position')
plt.grid(True)
plt.show()

# Correlation between skill and salary
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Skill', y='Salary', hue='Position', data=players_df)
plt.title('Correlation between Skill and Salary')
plt.grid(True)
plt.show()

# %% [markdown]
# ## 2. Experimental Setup
#
# ### 2.1 Metrics for Comparison
#
# To ensure a fair comparison between different algorithms, we'll track the following metrics:
#
# 1. **Solution Quality**: The fitness value (standard deviation of average team skills)
# 2. **Function Evaluations**: Number of fitness function calls
# 3. **Iterations**: Number of algorithm iterations
# 4. **Runtime**: Actual execution time in seconds
#
# ### 2.2 Algorithm Configurations
#
# We'll test the following algorithm configurations:

# %%
# Define algorithm configurations
configs = {
    # Hill Climbing configurations
    'HC_Standard': {
        'algorithm': 'Hill Climbing',
        'params': {
            'max_iterations': 500,
            'max_no_improvement': 100,
            'verbose': False
        }
    },
    
    # Simulated Annealing configurations
    'SA_Standard': {
        'algorithm': 'Simulated Annealing',
        'params': {
            'initial_temperature': 200.0,
            'cooling_rate': 0.95,
            'min_temperature': 1e-5,
            'iterations_per_temp': 20,
            'verbose': False
        }
    },
    
    # Genetic Algorithm configurations
    'GA_Tournament_OnePoint': {
        'algorithm': 'Genetic Algorithm',
        'params': {
            'population_size': 100,
            'max_generations': 50,
            'selection_operator': selection_tournament,
            'crossover_operator': crossover_one_point_prefer_valid,
            'crossover_rate': 0.8,
            'mutation_operator': mutate_swap_constrained,
            'mutation_rate': 0.1,
            'elitism': True,
            'elitism_size': 2,
            'verbose': False
        }
    },
    'GA_Ranking_Uniform': {
        'algorithm': 'Genetic Algorithm',
        'params': {
            'population_size': 100,
            'max_generations': 50,
            'selection_operator': selection_ranking,
            'crossover_operator': crossover_uniform_prefer_valid,
            'crossover_rate': 0.8,
            'mutation_operator': mutate_targeted_player_exchange,
            'mutation_rate': 0.1,
            'elitism': True,
            'elitism_size': 2,
            'verbose': False
        }
    },
    'GA_Boltzmann_TeamShift': {
        'algorithm': 'Genetic Algorithm',
        'params': {
            'population_size': 100,
            'max_generations': 50,
            'selection_operator': selection_boltzmann,
            'selection_params': {'temperature': 1.0},
            'crossover_operator': crossover_one_point_prefer_valid,
            'crossover_rate': 0.8,
            'mutation_operator': mutate_team_shift,
            'mutation_rate': 0.1,
            'elitism': True,
            'elitism_size': 2,
            'verbose': False
        }
    },
    'GA_Hybrid': {
        'algorithm': 'Hybrid GA',
        'params': {
            'population_size': 75,
            'max_generations': 40,
            'selection_operator': selection_tournament_variable_k,
            'selection_params': {'k': 3},
            'crossover_operator': crossover_uniform_prefer_valid,
            'crossover_rate': 0.85,
            'mutation_operator': mutate_shuffle_within_team_constrained,
            'mutation_rate': 0.15,
            'elitism': True,
            'elitism_size': 1,
            'local_search': {
                'algorithm': 'hill_climbing',
                'frequency': 5,  # Apply HC every 5 generations
                'iterations': 50  # HC iterations per application
            },
            'verbose': False
        }
    }
}

# Display the configurations
for name, config in configs.items():
    print(f"Configuration: {name}")
    print(f"Algorithm: {config['algorithm']}")
    print("Parameters:")
    for param, value in config['params'].items():
        if param not in ['selection_operator', 'crossover_operator', 'mutation_operator', 'verbose']:
            print(f"  {param}: {value}")
    print("")


# %% [markdown]
# ### 2.3 Tracking Function Evaluations
#
# To ensure fair comparison between algorithms, we'll implement a counter for fitness function evaluations:

# %%
# Create a wrapper for the fitness function to count evaluations
class FitnessCounter:
    def __init__(self):
        self.count = 0
        self.original_fitness = LeagueSolution.fitness
        
    def start_counting(self):
        self.count = 0
        LeagueSolution.fitness = self.counting_fitness
        
    def stop_counting(self):
        LeagueSolution.fitness = self.original_fitness
        return self.count
    
    def counting_fitness(self, solution):
        self.count += 1
        return self.original_fitness(solution)

# Initialize the counter
fitness_counter = FitnessCounter()


# %% [markdown]
# ### 2.4 Experiment Runner
#
# We'll create a function to run a single experiment with a specific configuration and run number:

# %%
def run_single_experiment(config_name, config, players_data, run):
    """
    Run a single experiment with a specific configuration and run number.
    
    Args:
        config_name (str): Name of the configuration
        config (dict): Configuration dictionary
        players_data (list): List of player dictionaries
        run (int): Run number (0-based)
        
    Returns:
        dict: Results of the experiment
    """
    # Reset random seed for this run to ensure reproducibility
    random.seed(42 + run)
    np.random.seed(42 + run)
    
    # Create a local fitness counter for this process
    local_counter = FitnessCounter()
    local_counter.start_counting()
    
    # Record start time
    start_time = time.time()
    
    # Run the appropriate algorithm
    if config['algorithm'] == 'Hill Climbing':
        # Create initial solution
        initial_solution = LeagueHillClimbingSolution(players=players_data)
        
        # Run Hill Climbing
        best_solution, best_fitness, history = hill_climbing(
            initial_solution,
            **config['params']
        )
        
        iterations = len(history)
        
    elif config['algorithm'] == 'Simulated Annealing':
        # Create initial solution
        initial_solution = LeagueSASolution(players=players_data)
        
        # Run Simulated Annealing
        best_solution, best_fitness, history = simulated_annealing(
            initial_solution,
            **config['params']
        )
        
        iterations = len(history)
        
    elif config['algorithm'] in ['Genetic Algorithm', 'Hybrid GA']:
        # Run Genetic Algorithm
        best_solution, best_fitness, history = genetic_algorithm(
            players_data,
            **config['params']
        )
        
        iterations = len(history)
    
    # Record end time and calculate runtime
    runtime = time.time() - start_time
    
    # Get number of fitness evaluations
    evaluations = local_counter.stop_counting()
    
    # Return results
    return {
        'Configuration': config_name,
        'Algorithm': config['algorithm'],
        'Run': run + 1,
        'Best Fitness': best_fitness,
        'Iterations': iterations,
        'Function Evaluations': evaluations,
        'Runtime (s)': runtime,
        'History': history
    }


# %% [markdown]
# Now we'll create a function to run all experiments, with options for parallel or sequential execution:

# %%
def run_experiments(configs, players_data, num_runs=5, parallel=True, max_workers=None, save_csv=True):
    """
    Run experiments with all configurations and collect results.
    
    Args:
        configs (dict): Dictionary of configurations
        players_data (list): List of player dictionaries
        num_runs (int): Number of runs per configuration
        parallel (bool): Whether to run experiments in parallel
        max_workers (int): Maximum number of worker processes (None = auto)
        save_csv (bool): Whether to save results to CSV
        
    Returns:
        pandas.DataFrame: Results of all experiments
    """
    all_tasks = []
    
    # Prepare all tasks
    for config_name, config in configs.items():
        for run in range(num_runs):
            all_tasks.append((config_name, config, players_data, run))
    
    results = []
    
    if parallel:
        print(f"Running {len(all_tasks)} experiments in parallel mode...")
        
        # Run experiments in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(run_single_experiment, *task): task for task in all_tasks}
            
            # Process results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(future_to_task)):
                task = future_to_task[future]
                config_name, _, _, run = task
                
                try:
                    result = future.result()
                    results.append(result)
                    print(f"  Completed {config_name} - Run {run + 1}: Fitness = {result['Best Fitness']:.6f}, "
                          f"Evaluations = {result['Function Evaluations']}, Runtime = {result['Runtime (s)']:.2f}s")
                except Exception as e:
                    print(f"  Error in {config_name} - Run {run + 1}: {e}")
    else:
        print(f"Running {len(all_tasks)} experiments in sequential mode...")
        
        # Run experiments sequentially
        for task in all_tasks:
            config_name, config, players_data, run = task
            print(f"Running {config_name} - Run {run + 1}...")
            
            try:
                result = run_single_experiment(config_name, config, players_data, run)
                results.append(result)
                print(f"  Completed: Fitness = {result['Best Fitness']:.6f}, "
                      f"Evaluations = {result['Function Evaluations']}, Runtime = {result['Runtime (s)']:.2f}s")
            except Exception as e:
                print(f"  Error: {e}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV if requested
    if save_csv:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"experiment_results_{timestamp}.csv"
        
        # Create a copy without the history column for CSV export
        export_df = results_df.drop(columns=['History'])
        export_df.to_csv(csv_filename, index=False)
        print(f"Results saved to {csv_filename}")
    
    return results_df


# %% [markdown]
# ## 3. Algorithm Implementations
#
# ### 3.1 Hill Climbing
#
# Hill Climbing is a local search algorithm that starts with an initial solution and iteratively moves to better neighboring solutions until no improvement is possible.
#
# **Key Components:**
# - **Neighborhood Generation**: Defined in `LeagueHillClimbingSolution.get_neighbors()`, which generates valid neighboring solutions by swapping players between teams.
# - **Selection Strategy**: We use steepest ascent, selecting the best neighbor at each iteration.
# - **Termination Criteria**: The algorithm stops when no better neighbor is found or after a maximum number of iterations.
#
# ### 3.2 Simulated Annealing
#
# Simulated Annealing is inspired by the annealing process in metallurgy. It allows accepting worse solutions with a probability that decreases over time, helping to escape local optima.
#
# **Key Components:**
# - **Random Neighbor Generation**: Defined in `LeagueSASolution.get_random_neighbor()`, which generates a random valid neighboring solution.
# - **Acceptance Probability**: Based on the temperature and the fitness difference between the current and new solutions.
# - **Cooling Schedule**: The temperature decreases over time, reducing the probability of accepting worse solutions.
#
# ### 3.3 Genetic Algorithm
#
# Genetic Algorithm is a population-based search algorithm inspired by natural selection and genetics.
#
# **Key Components:**
# - **Selection Operators**: We've implemented three selection mechanisms:
#   - Tournament Selection: Selects the best solution from k random candidates.
#   - Ranking Selection: Selects solutions with probability proportional to their rank.
#   - Boltzmann Selection: Uses Boltzmann distribution to select solutions.
#
# - **Crossover Operators**: We've implemented three crossover operators:
#   - One-Point Crossover: Creates a child by taking a portion from each parent.
#   - One-Point Prefer Valid: Tries multiple cut points to find a valid solution.
#   - Uniform Crossover: Creates a child by randomly selecting genes from either parent.
#
# - **Mutation Operators**: We've implemented four mutation operators:
#   - Swap: Randomly swaps two players between teams.
#   - Swap Constrained: Swaps players of the same position.
#   - Team Shift: Shifts all player assignments by a random number.
#   - Targeted Player Exchange: Swaps players between teams to improve balance.
#   - Shuffle Within Team: Shuffles players within a team with other teams.
#
# - **Elitism**: Preserves the best solutions from one generation to the next.

# %% [markdown]
# ## 4. Performance Comparison
#
# Let's run the experiments and compare the performance of different algorithms:

# %%
# Run experiments with all configurations
# You can choose between parallel (parallel=True) and sequential (parallel=False) execution
# Set max_workers to control the number of parallel processes (None = auto)
# Set save_csv=True to save results to CSV file

results_df = run_experiments(
    configs=configs, 
    players_data=players_data, 
    num_runs=5,
    parallel=True,  # Set to False for sequential execution
    max_workers=None,  # None = auto, or specify a number
    save_csv=True  # Save results to CSV
)

# %% [markdown]
# ### 4.1 Solution Quality Comparison
#
# Let's compare the quality of solutions found by different algorithms:

# %%
# Calculate summary statistics for each configuration
summary = results_df.groupby('Configuration')['Best Fitness'].agg(['mean', 'std', 'min', 'max']).reset_index()
summary = summary.sort_values('mean')

# Display summary statistics
print("Solution Quality Summary (lower is better):")
print(summary)

# Plot solution quality comparison
plt.figure(figsize=(12, 6))
sns.boxplot(x='Configuration', y='Best Fitness', data=results_df, order=summary['Configuration'])
plt.title('Solution Quality Comparison (lower is better)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.2 Computational Efficiency Comparison
#
# Let's compare the computational efficiency of different algorithms:

# %%
# Calculate summary statistics for runtime and function evaluations
runtime_summary = results_df.groupby('Configuration')['Runtime (s)'].mean().reset_index()
runtime_summary = runtime_summary.sort_values('Runtime (s)')

evaluations_summary = results_df.groupby('Configuration')['Function Evaluations'].mean().reset_index()
evaluations_summary = evaluations_summary.sort_values('Function Evaluations')

# Plot runtime comparison
plt.figure(figsize=(12, 6))
sns.barplot(x='Configuration', y='Runtime (s)', data=runtime_summary)
plt.title('Runtime Comparison (lower is better)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot function evaluations comparison
plt.figure(figsize=(12, 6))
sns.barplot(x='Configuration', y='Function Evaluations', data=evaluations_summary)
plt.title('Function Evaluations Comparison (lower is better)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 4.3 Convergence Analysis
#
# Let's analyze the convergence behavior of different algorithms:

# %%
# Plot convergence curves for each algorithm (using the first run)
plt.figure(figsize=(14, 8))

for config_name in configs.keys():
    # Get the first run for this configuration
    run_data = results_df[(results_df['Configuration'] == config_name) & (results_df['Run'] == 1)].iloc[0]
    
    # Plot the convergence curve
    plt.plot(run_data['History'], label=config_name)

plt.title('Convergence Curves (first run)')
plt.xlabel('Iterations')
plt.ylabel('Fitness (lower is better)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot normalized convergence curves (by function evaluations)
plt.figure(figsize=(14, 8))

for config_name in configs.keys():
    # Get the first run for this configuration
    run_data = results_df[(results_df['Configuration'] == config_name) & (results_df['Run'] == 1)].iloc[0]
    
    # Calculate evaluations per iteration
    evals_per_iter = run_data['Function Evaluations'] / len(run_data['History'])
    
    # Create x-axis values (cumulative evaluations)
    x_values = [i * evals_per_iter for i in range(len(run_data['History']))]
    
    # Plot the normalized convergence curve
    plt.plot(x_values, run_data['History'], label=config_name)

plt.title('Normalized Convergence Curves (by function evaluations)')
plt.xlabel('Function Evaluations')
plt.ylabel('Fitness (lower is better)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 5. Statistical Analysis
#
# Let's perform statistical tests to determine if the differences between algorithms are significant. We'll follow a structured decision flow to select the appropriate statistical tests.

# %%
def perform_statistical_analysis(results_df):
    """
    Perform statistical analysis on experiment results following a structured decision flow.
    
    Args:
        results_df (pandas.DataFrame): DataFrame containing experiment results
        
    Returns:
        dict: Dictionary containing statistical test results
    """
    # Step 1: Check if we have at least 2 configurations to compare
    configurations = results_df['Configuration'].unique()
    if len(configurations) < 2:
        print("Not enough configurations to perform statistical analysis.")
        return {}
    
    # Step 2: Determine if we have 2 or more configurations
    if len(configurations) == 2:
        print("\n=== Two-Group Comparison ===")
        return two_group_comparison(results_df, configurations)
    else:
        print("\n=== Multiple-Group Comparison ===")
        return multiple_group_comparison(results_df, configurations)

def two_group_comparison(results_df, configurations):
    """
    Perform statistical comparison between two groups.
    
    Args:
        results_df (pandas.DataFrame): DataFrame containing experiment results
        configurations (array): Array of configuration names
        
    Returns:
        dict: Dictionary containing statistical test results
    """
    # Extract data for each configuration
    group1 = results_df[results_df['Configuration'] == configurations[0]]['Best Fitness'].values
    group2 = results_df[results_df['Configuration'] == configurations[1]]['Best Fitness'].values
    
    # Step 3: Test for normality using Shapiro-Wilk test
    print("Testing for normality (Shapiro-Wilk):")
    _, p_value1 = stats.shapiro(group1)
    _, p_value2 = stats.shapiro(group2)
    print(f"  {configurations[0]}: p-value = {p_value1:.4f} ({'Normal' if p_value1 > 0.05 else 'Non-normal'})")
    print(f"  {configurations[1]}: p-value = {p_value2:.4f} ({'Normal' if p_value2 > 0.05 else 'Non-normal'})")
    
    # Both groups must be normal to use parametric tests
    is_normal = p_value1 > 0.05 and p_value2 > 0.05
    
    results = {}
    
    if is_normal:
        # Step 4a: Test for equal variances using Levene's test
        print("\nTesting for equal variances (Levene's test):")
        _, p_value_var = stats.levene(group1, group2)
        equal_var = p_value_var > 0.05
        print(f"  p-value = {p_value_var:.4f} ({'Equal variances' if equal_var else 'Unequal variances'})")
        
        # Step 5a: Perform t-test (either with equal or unequal variances)
        if equal_var:
            print("\nPerforming Independent t-test (equal variances):")
            t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=True)
            test_name = "Independent t-test"
        else:
            print("\nPerforming Welch's t-test (unequal variances):")
            t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
            test_name = "Welch's t-test"
            
        print(f"  {test_name}: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
        print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
        
        # Calculate effect size (Cohen's d)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        cohen_d = abs(mean1 - mean2) / pooled_std
        
        print(f"  Effect size (Cohen's d): {cohen_d:.4f}")
        
        # Interpret Cohen's d
        if cohen_d < 0.2:
            effect_interpretation = "Negligible effect"
        elif cohen_d < 0.5:
            effect_interpretation = "Small effect"
        elif cohen_d < 0.8:
            effect_interpretation = "Medium effect"
        else:
            effect_interpretation = "Large effect"
            
        print(f"  Interpretation: {effect_interpretation}")
        
        results = {
            'test': test_name,
            'statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': cohen_d,
            'effect_interpretation': effect_interpretation
        }
    else:
        # Step 4b: Perform Mann-Whitney U test (non-parametric)
        print("\nPerforming Mann-Whitney U test (non-parametric):")
        u_stat, p_value = stats.mannwhitneyu(group1, group2)
        print(f"  Mann-Whitney U: U-statistic = {u_stat:.4f}, p-value = {p_value:.4f}")
        print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
        
        # Calculate effect size (r = Z / sqrt(N))
        n1, n2 = len(group1), len(group2)
        n_total = n1 + n2
        
        # Convert U to Z
        mean_u = n1 * n2 / 2
        std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z = (u_stat - mean_u) / std_u
        
        # Calculate effect size r
        r = abs(z) / np.sqrt(n_total)
        
        print(f"  Effect size (r): {r:.4f}")
        
        # Interpret r
        if r < 0.1:
            effect_interpretation = "Negligible effect"
        elif r < 0.3:
            effect_interpretation = "Small effect"
        elif r < 0.5:
            effect_interpretation = "Medium effect"
        else:
            effect_interpretation = "Large effect"
            
        print(f"  Interpretation: {effect_interpretation}")
        
        results = {
            'test': 'Mann-Whitney U',
            'statistic': u_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': r,
            'effect_interpretation': effect_interpretation
        }
    
    return results

def multiple_group_comparison(results_df, configurations):
    """
    Perform statistical comparison between multiple groups.
    
    Args:
        results_df (pandas.DataFrame): DataFrame containing experiment results
        configurations (array): Array of configuration names
        
    Returns:
        dict: Dictionary containing statistical test results
    """
    # Extract data for each configuration
    groups = [results_df[results_df['Configuration'] == config]['Best Fitness'].values 
              for config in configurations]
    
    # Step 3: Test for normality using Shapiro-Wilk test
    print("Testing for normality (Shapiro-Wilk):")
    normality_results = []
    for i, config in enumerate(configurations):
        _, p_value = stats.shapiro(groups[i])
        is_normal = p_value > 0.05
        normality_results.append(is_normal)
        print(f"  {config}: p-value = {p_value:.4f} ({'Normal' if is_normal else 'Non-normal'})")
    
    # All groups must be normal to use parametric tests
    all_normal = all(normality_results)
    
    results = {}
    
    if all_normal:
        # Step 4a: Perform ANOVA (parametric)
        print("\nPerforming One-way ANOVA (parametric):")
        f_stat, p_value = stats.f_oneway(*groups)
        print(f"  ANOVA: F-statistic = {f_stat:.4f}, p-value = {p_value:.4f}")
        print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
        
        # Calculate effect size (Eta-squared)
        # Flatten all groups into a single array
        all_values = np.concatenate(groups)
        grand_mean = np.mean(all_values)
        
        # Calculate sum of squares between groups (SSB)
        ssb = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)
        
        # Calculate sum of squares total (SST)
        sst = sum((x - grand_mean)**2 for x in all_values)
        
        # Calculate Eta-squared
        eta_squared = ssb / sst
        
        print(f"  Effect size (Eta-squared): {eta_squared:.4f}")
        
        # Interpret Eta-squared
        if eta_squared < 0.01:
            effect_interpretation = "Negligible effect"
        elif eta_squared < 0.06:
            effect_interpretation = "Small effect"
        elif eta_squared < 0.14:
            effect_interpretation = "Medium effect"
        else:
            effect_interpretation = "Large effect"
            
        print(f"  Interpretation: {effect_interpretation}")
        
        results = {
            'test': 'One-way ANOVA',
            'statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': eta_squared,
            'effect_interpretation': effect_interpretation
        }
        
        # Step 5a: Perform post-hoc tests if ANOVA is significant
        if p_value < 0.05:
            print("\nPerforming Tukey HSD post-hoc test:")
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            
            # Prepare data for Tukey HSD
            fitness_values = results_df['Best Fitness'].values
            config_labels = results_df['Configuration'].values
            
            # Perform Tukey HSD test
            tukey_results = pairwise_tukeyhsd(fitness_values, config_labels, alpha=0.05)
            print(tukey_results)
            
            # Store significant pairs
            significant_pairs = []
            for i, row in enumerate(tukey_results.summary().data[1:]):
                group1, group2, _, _, _, reject = row
                if reject:
                    significant_pairs.append((group1, group2))
            
            results['post_hoc'] = {
                'test': 'Tukey HSD',
                'significant_pairs': significant_pairs
            }
    else:
        # Step 4b: Perform Kruskal-Wallis test (non-parametric)
        print("\nPerforming Kruskal-Wallis test (non-parametric):")
        h_stat, p_value = stats.kruskal(*groups)
        print(f"  Kruskal-Wallis: H-statistic = {h_stat:.4f}, p-value = {p_value:.4f}")
        print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
        
        # Calculate effect size (Eta-squared for Kruskal-Wallis)
        n_total = sum(len(group) for group in groups)
        eta_squared_h = (h_stat - len(groups) + 1) / (n_total - len(groups))
        eta_squared_h = max(0, eta_squared_h)  # Ensure non-negative
        
        print(f"  Effect size (Eta-squared H): {eta_squared_h:.4f}")
        
        # Interpret Eta-squared H (same thresholds as Eta-squared)
        if eta_squared_h < 0.01:
            effect_interpretation = "Negligible effect"
        elif eta_squared_h < 0.06:
            effect_interpretation = "Small effect"
        elif eta_squared_h < 0.14:
            effect_interpretation = "Medium effect"
        else:
            effect_interpretation = "Large effect"
            
        print(f"  Interpretation: {effect_interpretation}")
        
        results = {
            'test': 'Kruskal-Wallis',
            'statistic': h_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': eta_squared_h,
            'effect_interpretation': effect_interpretation
        }
        
        # Step 5b: Perform post-hoc tests if Kruskal-Wallis is significant
        if p_value < 0.05:
            print("\nPerforming Dunn's test with Bonferroni correction:")
            from scikit_posthocs import posthoc_dunn
            
            # Prepare data for Dunn's test
            dunn_data = {}
            for i, config in enumerate(configurations):
                dunn_data[config] = groups[i]
            
            # Perform Dunn's test
            dunn_results = posthoc_dunn(dunn_data, p_adjust='bonferroni')
            print(dunn_results)
            
            # Store significant pairs
            significant_pairs = []
            for i in range(len(configurations)):
                for j in range(i+1, len(configurations)):
                    if dunn_results.iloc[i, j] < 0.05:
                        significant_pairs.append((configurations[i], configurations[j]))
            
            results['post_hoc'] = {
                'test': "Dunn's test with Bonferroni correction",
                'significant_pairs': significant_pairs
            }
    
    return results

# Perform statistical analysis
statistical_results = perform_statistical_analysis(results_df)


# %% [markdown]
# ### 5.1 Visualizing Statistical Results
#
# Let's create visualizations that incorporate the statistical significance information:

# %%
def plot_with_significance(results_df, statistical_results):
    """
    Create boxplot with statistical significance annotations.
    
    Args:
        results_df (pandas.DataFrame): DataFrame containing experiment results
        statistical_results (dict): Dictionary containing statistical test results
    """
    # Calculate summary statistics for each configuration
    summary = results_df.groupby('Configuration')['Best Fitness'].agg(['mean', 'std', 'min', 'max']).reset_index()
    summary = summary.sort_values('mean')
    
    # Create boxplot
    plt.figure(figsize=(14, 8))
    ax = sns.boxplot(x='Configuration', y='Best Fitness', data=results_df, order=summary['Configuration'])
    
    # Add statistical significance annotations if available
    if 'significant' in statistical_results and statistical_results['significant']:
        if 'post_hoc' in statistical_results:
            # Get the ordered configurations
            ordered_configs = summary['Configuration'].tolist()
            
            # Get significant pairs from post-hoc tests
            significant_pairs = statistical_results['post_hoc']['significant_pairs']
            
            # Add significance bars
            y_max = results_df['Best Fitness'].max()
            y_range = results_df['Best Fitness'].max() - results_df['Best Fitness'].min()
            bar_height = y_range * 0.05
            
            for i, (config1, config2) in enumerate(significant_pairs):
                # Get indices in the ordered list
                idx1 = ordered_configs.index(config1)
                idx2 = ordered_configs.index(config2)
                
                # Ensure idx1 < idx2
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                    config1, config2 = config2, config1
                
                # Calculate bar position
                y_pos = y_max + bar_height * (i + 1)
                
                # Draw the bar
                plt.plot([idx1, idx2], [y_pos, y_pos], 'k-', linewidth=1.5)
                plt.plot([idx1, idx1], [y_pos - bar_height/2, y_pos], 'k-', linewidth=1.5)
                plt.plot([idx2, idx2], [y_pos - bar_height/2, y_pos], 'k-', linewidth=1.5)
                
                # Add asterisk
                plt.text((idx1 + idx2) / 2, y_pos + bar_height/4, '*', ha='center', va='center', fontsize=14)
        else:
            # For two-group comparison, add a single significance indicator
            plt.title(f"Solution Quality Comparison (p = {statistical_results['p_value']:.4f}, {statistical_results['effect_interpretation']})")
    
    plt.title('Solution Quality Comparison with Statistical Significance')
    plt.xlabel('Configuration')
    plt.ylabel('Best Fitness (lower is better)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot with significance annotations
plot_with_significance(results_df, statistical_results)

# %% [markdown]
# ## 6. Conclusion
#
# Based on our experiments and analysis, we can draw the following conclusions:
#
# 1. **Solution Quality**: [To be filled after running experiments]
# 2. **Computational Efficiency**: [To be filled after running experiments]
# 3. **Convergence Behavior**: [To be filled after running experiments]
# 4. **Statistical Significance**: [To be filled after running experiments]
#
# ### 6.1 Best Algorithm for the Sports League Problem
#
# [To be filled after running experiments]
#
# ### 6.2 Trade-offs and Recommendations
#
# [To be filled after running experiments]
#
# ### 6.3 Future Work
#
# 1. Explore more advanced hybrid approaches
# 2. Implement adaptive parameter tuning
# 3. Test with larger problem instances
# 4. Develop more specialized operators for the Sports League problem
