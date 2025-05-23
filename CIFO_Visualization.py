# %% [markdown]
# # CIFO - Visualization and Analysis of Optimization Results
# 
# This notebook is dedicated to visualizing and analyzing the results of various optimization algorithms applied to the Fantasy League Team Optimization problem.

# %% [markdown]
# ## 1. Introduction
# 
# ### 1.1 Problem Overview
# 
# The Fantasy League Team Optimization problem involves creating balanced teams by assigning players to teams while minimizing the standard deviation of average team skill ratings. This is subject to constraints such as team composition requirements, budget limitations, and ensuring each player is assigned to exactly one team.
# 
# ### 1.2 Algorithms Implemented
# 
# We have implemented and compared several optimization algorithms:
# 
# - **Hill Climbing (HC)**: A local search algorithm that iteratively moves to better neighboring solutions
# - **Simulated Annealing (SA)**: A probabilistic technique that can escape local optima
# - **Genetic Algorithm (GA)**: A population-based approach inspired by natural selection
# - **Hybrid GA**: A combination of GA with local search techniques
# 
# ### 1.3 Metrics for Comparison
# 
# To ensure a fair comparison between different algorithms, we track the following metrics:
# 
# 1. **Solution Quality**: The fitness value (standard deviation of average team skills)  
# 2. **Function Evaluations**: Number of fitness function calls  
# 3. **Iterations**: Number of algorithm iterations  
# 4. **Runtime**: Actual execution time in seconds

# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import os
from datetime import datetime
import warnings

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# Suppress specific warnings
warnings.filterwarnings("ignore", message="scipy.stats.shapiro: Input data has range zero.*")

# %% [markdown]
# ## 2. Loading and Preparing Data

# %%
# Find the most recent results file
result_files = [f for f in os.listdir() if f.startswith('experiment_results_') and f.endswith('.csv')]
result_files.sort(reverse=True)  # Sort by name (which includes timestamp)

if result_files:
    latest_result_file = result_files[0]
    print(f"Loading most recent results file: {latest_result_file}")
    # Read CSV without specifying index_col to ensure all columns are properly recognized
    results_df = pd.read_csv(latest_result_file)
    print(f"Columns in the CSV: {results_df.columns.tolist()}")
else:
    print("No results files found. Please run the algorithms first.")
    results_df = None

# Find the corresponding history data file
history_files = [f for f in os.listdir() if f.startswith('history_data_') and f.endswith('.npy')]
history_files.sort(reverse=True)  # Sort by name (which includes timestamp)

if history_files:
    latest_history_file = history_files[0]
    print(f"Loading most recent history file: {latest_history_file}")
    try:
        # Load the history data with allow_pickle=True
        history_data = np.load(latest_history_file, allow_pickle=True).item()
        
        # Debug the structure of history_data
        print("\nHistory data structure:")
        for config in history_data:
            print(f"Configuration: {config}")
            print(f"  Type: {type(history_data[config])}")
            if isinstance(history_data[config], list):
                print(f"  Number of runs: {len(history_data[config])}")
                for i, run in enumerate(history_data[config]):
                    print(f"    Run {i+1} type: {type(run)}")
                    if hasattr(run, '__len__'):
                        print(f"    Run {i+1} length: {len(run)}")
                    else:
                        print(f"    Run {i+1} value: {run} (not a sequence)")
            else:
                print(f"  Value: {history_data[config]}")
        
        # Fix history_data structure if needed
        fixed_history_data = {}
        for config in history_data:
            if not isinstance(history_data[config], list):
                # If not a list, convert to a list with a single empty list
                fixed_history_data[config] = [[]]
            else:
                fixed_runs = []
                for run in history_data[config]:
                    if not hasattr(run, '__len__'):
                        # If run is not a sequence, convert to a list with that single value
                        fixed_runs.append([float(run)])
                    else:
                        fixed_runs.append(run)
                fixed_history_data[config] = fixed_runs
        
        # Replace with fixed structure
        history_data = fixed_history_data
        print("\nFixed history data structure.")
    except Exception as e:
        print(f"Error loading history data: {e}")
        history_data = None
else:
    print("No history files found. Please run the algorithms first.")
    history_data = None

# Display basic information about the results
if results_df is not None:
    print("\nResults summary:")
    print(f"Number of configurations: {results_df['Configuration'].nunique()}")
    print(f"Number of runs per configuration: {results_df.groupby('Configuration').size().iloc[0]}")
    print(f"Total number of experiments: {len(results_df)}")
    
    # Check for errors in the results
    if 'Error' in results_df.columns:
        error_count = results_df['Error'].notna().sum()
        if error_count > 0:
            print(f"\nWarning: {error_count} experiments encountered errors.")
            print("Error summary:")
            print(results_df[results_df['Error'].notna()].groupby(['Configuration', 'Error']).size())

# %% [markdown]
# ## 3. Basic Performance Analysis

# %%
# Function to create a summary dataframe with mean and std for each configuration
def create_summary_df(results_df):
    if results_df is None:
        return None
    
    # Define column mappings for flexibility
    column_mappings = {
        'fitness': 'Best Fitness',
        'evaluations': 'Function Evaluations' if 'Function Evaluations' in results_df.columns else 'Evaluations',
        'time': 'Runtime (s)' if 'Runtime (s)' in results_df.columns else 'Time',
        'valid': 'Valid' if 'Valid' in results_df.columns else None
    }
    
    # Prepare aggregation dictionary
    agg_dict = {
        column_mappings['fitness']: ['mean', 'std', 'min', 'max'],
        column_mappings['evaluations']: ['mean', 'std'],
        column_mappings['time']: ['mean', 'std']
    }
    
    # Add Valid column if it exists
    if column_mappings['valid'] is not None and column_mappings['valid'] in results_df.columns:
        agg_dict[column_mappings['valid']] = 'mean'
    
    # Group by Configuration and calculate statistics
    summary = results_df.groupby('Configuration').agg(agg_dict)
    
    # Flatten the multi-index columns
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    # Reset index to make Configuration a column
    summary = summary.reset_index()
    
    # Sort by mean fitness (ascending for minimization problems)
    summary = summary.sort_values(f"{column_mappings['fitness']}_mean")
    
    return summary

# Create and display summary dataframe
summary_df = create_summary_df(results_df)
if summary_df is not None:
    print("Performance summary by configuration:")
    print(summary_df)

# %% [markdown]
# ## 4. Visualization of Results

# %%
# Function to plot fitness comparison across configurations
def plot_fitness_comparison(summary_df, title="Fitness Comparison Across Configurations"):
    if summary_df is None:
        return
    
    # Identify the fitness column
    fitness_col = [col for col in summary_df.columns if col.endswith('_mean') and 'Fitness' in col][0]
    std_col = [col for col in summary_df.columns if col.endswith('_std') and 'Fitness' in col][0]
    
    plt.figure(figsize=(14, 8))
    
    # Create bar plot
    ax = sns.barplot(x='Configuration', y=fitness_col, data=summary_df, 
                    hue='Configuration', legend=False)
    
    # Add error bars
    ax.errorbar(x=range(len(summary_df)), y=summary_df[fitness_col], 
               yerr=summary_df[std_col], fmt='none', color='black', capsize=5)
    
    # Customize plot
    plt.title(title, fontsize=16)
    plt.xlabel('Configuration', fontsize=14)
    plt.ylabel('Mean Fitness (lower is better)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add value labels on top of bars
    for i, v in enumerate(summary_df[fitness_col]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=10)
    
    return ax

# Function to plot evaluation count comparison
def plot_evaluations_comparison(summary_df, title="Function Evaluations Comparison"):
    if summary_df is None:
        return
    
    # Identify the evaluations column
    evals_col = [col for col in summary_df.columns if col.endswith('_mean') and ('Evaluations' in col or 'Function' in col)][0]
    std_col = [col for col in summary_df.columns if col.endswith('_std') and ('Evaluations' in col or 'Function' in col)][0]
    
    plt.figure(figsize=(14, 8))
    
    # Create bar plot
    ax = sns.barplot(x='Configuration', y=evals_col, data=summary_df, 
                    hue='Configuration', legend=False)
    
    # Add error bars
    ax.errorbar(x=range(len(summary_df)), y=summary_df[evals_col], 
               yerr=summary_df[std_col], fmt='none', color='black', capsize=5)
    
    # Customize plot
    plt.title(title, fontsize=16)
    plt.xlabel('Configuration', fontsize=14)
    plt.ylabel('Mean Number of Function Evaluations', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add value labels on top of bars
    for i, v in enumerate(summary_df[evals_col]):
        ax.text(i, v + 0.01, f"{int(v)}", ha='center', fontsize=10)
    
    return ax

# Function to plot execution time comparison
def plot_time_comparison(summary_df, title="Execution Time Comparison"):
    if summary_df is None:
        return
    
    # Identify the time column
    time_col = [col for col in summary_df.columns if col.endswith('_mean') and ('Time' in col or 'Runtime' in col)][0]
    std_col = [col for col in summary_df.columns if col.endswith('_std') and ('Time' in col or 'Runtime' in col)][0]
    
    plt.figure(figsize=(14, 8))
    
    # Create bar plot
    ax = sns.barplot(x='Configuration', y=time_col, data=summary_df, 
                    hue='Configuration', legend=False)
    
    # Add error bars
    ax.errorbar(x=range(len(summary_df)), y=summary_df[time_col], 
               yerr=summary_df[std_col], fmt='none', color='black', capsize=5)
    
    # Customize plot
    plt.title(title, fontsize=16)
    plt.xlabel('Configuration', fontsize=14)
    plt.ylabel('Mean Execution Time (seconds)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add value labels on top of bars
    for i, v in enumerate(summary_df[time_col]):
        ax.text(i, v + 0.01, f"{v:.2f}s", ha='center', fontsize=10)
    
    return ax

# Plot fitness comparison
if summary_df is not None:
    plot_fitness_comparison(summary_df)
    plot_evaluations_comparison(summary_df)
    plot_time_comparison(summary_df)

# %% [markdown]
# ## 5. Convergence Analysis
# 
# ### 5.1 Convergence Curves by Iteration
# 
# The following graph shows the convergence curves for each algorithm configuration across multiple runs. Each line represents a single execution (run) of a specific algorithm configuration. This visualization helps us understand:
# 
# - **Convergence Speed**: How quickly each algorithm reaches its best solution
# - **Convergence Stability**: Whether the algorithm consistently improves or fluctuates
# - **Final Solution Quality**: The fitness value achieved at the end of execution
# - **Run Variability**: How performance varies across different runs of the same algorithm
# 
# The x-axis represents iterations (generations for GA, temperature steps for SA, improvement attempts for HC), while the y-axis shows the fitness value (lower is better).

# %%
# Function to plot convergence curves for all configurations
def plot_convergence_curves(history_data, title="Convergence Curves by Run"):
    if history_data is None:
        print("No history data available for plotting convergence curves.")
        return
    
    plt.figure(figsize=(14, 10))
    
    # Define a color map for different configurations
    config_names = list(history_data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(config_names)))
    
    for i, config_name in enumerate(config_names):
        histories = history_data[config_name]
        
        # Plot each run with a different line style
        for j, history in enumerate(histories):
            # Skip if history is not a sequence or is empty
            if not hasattr(history, '__len__') or len(history) == 0:
                continue
                
            # Use different line styles for different runs
            line_style = ['-', '--', '-.', ':'][j % 4]
            plt.plot(history, color=colors[i], linestyle=line_style, alpha=0.7, 
                     label=f"{config_name} (Run {j+1})" if j == 0 else None)
    
    # Customize plot
    plt.title(title, fontsize=16)
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Fitness (lower is better)', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gca()

# Plot convergence curves
if history_data is not None:
    plot_convergence_curves(history_data, "Convergence Curves by Run")

# %% [markdown]
# ### 5.2 Average Convergence Curves
# 
# The graph below shows the average convergence behavior for each algorithm configuration. For each configuration, we calculate the mean fitness value at each iteration across all runs, with shaded areas representing the standard deviation. This visualization provides:
# 
# - **Average Performance**: The typical convergence pattern for each algorithm
# - **Performance Consistency**: The width of the shaded area indicates how consistent the algorithm is across runs
# - **Comparative Analysis**: Easier comparison between different algorithms without the visual clutter of individual runs
# 
# This aggregated view helps identify which algorithms consistently perform better on average.

# %%
# Function to plot average convergence curves
def plot_average_convergence(history_data, title="Average Convergence Curves"):
    if history_data is None:
        print("No history data available for plotting average convergence curves.")
        return
    
    plt.figure(figsize=(14, 10))
    
    # Define a color map for different configurations
    config_names = list(history_data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(config_names)))
    
    # Find the maximum length of histories
    max_len = 0
    for config_name in config_names:
        for history in history_data[config_name]:
            if hasattr(history, '__len__'):
                if len(history) > max_len:
                    max_len = len(history)
    
    for i, config_name in enumerate(config_names):
        histories = history_data[config_name]
        
        # Pad histories to the same length
        padded_histories = []
        for history in histories:
            if hasattr(history, '__len__') and len(history) > 0:  # Only include valid histories
                # Pad with the last value
                padded = list(history)
                if len(padded) < max_len:
                    padded.extend([padded[-1]] * (max_len - len(padded)))
                padded_histories.append(padded)
        
        if padded_histories:  # Only proceed if we have valid histories
            # Convert to numpy array for easier manipulation
            histories_array = np.array(padded_histories)
            
            # Calculate mean and std
            mean_history = np.mean(histories_array, axis=0)
            std_history = np.std(histories_array, axis=0)
            
            # Plot mean with confidence interval
            x = np.arange(len(mean_history))
            plt.plot(x, mean_history, color=colors[i], label=config_name)
            plt.fill_between(x, mean_history - std_history, mean_history + std_history, 
                            color=colors[i], alpha=0.2)
    
    # Customize plot
    plt.title(title, fontsize=16)
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Fitness (lower is better)', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gca()

# Plot average convergence curves
if history_data is not None:
    plot_average_convergence(history_data)

# %% [markdown]
# ### 5.3 Convergence Curves Normalized by Function Evaluations
# 
# This visualization shows convergence curves normalized by the number of function evaluations rather than iterations. Function evaluations represent the computational cost of running each algorithm, providing a fairer comparison between different approaches.
# 
# **What are Function Evaluations?**
# 
# Function evaluations count how many times the fitness function is calculated during optimization:
# - For Hill Climbing: Each neighbor evaluation counts as one function evaluation
# - For Simulated Annealing: Each candidate solution evaluation counts as one function evaluation
# - For Genetic Algorithms: Each individual in each generation requires a function evaluation
# 
# Normalizing by function evaluations helps us understand which algorithms are most efficient in terms of computational resources, showing which methods achieve better solutions with fewer fitness calculations.

# %%
# Function to plot convergence curves normalized by function evaluations
def plot_normalized_convergence(history_data, results_df, title="Convergence Curves Normalized by Function Evaluations"):
    if history_data is None or results_df is None:
        print("Missing data for plotting normalized convergence curves.")
        return
    
    # Identify the evaluations column
    evals_col = [col for col in results_df.columns if 'Evaluations' in col or 'Function' in col]
    if not evals_col:
        print("Function evaluations column not found in results dataframe")
        return
    evals_col = evals_col[0]
    
    plt.figure(figsize=(14, 10))
    
    # Define a color map for different configurations
    config_names = list(history_data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(config_names)))
    
    for i, config_name in enumerate(config_names):
        histories = history_data[config_name]
        
        # Get evaluations for this configuration
        config_results = results_df[results_df['Configuration'] == config_name]
        
        # Plot each run with a different line style
        for j, history in enumerate(histories):
            # Skip if history is not a sequence or is empty
            if not hasattr(history, '__len__') or len(history) == 0:
                continue
                
            if j < len(config_results):  # Only plot if we have evaluation data
                # Get total evaluations for this run
                total_evals = config_results.iloc[j][evals_col]
                
                # Create normalized x-axis
                x = np.linspace(0, 1, len(history)) * total_evals / 1e6  # Normalize to millions
                
                # Use different line styles for different runs
                line_style = ['-', '--', '-.', ':'][j % 4]
                plt.plot(x, history, color=colors[i], linestyle=line_style, alpha=0.7, 
                         label=f"{config_name} (Run {j+1})" if j == 0 else None)
    
    # Customize plot
    plt.title(title, fontsize=16)
    plt.xlabel('Number of Function Evaluations (millions)', fontsize=14)
    plt.ylabel('Fitness (lower is better)', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gca()

# Plot normalized convergence curves
if history_data is not None and results_df is not None:
    plot_normalized_convergence(history_data, results_df)

# %% [markdown]
# ## 6. Genetic Algorithm Specific Analysis
# 
# This section focuses specifically on analyzing the performance of different Genetic Algorithm configurations. We compare various GA variants to understand the impact of different selection methods, crossover operators, mutation rates, and other parameters on solution quality.

# %%
# Function to filter and analyze only GA configurations
def analyze_ga_configurations(results_df, history_data):
    if results_df is None or history_data is None:
        print("Missing data for GA configuration analysis.")
        return
    
    # Filter only GA configurations
    ga_configs = [config for config in results_df['Configuration'].unique() 
                 if config.startswith('GA_')]
    
    if not ga_configs:
        print("No Genetic Algorithm configurations found in the results.")
        return
    
    # Filter results and history data
    ga_results = results_df[results_df['Configuration'].isin(ga_configs)]
    ga_history = {config: history_data[config] for config in ga_configs if config in history_data}
    
    # Create summary for GA configurations
    ga_summary = create_summary_df(ga_results)
    
    # Plot GA-specific comparisons
    plt.figure(figsize=(14, 8))
    ax = plot_fitness_comparison(ga_summary, "Fitness Comparison Across GA Configurations")
    plt.figure(figsize=(14, 10))
    ax = plot_average_convergence(ga_history, "Average Convergence Curves for GA Configurations")
    
    # Analyze impact of different GA parameters
    
    # 1. Impact of selection method
    selection_configs = {
        'Tournament': [c for c in ga_configs if 'Tournament' in c and not any(x in c for x in ['Low', 'High', 'Small', 'Large', 'Valid', 'Repair'])],
        'Rank': [c for c in ga_configs if 'Rank' in c],
        'Boltzmann': [c for c in ga_configs if 'Boltzmann' in c]
    }
    
    # Filter out empty categories
    selection_configs = {k: v for k, v in selection_configs.items() if v}
    
    if len(selection_configs) > 1:
        plt.figure(figsize=(14, 8))
        plt.title("Impact of Selection Method on Fitness", fontsize=16)
        
        selection_results = []
        for method, configs in selection_configs.items():
            method_results = ga_results[ga_results['Configuration'].isin(configs)]
            method_results['Selection'] = method
            selection_results.append(method_results)
        
        selection_df = pd.concat(selection_results)
        sns.boxplot(x='Selection', y='Best Fitness', data=selection_df)
        plt.xlabel('Selection Method', fontsize=14)
        plt.ylabel('Fitness (lower is better)', fontsize=14)
        plt.tight_layout()
    
    # 2. Impact of crossover type
    crossover_configs = {
        'One Point': [c for c in ga_configs if 'OnePoint' in c and not any(x in c for x in ['Low', 'High', 'Small', 'Large', 'Valid', 'Repair'])],
        'Two Point': [c for c in ga_configs if 'TwoPoint' in c and not any(x in c for x in ['Low', 'High', 'Small', 'Large', 'Valid', 'Repair'])],
        'Uniform': [c for c in ga_configs if 'Uniform' in c]
    }
    
    # Filter out empty categories
    crossover_configs = {k: v for k, v in crossover_configs.items() if v}
    
    if len(crossover_configs) > 1:
        plt.figure(figsize=(14, 8))
        plt.title("Impact of Crossover Type on Fitness", fontsize=16)
        
        crossover_results = []
        for method, configs in crossover_configs.items():
            method_results = ga_results[ga_results['Configuration'].isin(configs)]
            method_results['Crossover'] = method
            crossover_results.append(method_results)
        
        crossover_df = pd.concat(crossover_results)
        sns.boxplot(x='Crossover', y='Best Fitness', data=crossover_df)
        plt.xlabel('Crossover Type', fontsize=14)
        plt.ylabel('Fitness (lower is better)', fontsize=14)
        plt.tight_layout()
    
    # 3. Impact of mutation rate
    mutation_configs = {
        'Low': [c for c in ga_configs if 'Low_Mutation' in c],
        'Standard': [c for c in ga_configs if c == 'GA_Tournament_TwoPoint'],
        'High': [c for c in ga_configs if 'High_Mutation' in c]
    }
    
    # Filter out empty categories
    mutation_configs = {k: v for k, v in mutation_configs.items() if v}
    
    if len(mutation_configs) > 1:
        plt.figure(figsize=(14, 8))
        plt.title("Impact of Mutation Rate on Fitness", fontsize=16)
        
        mutation_results = []
        for rate, configs in mutation_configs.items():
            rate_results = ga_results[ga_results['Configuration'].isin(configs)]
            rate_results['Mutation Rate'] = rate
            mutation_results.append(rate_results)
        
        mutation_df = pd.concat(mutation_results)
        sns.boxplot(x='Mutation Rate', y='Best Fitness', data=mutation_df)
        plt.xlabel('Mutation Rate', fontsize=14)
        plt.ylabel('Fitness (lower is better)', fontsize=14)
        plt.tight_layout()
    
    # 4. Impact of elitism
    elitism_configs = {
        'None': [c for c in ga_configs if 'No_Elitism' in c],
        'Standard': [c for c in ga_configs if c == 'GA_Tournament_TwoPoint'],
        'High': [c for c in ga_configs if 'High_Elitism' in c]
    }
    
    # Filter out empty categories
    elitism_configs = {k: v for k, v in elitism_configs.items() if v}
    
    if len(elitism_configs) > 1:
        plt.figure(figsize=(14, 8))
        plt.title("Impact of Elitism on Fitness", fontsize=16)
        
        elitism_results = []
        for level, configs in elitism_configs.items():
            level_results = ga_results[ga_results['Configuration'].isin(configs)]
            level_results['Elitism'] = level
            elitism_results.append(level_results)
        
        elitism_df = pd.concat(elitism_results)
        sns.boxplot(x='Elitism', y='Best Fitness', data=elitism_df)
        plt.xlabel('Elitism Level', fontsize=14)
        plt.ylabel('Fitness (lower is better)', fontsize=14)
        plt.tight_layout()
    
    # 5. Impact of population size
    population_configs = {
        'Small': [c for c in ga_configs if 'Small_Population' in c],
        'Standard': [c for c in ga_configs if c == 'GA_Tournament_TwoPoint'],
        'Large': [c for c in ga_configs if 'Large_Population' in c]
    }
    
    # Filter out empty categories
    population_configs = {k: v for k, v in population_configs.items() if v}
    
    if len(population_configs) > 1:
        plt.figure(figsize=(14, 8))
        plt.title("Impact of Population Size on Fitness", fontsize=16)
        
        population_results = []
        for size, configs in population_configs.items():
            size_results = ga_results[ga_results['Configuration'].isin(configs)]
            size_results['Population Size'] = size
            population_results.append(size_results)
        
        population_df = pd.concat(population_results)
        sns.boxplot(x='Population Size', y='Best Fitness', data=population_df)
        plt.xlabel('Population Size', fontsize=14)
        plt.ylabel('Fitness (lower is better)', fontsize=14)
        plt.tight_layout()
    
    return ga_summary

# Analyze GA configurations
if results_df is not None and history_data is not None:
    ga_summary = analyze_ga_configurations(results_df, history_data)

# %% [markdown]
# ## 7. Algorithm Type Comparison
# 
# This section compares the performance of different algorithm types (Hill Climbing, Simulated Annealing, Genetic Algorithms, and Hybrid approaches) to understand their relative strengths and weaknesses for the Fantasy League Team Optimization problem.

# %%
# Function to compare different algorithm types
def compare_algorithm_types(results_df, history_data):
    if results_df is None or history_data is None:
        print("Missing data for algorithm type comparison.")
        return
    
    # Define algorithm types
    algorithm_types = {
        'Hill Climbing': [c for c in results_df['Configuration'].unique() if c.startswith('HC_')],
        'Simulated Annealing': [c for c in results_df['Configuration'].unique() if c.startswith('SA_')],
        'Genetic Algorithm': [c for c in results_df['Configuration'].unique() if c.startswith('GA_') and 'Hybrid' not in c],
        'Hybrid GA': [c for c in results_df['Configuration'].unique() if 'Hybrid' in c]
    }
    
    # Filter out empty categories
    algorithm_types = {k: v for k, v in algorithm_types.items() if v}
    
    if len(algorithm_types) > 1:
        plt.figure(figsize=(14, 8))
        plt.title("Comparison of Algorithm Types", fontsize=16)
        
        algorithm_results = []
        for algo_type, configs in algorithm_types.items():
            type_results = results_df[results_df['Configuration'].isin(configs)]
            type_results['Algorithm Type'] = algo_type
            algorithm_results.append(type_results)
        
        algorithm_df = pd.concat(algorithm_results)
        
        # Box plot for fitness comparison
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='Algorithm Type', y='Best Fitness', data=algorithm_df)
        plt.title("Fitness Comparison by Algorithm Type", fontsize=16)
        plt.xlabel('Algorithm Type', fontsize=14)
        plt.ylabel('Fitness (lower is better)', fontsize=14)
        plt.tight_layout()
        
        # Box plot for evaluations comparison
        # Identify the evaluations column
        evals_col = [col for col in results_df.columns if 'Evaluations' in col or 'Function' in col]
        if evals_col:
            evals_col = evals_col[0]
            plt.figure(figsize=(14, 8))
            sns.boxplot(x='Algorithm Type', y=evals_col, data=algorithm_df)
            plt.title("Function Evaluations by Algorithm Type", fontsize=16)
            plt.xlabel('Algorithm Type', fontsize=14)
            plt.ylabel('Number of Function Evaluations', fontsize=14)
            plt.tight_layout()
        
        # Box plot for time comparison
        # Identify the time column
        time_col = [col for col in results_df.columns if 'Time' in col or 'Runtime' in col]
        if time_col:
            time_col = time_col[0]
            plt.figure(figsize=(14, 8))
            sns.boxplot(x='Algorithm Type', y=time_col, data=algorithm_df)
            plt.title("Execution Time by Algorithm Type", fontsize=16)
            plt.xlabel('Algorithm Type', fontsize=14)
            plt.ylabel('Execution Time (seconds)', fontsize=14)
            plt.tight_layout()
        
        # Average convergence curves by algorithm type
        plt.figure(figsize=(14, 10))
        plt.title("Average Convergence by Algorithm Type", fontsize=16)
        
        # Define a color map for different algorithm types
        colors = plt.cm.tab10(np.linspace(0, 1, len(algorithm_types)))
        
        # Find the maximum length of histories
        max_len = 0
        for algo_type, configs in algorithm_types.items():
            for config in configs:
                if config in history_data:
                    for history in history_data[config]:
                        if hasattr(history, '__len__') and len(history) > max_len:
                            max_len = len(history)
        
        for i, (algo_type, configs) in enumerate(algorithm_types.items()):
            # Collect all histories for this algorithm type
            all_histories = []
            for config in configs:
                if config in history_data:
                    for history in history_data[config]:
                        if hasattr(history, '__len__') and len(history) > 0:  # Only include valid histories
                            # Pad with the last value
                            padded = list(history)
                            if len(padded) < max_len:
                                padded.extend([padded[-1]] * (max_len - len(padded)))
                            all_histories.append(padded)
            
            if all_histories:  # Only proceed if we have valid histories
                # Convert to numpy array for easier manipulation
                histories_array = np.array(all_histories)
                
                # Calculate mean and std
                mean_history = np.mean(histories_array, axis=0)
                std_history = np.std(histories_array, axis=0)
                
                # Plot mean with confidence interval
                x = np.arange(len(mean_history))
                plt.plot(x, mean_history, color=colors[i], label=algo_type)
                plt.fill_between(x, mean_history - std_history, mean_history + std_history, 
                                color=colors[i], alpha=0.2)
        
        # Customize plot
        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel('Fitness (lower is better)', fontsize=14)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return algorithm_df
    
    return None

# Compare algorithm types
if results_df is not None and history_data is not None:
    algorithm_comparison = compare_algorithm_types(results_df, history_data)

# %% [markdown]
# ## 8. Statistical Analysis
# 
# This section performs statistical tests to determine if there are significant differences between the performance of different algorithm configurations. We use ANOVA to test for overall differences, followed by post-hoc Tukey HSD tests to identify which specific configurations differ significantly from each other.

# %%
# Function to perform statistical analysis on the results
def perform_statistical_analysis(results_df):
    if results_df is None:
        print("No results data available for statistical analysis.")
        return None
    
    try:
        # Import statistical libraries
        from scipy import stats
        import statsmodels.stats.multicomp as mc
        
        # Get unique configurations
        configs = results_df['Configuration'].unique()
        
        if len(configs) < 2:
            print("Need at least two configurations for statistical comparison.")
            return None
        
        # Prepare data for analysis
        valid_results = []
        valid_configs = []
        
        for config in configs:
            group = results_df[results_df['Configuration'] == config]['Best Fitness'].values
            group = group[~np.isnan(group)]  # Remove NaNs
            group = group[~np.isinf(group)]  # Remove infinities
            
            if len(group) > 0:
                valid_results.append(group)
                valid_configs.append(config)
        
        if len(valid_configs) < 2:
            print("Need at least two configurations with valid data for statistical comparison.")
            return None
        
        # Check for normality
        print("\n=== Multiple-Group Comparison ===")
        normality_results = {}
        for i, config in enumerate(valid_configs):
            # Suppress specific warnings about zero range
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Input data has range zero.*")
                if len(set(valid_results[i])) > 1:  # Only perform test if there's variation
                    _, p_value = stats.shapiro(valid_results[i])
                else:
                    p_value = 0  # Automatically fail normality test if all values are identical
                
            normality_results[config] = p_value
            print(f"Shapiro-Wilk normality test p-value for {config}: {p_value:.4f}")
        
        # Decide on parametric or non-parametric test
        all_normal = all(p > 0.05 for p in normality_results.values())
        
        if all_normal:
            print("Data appears normally distributed, using one-way ANOVA")
            f_val, p_val = stats.f_oneway(*valid_results)
            
            # Calculate effect size (Eta-squared)
            # Sum of squares between groups
            grand_mean = np.mean([np.mean(group) for group in valid_results])
            ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in valid_results)
            
            # Total sum of squares
            all_values = np.concatenate(valid_results)
            ss_total = sum((x - grand_mean)**2 for x in all_values)
            
            # Eta-squared
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            # Interpret effect size
            if eta_squared < 0.01:
                effect_size_interp = "Negligible"
            elif eta_squared < 0.06:
                effect_size_interp = "Small"
            elif eta_squared < 0.14:
                effect_size_interp = "Medium"
            else:
                effect_size_interp = "Large"
            
            print(f"One-way ANOVA p-value: {p_val:.4f}")
            print(f"Effect size (Eta-squared): {eta_squared:.4f} ({effect_size_interp})")
            print(f"Significant difference: {p_val < 0.05}")
            
            if p_val < 0.05:
                # Perform Tukey HSD post-hoc test
                print("\n=== Post-hoc Tests ===")
                
                # Prepare data for Tukey test
                data = []
                labels = []
                for i, config in enumerate(valid_configs):
                    data.extend(valid_results[i])
                    labels.extend([config] * len(valid_results[i]))
                
                # Create a MultiComparison object
                mc_obj = mc.MultiComparison(data, labels)
                
                # Perform Tukey HSD test
                try:
                    tukey_results = mc_obj.tukeyhsd(alpha=0.05)
                    print(tukey_results)
                    
                    # Extract significant pairs
                    significant_pairs = []
                    for i, reject in enumerate(tukey_results.reject):
                        if reject:
                            pair = (tukey_results.groupsunique[tukey_results.pairindices[i, 0]], 
                                   tukey_results.groupsunique[tukey_results.pairindices[i, 1]])
                            significant_pairs.append(pair)
                    
                    return {
                        'test': 'ANOVA',
                        'p_value': p_val,
                        'effect_size': eta_squared,
                        'effect_size_interp': effect_size_interp,
                        'post_hoc': 'Tukey HSD',
                        'significant_pairs': significant_pairs
                    }
                except Exception as e:
                    print(f"Error in Tukey HSD test: {e}")
                    return {
                        'test': 'ANOVA',
                        'p_value': p_val,
                        'effect_size': eta_squared,
                        'effect_size_interp': effect_size_interp,
                        'significant': True,
                        'error': str(e)
                    }
            
            else:
                return {
                    'test': 'ANOVA',
                    'p_value': p_val,
                    'effect_size': eta_squared,
                    'effect_size_interp': effect_size_interp,
                    'significant': False
                }
        
        else:
            print("Data not normally distributed, using Kruskal-Wallis H-test")
            h_val, p_val = stats.kruskal(*valid_results)
            
            # Calculate effect size (Eta-squared)
            n = sum(len(group) for group in valid_results)
            eta_squared = (h_val - len(valid_results) + 1) / (n - len(valid_results)) if n > len(valid_results) else 0
            
            # Interpret effect size
            if eta_squared < 0.01:
                effect_size_interp = "Negligible"
            elif eta_squared < 0.06:
                effect_size_interp = "Small"
            elif eta_squared < 0.14:
                effect_size_interp = "Medium"
            else:
                effect_size_interp = "Large"
            
            print(f"Kruskal-Wallis H-test p-value: {p_val:.4f}")
            print(f"Effect size (Eta-squared): {eta_squared:.4f} ({effect_size_interp})")
            print(f"Significant difference: {p_val < 0.05}")
            
            if p_val < 0.05:
                # Perform Dunn's post-hoc test
                print("\n=== Post-hoc Tests ===")
                
                # Prepare data for Dunn's test
                data = []
                labels = []
                for i, config in enumerate(valid_configs):
                    data.extend(valid_results[i])
                    labels.extend([config] * len(valid_results[i]))
                
                # Convert to DataFrame for easier analysis
                dunn_data = pd.DataFrame({'Fitness': data, 'Configuration': labels})
                
                try:
                    # Perform Dunn's test
                    from scikit_posthocs import posthoc_dunn
                    dunn_results = posthoc_dunn(dunn_data, val_col='Fitness', group_col='Configuration', p_adjust='bonferroni')
                    print(dunn_results)
                    
                    # Extract significant pairs
                    significant_pairs = []
                    for i in range(len(valid_configs)):
                        for j in range(i+1, len(valid_configs)):
                            if dunn_results.iloc[i, j] < 0.05:
                                pair = (valid_configs[i], valid_configs[j])
                                significant_pairs.append(pair)
                    
                    return {
                        'test': 'Kruskal-Wallis',
                        'p_value': p_val,
                        'effect_size': eta_squared,
                        'effect_size_interp': effect_size_interp,
                        'post_hoc': "Dunn's Test",
                        'significant_pairs': significant_pairs
                    }
                except Exception as e:
                    print(f"Error in Dunn's test: {e}")
                    return {
                        'test': 'Kruskal-Wallis',
                        'p_value': p_val,
                        'effect_size': eta_squared,
                        'effect_size_interp': effect_size_interp,
                        'significant': True,
                        'error': str(e)
                    }
            
            else:
                return {
                    'test': 'Kruskal-Wallis',
                    'p_value': p_val,
                    'effect_size': eta_squared,
                    'effect_size_interp': effect_size_interp,
                    'significant': False
                }
    except Exception as e:
        print(f"Error in statistical analysis: {e}")
        return None

# Perform statistical analysis
if results_df is not None:
    try:
        statistical_results = perform_statistical_analysis(results_df)
    except Exception as e:
        print(f"Error in statistical analysis: {e}")
        statistical_results = None

# %% [markdown]
# ## 9. Rationale for Algorithm Configuration Selection
# 
# The selection of these specific algorithm configurations represents a strategic approach to solving the Fantasy League Team Optimization problem, with each configuration designed to explore different search strategies and mechanisms. Here's a detailed explanation of the rationale behind each selection:

# %% [markdown]
# ### 9.1 Hill Climbing Variants
# 
# #### HC_Standard
# - **Purpose**: Serves as a baseline local search algorithm that follows a greedy improvement strategy.
# - **Configuration Details**:
#   - Maximum of 500 iterations to ensure sufficient exploration of the search space.
#   - Stops after 100 iterations without improvement to prevent wasting computational resources.
#   - Simple yet effective for problems with relatively smooth fitness landscapes.
#   - Provides a performance benchmark for more complex algorithms.
# 
# #### HC_Valid_Initial
# - **Purpose**: Tests the hypothesis that starting from a valid solution improves overall performance.
# - **Configuration Details**:
#   - Identical parameters to HC_Standard (500 max iterations, 100 iterations without improvement).
#   - The key difference is initialization with a valid team composition.
#   - Expected to reach better solutions faster by avoiding constraint violations from the beginning.
#   - Particularly valuable in highly constrained problems like team selection.

# %% [markdown]
# ### 9.2 Simulated Annealing
# 
# #### SA_Standard
# - **Purpose**: Overcomes local optima limitations of Hill Climbing through probabilistic acceptance.
# - **Configuration Details**:
#   - High initial temperature (200.0) enables extensive exploration early in the search.
#   - Gradual cooling rate (0.95) provides a balanced transition from exploration to exploitation.
#   - Multiple iterations per temperature (20) allows thorough sampling at each temperature level.
#   - Minimum temperature threshold (1e-5) ensures the algorithm eventually converges.
#   - Particularly effective for rugged fitness landscapes with many local optima.

# %% [markdown]
# ### 9.3 Genetic Algorithm Variants
# 
# #### GA_Tournament_OnePoint
# - **Purpose**: Implements a classic GA with tournament selection and one-point crossover.
# - **Configuration Details**:
#   - Tournament selection provides adjustable selection pressure.
#   - One-point crossover preserves contiguous building blocks of good solutions.
#   - Population size of 100 balances diversity and computational efficiency.
#   - Crossover rate of 0.8 favors recombination while still allowing some solutions to pass unchanged.
#   - Mutation rate of 0.1 provides sufficient exploration without disrupting too many good solutions.
#   - Small elitism (2 individuals) preserves the best solutions across generations.
# 
# #### GA_Ranking_Uniform
# - **Purpose**: Tests an alternative selection and recombination strategy.
# - **Configuration Details**:
#   - Ranking selection provides more uniform selection pressure than tournament selection.
#   - Uniform crossover enables more thorough mixing of parental traits.
#   - Particularly effective when beneficial traits are distributed throughout the solution.
#   - Same population size, crossover/mutation rates, and elitism as GA_Tournament_OnePoint.
#   - Allows direct comparison of selection and crossover operator effects.
# 
# #### GA_Boltzmann_TeamShift
# - **Purpose**: Implements temperature-based selection with domain-specific mutation.
# - **Configuration Details**:
#   - Boltzmann selection dynamically adjusts selection pressure based on population diversity.
#   - Temperature parameter (1.0) controls the selection pressure.
#   - Team shift mutation operator makes domain-specific changes to team composition.
#   - Combines theoretical advantages of adaptive selection with problem-specific mutation.
#   - Expected to perform well in later generations when fine-tuning is required.

# %% [markdown]
# ### 9.4 Hybrid Approach
# 
# #### GA_Hybrid
# - **Purpose**: Combines global search (GA) with local search (Hill Climbing) for enhanced performance.
# - **Configuration Details**:
#   - Smaller population (75) compensated by intensive local search.
#   - Local search applied every 5 generations to refine promising solutions.
#   - Tournament selection with size 3 for moderate selection pressure.
#   - One-point crossover with preference for valid solutions to maintain feasibility.
#   - Higher mutation rate (0.15) and targeted player exchange for intelligent exploration.
#   - Reduced elitism (1 individual) since local search helps preserve good solutions.
#   - Represents a memetic algorithm approach that leverages strengths of both global and local search.

# %% [markdown]
# ## 10. Conclusion and Recommendations

# %%
# Function to generate conclusions and recommendations based on the analysis
def generate_conclusions(results_df, history_data):
    if results_df is None or history_data is None:
        print("Missing data for generating conclusions.")
        return
    
    print("\n## Conclusions and Recommendations")
    
    # Find the best configuration
    best_config = results_df.groupby('Configuration')['Best Fitness'].mean().idxmin()
    best_fitness = results_df.groupby('Configuration')['Best Fitness'].mean().min()
    
    print(f"\n### Best Configuration")
    print(f"The best performing configuration is **{best_config}** with an average fitness of {best_fitness:.4f}.")
    
    # Analyze algorithm types
    algorithm_types = {
        'Hill Climbing': [c for c in results_df['Configuration'].unique() if c.startswith('HC_')],
        'Simulated Annealing': [c for c in results_df['Configuration'].unique() if c.startswith('SA_')],
        'Genetic Algorithm': [c for c in results_df['Configuration'].unique() if c.startswith('GA_') and 'Hybrid' not in c],
        'Hybrid GA': [c for c in results_df['Configuration'].unique() if 'Hybrid' in c]
    }
    
    # Filter out empty categories and calculate mean fitness
    algo_performance = {}
    for algo_type, configs in algorithm_types.items():
        if configs:
            mean_fitness = results_df[results_df['Configuration'].isin(configs)]['Best Fitness'].mean()
            algo_performance[algo_type] = mean_fitness
    
    if algo_performance:
        best_algo = min(algo_performance.items(), key=lambda x: x[1])[0]
        print(f"\n### Algorithm Type Comparison")
        print(f"The best performing algorithm type is **{best_algo}** with an average fitness of {algo_performance[best_algo]:.4f}.")
        
        # Print all algorithm types in order of performance
        print("\nAlgorithm types ranked by performance (best to worst):")
        for i, (algo, fitness) in enumerate(sorted(algo_performance.items(), key=lambda x: x[1])):
            print(f"{i+1}. {algo}: {fitness:.4f}")
    
    # Analyze GA-specific parameters if we have GA configurations
    ga_configs = [c for c in results_df['Configuration'].unique() if c.startswith('GA_')]
    if ga_configs:
        print("\n### Genetic Algorithm Parameters")
        
        # Selection methods
        selection_configs = {
            'Tournament': [c for c in ga_configs if 'Tournament' in c and not any(x in c for x in ['Low', 'High', 'Small', 'Large', 'Valid', 'Repair'])],
            'Rank': [c for c in ga_configs if 'Rank' in c],
            'Boltzmann': [c for c in ga_configs if 'Boltzmann' in c]
        }
        
        selection_performance = {}
        for method, configs in selection_configs.items():
            if configs:
                mean_fitness = results_df[results_df['Configuration'].isin(configs)]['Best Fitness'].mean()
                selection_performance[method] = mean_fitness
        
        if len(selection_performance) > 1:
            best_selection = min(selection_performance.items(), key=lambda x: x[1])[0]
            print(f"\n**Selection Method**: The best performing selection method is **{best_selection}** with an average fitness of {selection_performance[best_selection]:.4f}.")
        
        # Crossover types
        crossover_configs = {
            'One Point': [c for c in ga_configs if 'OnePoint' in c and not any(x in c for x in ['Low', 'High', 'Small', 'Large', 'Valid', 'Repair'])],
            'Two Point': [c for c in ga_configs if 'TwoPoint' in c and not any(x in c for x in ['Low', 'High', 'Small', 'Large', 'Valid', 'Repair'])],
            'Uniform': [c for c in ga_configs if 'Uniform' in c]
        }
        
        crossover_performance = {}
        for method, configs in crossover_configs.items():
            if configs:
                mean_fitness = results_df[results_df['Configuration'].isin(configs)]['Best Fitness'].mean()
                crossover_performance[method] = mean_fitness
        
        if len(crossover_performance) > 1:
            best_crossover = min(crossover_performance.items(), key=lambda x: x[1])[0]
            print(f"\n**Crossover Type**: The best performing crossover type is **{best_crossover}** with an average fitness of {crossover_performance[best_crossover]:.4f}.")
        
        # Mutation rates
        mutation_configs = {
            'Low': [c for c in ga_configs if 'Low_Mutation' in c],
            'Standard': [c for c in ga_configs if c == 'GA_Tournament_TwoPoint'],
            'High': [c for c in ga_configs if 'High_Mutation' in c]
        }
        
        mutation_performance = {}
        for rate, configs in mutation_configs.items():
            if configs:
                mean_fitness = results_df[results_df['Configuration'].isin(configs)]['Best Fitness'].mean()
                mutation_performance[rate] = mean_fitness
        
        if len(mutation_performance) > 1:
            best_mutation = min(mutation_performance.items(), key=lambda x: x[1])[0]
            print(f"\n**Mutation Rate**: The best performing mutation rate is **{best_mutation}** with an average fitness of {mutation_performance[best_mutation]:.4f}.")
        
        # Elitism levels
        elitism_configs = {
            'None': [c for c in ga_configs if 'No_Elitism' in c],
            'Standard': [c for c in ga_configs if c == 'GA_Tournament_TwoPoint'],
            'High': [c for c in ga_configs if 'High_Elitism' in c]
        }
        
        elitism_performance = {}
        for level, configs in elitism_configs.items():
            if configs:
                mean_fitness = results_df[results_df['Configuration'].isin(configs)]['Best Fitness'].mean()
                elitism_performance[level] = mean_fitness
        
        if len(elitism_performance) > 1:
            best_elitism = min(elitism_performance.items(), key=lambda x: x[1])[0]
            print(f"\n**Elitism Level**: The best performing elitism level is **{best_elitism}** with an average fitness of {elitism_performance[best_elitism]:.4f}.")
        
        # Population sizes
        population_configs = {
            'Small': [c for c in ga_configs if 'Small_Population' in c],
            'Standard': [c for c in ga_configs if c == 'GA_Tournament_TwoPoint'],
            'Large': [c for c in ga_configs if 'Large_Population' in c]
        }
        
        population_performance = {}
        for size, configs in population_configs.items():
            if configs:
                mean_fitness = results_df[results_df['Configuration'].isin(configs)]['Best Fitness'].mean()
                population_performance[size] = mean_fitness
        
        if len(population_performance) > 1:
            best_population = min(population_performance.items(), key=lambda x: x[1])[0]
            print(f"\n**Population Size**: The best performing population size is **{best_population}** with an average fitness of {population_performance[best_population]:.4f}.")
        
        # Valid solutions focus
        valid_configs = {
            'Standard': [c for c in ga_configs if c == 'GA_Tournament_TwoPoint'],
            'Valid Initial': [c for c in ga_configs if 'Valid_Initial' in c],
            'Repair': [c for c in ga_configs if 'Repair' in c]
        }
        
        valid_performance = {}
        for approach, configs in valid_configs.items():
            if configs:
                mean_fitness = results_df[results_df['Configuration'].isin(configs)]['Best Fitness'].mean()
                valid_performance[approach] = mean_fitness
        
        if len(valid_performance) > 1:
            best_valid = min(valid_performance.items(), key=lambda x: x[1])[0]
            print(f"\n**Valid Solutions Approach**: The best performing approach is **{best_valid}** with an average fitness of {valid_performance[best_valid]:.4f}.")
    
    # Analyze statistical significance
    if 'statistical_results' in globals() and statistical_results:
        print("\n### Statistical Significance")
        
        if statistical_results.get('significant', True):  # Default to True if key not present
            test_name = statistical_results['test']
            p_value = statistical_results['p_value']
            effect_size = statistical_results['effect_size']
            effect_size_interp = statistical_results['effect_size_interp']
            
            print(f"The {test_name} test shows statistically significant differences between configurations (p = {p_value:.4f}).")
            print(f"The effect size is {effect_size:.4f}, which is considered {effect_size_interp}.")
            
            if 'significant_pairs' in statistical_results:
                post_hoc = statistical_results['post_hoc']
                significant_pairs = statistical_results['significant_pairs']
                
                print(f"\nThe {post_hoc} post-hoc analysis identified {len(significant_pairs)} significantly different pairs:")
                for pair in significant_pairs:
                    print(f"- {pair[0]} vs {pair[1]}")
        else:
            test_name = statistical_results['test']
            p_value = statistical_results['p_value']
            
            print(f"The {test_name} test shows no statistically significant differences between configurations (p = {p_value:.4f}).")
    
    # Final recommendations
    print("\n### Final Recommendations")
    
    # Best overall configuration
    print(f"1. **Best Overall Configuration**: {best_config} with average fitness {best_fitness:.4f}")
    
    # Best algorithm type
    if algo_performance:
        print(f"2. **Recommended Algorithm Type**: {best_algo}")
    
    # GA-specific recommendations
    if ga_configs:
        print("3. **Genetic Algorithm Recommendations**:")
        
        if len(selection_performance) > 1:
            print(f"   - Selection Method: {best_selection}")
        
        if len(crossover_performance) > 1:
            print(f"   - Crossover Type: {best_crossover}")
        
        if len(mutation_performance) > 1:
            print(f"   - Mutation Rate: {best_mutation}")
        
        if len(elitism_performance) > 1:
            print(f"   - Elitism Level: {best_elitism}")
        
        if len(population_performance) > 1:
            print(f"   - Population Size: {best_population}")
        
        if len(valid_performance) > 1:
            print(f"   - Valid Solutions Approach: {best_valid}")
    
    # Note about GA_Boltzmann_TeamShift
    if 'GA_Boltzmann_TeamShift' in results_df['Configuration'].unique():
        boltzmann_fitness = results_df[results_df['Configuration'] == 'GA_Boltzmann_TeamShift']['Best Fitness'].mean()
        print(f"\n**Note on GA_Boltzmann_TeamShift**: This algorithm shows a constant fitness value ({boltzmann_fitness:.4f}) after initial iterations, suggesting potential implementation issues or premature convergence. Further investigation is recommended.")
    
    # Computational efficiency
    # Identify the evaluations column
    evals_col = [col for col in results_df.columns if 'Evaluations' in col or 'Function' in col]
    if evals_col:
        evals_col = evals_col[0]
        most_efficient_config = results_df.groupby('Configuration')[evals_col].mean().idxmin()
        most_efficient_evals = results_df.groupby('Configuration')[evals_col].mean().min()
        
        print(f"\n4. **Computational Efficiency**:")
        print(f"   - Most Evaluation-Efficient: {most_efficient_config} ({most_efficient_evals:.0f} evaluations)")
    
    # Identify the time column
    time_col = [col for col in results_df.columns if 'Time' in col or 'Runtime' in col]
    if time_col:
        time_col = time_col[0]
        fastest_config = results_df.groupby('Configuration')[time_col].mean().idxmin()
        fastest_time = results_df.groupby('Configuration')[time_col].mean().min()
        print(f"   - Fastest Algorithm: {fastest_config} ({fastest_time:.2f} seconds)")
    
    # Trade-off recommendation
    print(f"\n5. **Trade-off Recommendation**:")
    
    # Calculate a simple score based on normalized fitness and time
    if time_col:
        norm_fitness = results_df.groupby('Configuration')['Best Fitness'].mean() / results_df['Best Fitness'].mean()
        norm_time = results_df.groupby('Configuration')[time_col].mean() / results_df[time_col].mean()
        
        # Lower is better for both metrics
        trade_off_score = norm_fitness + 0.5 * norm_time
        best_trade_off = trade_off_score.idxmin()
        
        print(f"   - Best Balance of Quality and Speed: {best_trade_off}")
    
    # Future work
    print("\n### Future Work")
    print("1. Investigate the constant behavior of GA_Boltzmann_TeamShift")
    print("2. Experiment with hybrid approaches combining the best features of different algorithms")
    print("3. Test additional parameter settings for the best performing configurations")
    print("4. Implement adaptive parameter control for genetic algorithms")
    print("5. Explore multi-objective optimization to balance team cost and performance")

# Generate conclusions and recommendations
if results_df is not None and history_data is not None:
    generate_conclusions(results_df, history_data)

# %% [markdown]
# ## 11. Investigation of GA_Boltzmann_TeamShift Behavior
# 
# This section specifically investigates the constant behavior observed in the GA_Boltzmann_TeamShift algorithm. We analyze the convergence pattern and potential implementation issues that might be causing this behavior.

# %%
# Function to investigate GA_Boltzmann_TeamShift behavior
def investigate_boltzmann_teamshift(results_df, history_data):
    if results_df is None or history_data is None:
        print("Missing data for GA_Boltzmann_TeamShift investigation.")
        return
    
    # Check if GA_Boltzmann_TeamShift exists in the data
    if 'GA_Boltzmann_TeamShift' not in results_df['Configuration'].unique():
        print("GA_Boltzmann_TeamShift configuration not found in the results.")
        return
    
    print("## Investigation of GA_Boltzmann_TeamShift Behavior")
    
    # Get GA_Boltzmann_TeamShift results
    boltzmann_results = results_df[results_df['Configuration'] == 'GA_Boltzmann_TeamShift']
    
    # Basic statistics
    print("\n### Basic Statistics")
    print(f"Number of runs: {len(boltzmann_results)}")
    print(f"Mean fitness: {boltzmann_results['Best Fitness'].mean():.4f}")
    print(f"Standard deviation: {boltzmann_results['Best Fitness'].std():.4f}")
    print(f"Min fitness: {boltzmann_results['Best Fitness'].min():.4f}")
    print(f"Max fitness: {boltzmann_results['Best Fitness'].max():.4f}")
    
    # Check if all runs converge to the same value
    if boltzmann_results['Best Fitness'].std() < 1e-6:
        print("\nAll runs converge to exactly the same fitness value, suggesting a deterministic behavior or implementation issue.")
    
    # Plot convergence curves for GA_Boltzmann_TeamShift
    if 'GA_Boltzmann_TeamShift' in history_data:
        plt.figure(figsize=(14, 8))
        
        # Plot each run
        for i, history in enumerate(history_data['GA_Boltzmann_TeamShift']):
            if hasattr(history, '__len__') and len(history) > 0:
                plt.plot(history, label=f"Run {i+1}")
        
        plt.title("GA_Boltzmann_TeamShift Convergence Curves", fontsize=16)
        plt.xlabel("Iterations", fontsize=14)
        plt.ylabel("Fitness (lower is better)", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Analyze the convergence pattern
        print("\n### Convergence Analysis")
        
        # Check for early convergence
        early_convergence = []
        for history in history_data['GA_Boltzmann_TeamShift']:
            if hasattr(history, '__len__') and len(history) > 5:
                # Check if the fitness value becomes constant after a few iterations
                constant_after = None
                for i in range(1, len(history) - 1):
                    if abs(history[i] - history[i+1]) < 1e-6:
                        if constant_after is None:
                            constant_after = i
                    else:
                        constant_after = None
                
                if constant_after is not None:
                    early_convergence.append(constant_after)
        
        if early_convergence:
            avg_convergence = sum(early_convergence) / len(early_convergence)
            print(f"Algorithm converges to a constant value after an average of {avg_convergence:.1f} iterations.")
            print("This suggests premature convergence or an implementation issue in the selection or mutation operators.")
        
        # Compare with other GA variants
        print("\n### Comparison with Other GA Variants")
        
        ga_configs = [c for c in results_df['Configuration'].unique() 
                     if c.startswith('GA_') and c != 'GA_Boltzmann_TeamShift']
        
        if ga_configs:
            # Calculate improvement percentage for each GA variant
            improvement_data = []
            
            for config in ga_configs + ['GA_Boltzmann_TeamShift']:
                if config in history_data:
                    for run, history in enumerate(history_data[config]):
                        if hasattr(history, '__len__') and len(history) > 1:
                            initial = history[0]
                            final = history[-1]
                            improvement = (initial - final) / initial * 100 if initial > 0 else 0
                            improvement_data.append({
                                'Configuration': config,
                                'Run': run + 1,
                                'Initial Fitness': initial,
                                'Final Fitness': final,
                                'Improvement (%)': improvement
                            })
            
            if improvement_data:
                improvement_df = pd.DataFrame(improvement_data)
                
                # Group by configuration and calculate statistics
                improvement_stats = improvement_df.groupby('Configuration')['Improvement (%)'].agg(['mean', 'std', 'min', 'max']).reset_index()
                
                # Display improvement statistics
                print("\nImprovement Statistics (% reduction in fitness value):")
                print(improvement_stats)
                
                # Plot improvement comparison
                plt.figure(figsize=(14, 8))
                sns.barplot(x='Configuration', y='Improvement (%)', data=improvement_df)
                plt.title("Fitness Improvement Comparison", fontsize=16)
                plt.xlabel("Configuration", fontsize=14)
                plt.ylabel("Improvement (%)", fontsize=14)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
        
        # Potential issues and recommendations
        print("\n### Potential Issues and Recommendations")
        print("1. **Selection Pressure**: The Boltzmann selection may have too high or too low temperature parameter.")
        print("2. **Mutation Operator**: The TeamShift mutation might not be providing enough diversity.")
        print("3. **Premature Convergence**: The population might be converging too quickly to a suboptimal solution.")
        print("4. **Implementation Bug**: There might be an issue in how the algorithm is implemented.")
        
        print("\nRecommendations:")
        print("1. Verify the implementation of the Boltzmann selection operator")
        print("2. Test different temperature parameters for the Boltzmann selection")
        print("3. Modify the TeamShift mutation to provide more exploration")
        print("4. Implement diversity preservation mechanisms")
        print("5. Consider a restart mechanism when the algorithm stagnates")

# Investigate GA_Boltzmann_TeamShift behavior
if results_df is not None and history_data is not None:
    investigate_boltzmann_teamshift(results_df, history_data)

# %% [markdown]
# ## 12. Experimental Setup
# 
# ### 12.1 Metrics for Comparison
# 
# To ensure a fair comparison between different algorithms, we track the following metrics:
# 
# 1. **Solution Quality**: The fitness value (standard deviation of average team skills)  
# 2. **Function Evaluations**: Number of fitness function calls  
# 3. **Iterations**: Number of algorithm iterations  
# 4. **Runtime**: Actual execution time in seconds  
# 
# ### 12.2 Rationale for Algorithm Configuration Selection
# 
# The selection of these specific algorithm configurations represents a strategic approach to solving the Fantasy League Team Optimization problem, with each configuration designed to explore different search strategies and mechanisms. Here's a detailed explanation of the rationale behind each selection:
# 
# #### Hill Climbing Variants
# 
# ##### HC_Standard
# - **Purpose**: Serves as a baseline local search algorithm that follows a greedy improvement strategy.
# - **Configuration Details**:
#   - Maximum of 500 iterations to ensure sufficient exploration of the search space.
#   - Stops after 100 iterations without improvement to prevent wasting computational resources.
#   - Simple yet effective for problems with relatively smooth fitness landscapes.
#   - Provides a performance benchmark for more complex algorithms.
# 
# ##### HC_Valid_Initial
# - **Purpose**: Tests the hypothesis that starting from a valid solution improves overall performance.
# - **Configuration Details**:
#   - Identical parameters to HC_Standard (500 max iterations, 100 iterations without improvement).
#   - The key difference is initialization with a valid team composition.
#   - Expected to reach better solutions faster by avoiding constraint violations from the beginning.
#   - Particularly valuable in highly constrained problems like team selection.
# 
# #### Simulated Annealing
# 
# ##### SA_Standard
# - **Purpose**: Overcomes local optima limitations of Hill Climbing through probabilistic acceptance.
# - **Configuration Details**:
#   - High initial temperature (200.0) enables extensive exploration early in the search.
#   - Gradual cooling rate (0.95) provides a balanced transition from exploration to exploitation.
#   - Multiple iterations per temperature (20) allows thorough sampling at each temperature level.
#   - Minimum temperature threshold (1e-5) ensures the algorithm eventually converges.
#   - Particularly effective for rugged fitness landscapes with many local optima.
# 
# #### Genetic Algorithm Variants
# 
# ##### GA_Tournament_OnePoint
# - **Purpose**: Implements a classic GA with tournament selection and one-point crossover.
# - **Configuration Details**:
#   - Tournament selection provides adjustable selection pressure.
#   - One-point crossover preserves contiguous building blocks of good solutions.
#   - Population size of 100 balances diversity and computational efficiency.
#   - Crossover rate of 0.8 favors recombination while still allowing some solutions to pass unchanged.
#   - Mutation rate of 0.1 provides sufficient exploration without disrupting too many good solutions.
#   - Small elitism (2 individuals) preserves the best solutions across generations.
# 
# ##### GA_Ranking_Uniform
# - **Purpose**: Tests an alternative selection and recombination strategy.
# - **Configuration Details**:
#   - Ranking selection provides more uniform selection pressure than tournament selection.
#   - Uniform crossover enables more thorough mixing of parental traits.
#   - Particularly effective when beneficial traits are distributed throughout the solution.
#   - Same population size, crossover/mutation rates, and elitism as GA_Tournament_OnePoint.
#   - Allows direct comparison of selection and crossover operator effects.
# 
# ##### GA_Boltzmann_TeamShift
# - **Purpose**: Implements temperature-based selection with domain-specific mutation.
# - **Configuration Details**:
#   - Boltzmann selection dynamically adjusts selection pressure based on population diversity.
#   - Temperature parameter (1.0) controls the selection pressure.
#   - Team shift mutation operator makes domain-specific changes to team composition.
#   - Combines theoretical advantages of adaptive selection with problem-specific mutation.
#   - Expected to perform well in later generations when fine-tuning is required.
# 
# #### Hybrid Approach
# 
# ##### GA_Hybrid
# - **Purpose**: Combines global search (GA) with local search (Hill Climbing) for enhanced performance.
# - **Configuration Details**:
#   - Smaller population (75) compensated by intensive local search.
#   - Local search applied every 5 generations to refine promising solutions.
#   - Tournament selection with size 3 for moderate selection pressure.
#   - One-point crossover with preference for valid solutions to maintain feasibility.
#   - Higher mutation rate (0.15) and targeted player exchange for intelligent exploration.
#   - Reduced elitism (1 individual) since local search helps preserve good solutions.
#   - Represents a memetic algorithm approach that leverages strengths of both global and local search.
# 
# This carefully designed set of algorithms provides a comprehensive evaluation framework, ranging from simple local search methods to sophisticated hybrid approaches, allowing for robust comparison and insights into which strategies work best for the Fantasy League Team Optimization problem.
