# %% [markdown]
# # CIFO - Visualization and Analysis of Optimization Results
# 
# This notebook is dedicated to visualizing and analyzing the results of various optimization algorithms applied to the Fantasy League Team Optimization problem.

# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import os
from datetime import datetime

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# %% [markdown]
# ## 1. Loading and Preparing Data

# %%
# Find the most recent results file
result_files = [f for f in os.listdir() if f.startswith('experiment_results_') and f.endswith('.csv')]
result_files.sort(reverse=True)  # Sort by name (which includes timestamp)

if result_files:
    latest_result_file = result_files[0]
    print(f"Loading most recent results file: {latest_result_file}")
    results_df = pd.read_csv(latest_result_file)
else:
    print("No results files found. Please run the algorithms first.")
    results_df = None

# Find the corresponding history data file
history_files = [f for f in os.listdir() if f.startswith('history_data_') and f.endswith('.npy')]
history_files.sort(reverse=True)  # Sort by name (which includes timestamp)

if history_files:
    latest_history_file = history_files[0]
    print(f"Loading most recent history file: {latest_history_file}")
    history_data = np.load(latest_history_file, allow_pickle=True).item()
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
# ## 2. Basic Performance Analysis

# %%
# Function to create a summary dataframe with mean and std for each configuration
def create_summary_df(results_df):
    if results_df is None:
        return None
    
    # Group by Configuration and calculate statistics
    summary = results_df.groupby('Configuration').agg({
        'Best Fitness': ['mean', 'std', 'min', 'max'],
        'Evaluations': ['mean', 'std'],
        'Time': ['mean', 'std'],
        'Valid': 'mean'
    })
    
    # Flatten the multi-index columns
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    # Reset index to make Configuration a column
    summary = summary.reset_index()
    
    # Sort by mean fitness (ascending for minimization problems)
    summary = summary.sort_values('Best Fitness_mean')
    
    return summary

# Create and display summary dataframe
summary_df = create_summary_df(results_df)
if summary_df is not None:
    print("Performance summary by configuration:")
    display(summary_df)

# %% [markdown]
# ## 3. Visualization of Results

# %%
# Function to plot fitness comparison across configurations
def plot_fitness_comparison(summary_df, title="Fitness Comparison Across Configurations"):
    if summary_df is None:
        return
    
    plt.figure(figsize=(14, 8))
    
    # Create bar plot
    ax = sns.barplot(x='Configuration', y='Best Fitness_mean', data=summary_df, 
                    hue='Configuration', legend=False)
    
    # Add error bars
    ax.errorbar(x=range(len(summary_df)), y=summary_df['Best Fitness_mean'], 
               yerr=summary_df['Best Fitness_std'], fmt='none', color='black', capsize=5)
    
    # Customize plot
    plt.title(title, fontsize=16)
    plt.xlabel('Configuration', fontsize=14)
    plt.ylabel('Mean Fitness (lower is better)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add value labels on top of bars
    for i, v in enumerate(summary_df['Best Fitness_mean']):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=10)
    
    return ax

# Function to plot evaluation count comparison
def plot_evaluations_comparison(summary_df, title="Function Evaluations Comparison"):
    if summary_df is None:
        return
    
    plt.figure(figsize=(14, 8))
    
    # Create bar plot
    ax = sns.barplot(x='Configuration', y='Evaluations_mean', data=summary_df, 
                    hue='Configuration', legend=False)
    
    # Add error bars
    ax.errorbar(x=range(len(summary_df)), y=summary_df['Evaluations_mean'], 
               yerr=summary_df['Evaluations_std'], fmt='none', color='black', capsize=5)
    
    # Customize plot
    plt.title(title, fontsize=16)
    plt.xlabel('Configuration', fontsize=14)
    plt.ylabel('Mean Number of Function Evaluations', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add value labels on top of bars
    for i, v in enumerate(summary_df['Evaluations_mean']):
        ax.text(i, v + 0.01, f"{int(v)}", ha='center', fontsize=10)
    
    return ax

# Function to plot execution time comparison
def plot_time_comparison(summary_df, title="Execution Time Comparison"):
    if summary_df is None:
        return
    
    plt.figure(figsize=(14, 8))
    
    # Create bar plot
    ax = sns.barplot(x='Configuration', y='Time_mean', data=summary_df, 
                    hue='Configuration', legend=False)
    
    # Add error bars
    ax.errorbar(x=range(len(summary_df)), y=summary_df['Time_mean'], 
               yerr=summary_df['Time_std'], fmt='none', color='black', capsize=5)
    
    # Customize plot
    plt.title(title, fontsize=16)
    plt.xlabel('Configuration', fontsize=14)
    plt.ylabel('Mean Execution Time (seconds)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add value labels on top of bars
    for i, v in enumerate(summary_df['Time_mean']):
        ax.text(i, v + 0.01, f"{v:.2f}s", ha='center', fontsize=10)
    
    return ax

# Plot fitness comparison
if summary_df is not None:
    plot_fitness_comparison(summary_df)
    plot_evaluations_comparison(summary_df)
    plot_time_comparison(summary_df)

# %% [markdown]
# ## 4. Convergence Analysis

# %%
# Function to plot convergence curves for all configurations
def plot_convergence_curves(history_data, title="Convergence Curves"):
    if history_data is None:
        return
    
    plt.figure(figsize=(14, 10))
    
    # Define a color map for different configurations
    config_names = list(history_data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(config_names)))
    
    for i, config_name in enumerate(config_names):
        histories = history_data[config_name]
        
        # Plot each run with a different line style
        for j, history in enumerate(histories):
            if len(history) > 0:  # Only plot if history is not empty
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

# Function to plot average convergence curves
def plot_average_convergence(history_data, title="Average Convergence Curves"):
    if history_data is None:
        return
    
    plt.figure(figsize=(14, 10))
    
    # Define a color map for different configurations
    config_names = list(history_data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(config_names)))
    
    # Find the maximum length of histories
    max_len = 0
    for config_name in config_names:
        for history in history_data[config_name]:
            if len(history) > max_len:
                max_len = len(history)
    
    for i, config_name in enumerate(config_names):
        histories = history_data[config_name]
        
        # Pad histories to the same length
        padded_histories = []
        for history in histories:
            if len(history) > 0:  # Only include non-empty histories
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

# Plot convergence curves
if history_data is not None:
    plot_convergence_curves(history_data)
    plot_average_convergence(history_data)

# %% [markdown]
# ## 5. Genetic Algorithm Specific Analysis

# %%
# Function to filter and analyze only GA configurations
def analyze_ga_configurations(results_df, history_data):
    if results_df is None or history_data is None:
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
    
    # 6. Impact of valid solutions focus
    valid_configs = {
        'Standard': [c for c in ga_configs if c == 'GA_Tournament_TwoPoint'],
        'Valid Initial': [c for c in ga_configs if 'Valid_Initial' in c],
        'Repair Operator': [c for c in ga_configs if 'Repair_Operator' in c]
    }
    
    # Filter out empty categories
    valid_configs = {k: v for k, v in valid_configs.items() if v}
    
    if len(valid_configs) > 1:
        plt.figure(figsize=(14, 8))
        plt.title("Impact of Valid Solutions Focus on Fitness", fontsize=16)
        
        valid_results = []
        for approach, configs in valid_configs.items():
            approach_results = ga_results[ga_results['Configuration'].isin(configs)]
            approach_results['Valid Solutions Focus'] = approach
            valid_results.append(approach_results)
        
        valid_df = pd.concat(valid_results)
        sns.boxplot(x='Valid Solutions Focus', y='Best Fitness', data=valid_df)
        plt.xlabel('Valid Solutions Approach', fontsize=14)
        plt.ylabel('Fitness (lower is better)', fontsize=14)
        plt.tight_layout()
    
    return ga_summary

# Analyze GA configurations
if results_df is not None and history_data is not None:
    ga_summary = analyze_ga_configurations(results_df, history_data)
    if ga_summary is not None:
        print("\nGenetic Algorithm Configurations Summary:")
        display(ga_summary)

# %% [markdown]
# ## 6. Algorithm Type Comparison

# %%
# Function to compare different algorithm types
def compare_algorithm_types(results_df, history_data):
    if results_df is None or history_data is None:
        return
    
    # Categorize configurations by algorithm type
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
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='Algorithm Type', y='Evaluations', data=algorithm_df)
        plt.title("Function Evaluations by Algorithm Type", fontsize=16)
        plt.xlabel('Algorithm Type', fontsize=14)
        plt.ylabel('Number of Function Evaluations', fontsize=14)
        plt.tight_layout()
        
        # Box plot for time comparison
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='Algorithm Type', y='Time', data=algorithm_df)
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
                        if len(history) > max_len:
                            max_len = len(history)
        
        for i, (algo_type, configs) in enumerate(algorithm_types.items()):
            # Collect all histories for this algorithm type
            all_histories = []
            for config in configs:
                if config in history_data:
                    for history in history_data[config]:
                        if len(history) > 0:  # Only include non-empty histories
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
# ## 7. Statistical Analysis

# %%
# Function to perform statistical analysis on the results
def perform_statistical_analysis(results_df):
    if results_df is None:
        return
    
    # Import statistical libraries
    from scipy import stats
    
    # Get unique configurations
    configs = results_df['Configuration'].unique()
    
    if len(configs) < 2:
        print("Need at least two configurations for statistical comparison.")
        return
    
    # Perform ANOVA to test if there are significant differences between configurations
    groups = [results_df[results_df['Configuration'] == config]['Best Fitness'].values 
              for config in configs]
    
    # Remove any NaN values
    groups = [group[~np.isnan(group)] for group in groups]
    
    # Only proceed if we have valid data
    if all(len(group) > 0 for group in groups):
        f_val, p_val = stats.f_oneway(*groups)
        
        print("\nStatistical Analysis:")
        print(f"ANOVA F-value: {f_val:.4f}, p-value: {p_val:.4f}")
        
        if p_val < 0.05:
            print("There are statistically significant differences between configurations (p < 0.05).")
            
            # Perform post-hoc tests to identify which configurations differ
            print("\nPost-hoc analysis (Tukey HSD):")
            
            # Prepare data for Tukey test
            data = []
            labels = []
            for i, config in enumerate(configs):
                group = results_df[results_df['Configuration'] == config]['Best Fitness'].values
                group = group[~np.isnan(group)]  # Remove NaNs
                data.extend(group)
                labels.extend([config] * len(group))
            
            # Convert to DataFrame for easier analysis
            tukey_data = pd.DataFrame({'Fitness': data, 'Configuration': labels})
            
            # Perform Tukey HSD test
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            tukey_result = pairwise_tukeyhsd(tukey_data['Fitness'], tukey_data['Configuration'], alpha=0.05)
            
            # Display results
            print(tukey_result)
        else:
            print("No statistically significant differences between configurations (p >= 0.05).")
    else:
        print("Some configurations have no valid fitness values for statistical analysis.")

# Perform statistical analysis
if results_df is not None:
    perform_statistical_analysis(results_df)

# %% [markdown]
# ## 8. Conclusion and Recommendations

# %%
# Function to generate conclusions and recommendations based on the analysis
def generate_conclusions(results_df, history_data):
    if results_df is None or history_data is None:
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
            'Repair Operator': [c for c in ga_configs if 'Repair_Operator' in c]
        }
        
        valid_performance = {}
        for approach, configs in valid_configs.items():
            if configs:
                mean_fitness = results_df[results_df['Configuration'].isin(configs)]['Best Fitness'].mean()
                valid_performance[approach] = mean_fitness
        
        if len(valid_performance) > 1:
            best_valid = min(valid_performance.items(), key=lambda x: x[1])[0]
            print(f"\n**Valid Solutions Focus**: The best performing valid solutions approach is **{best_valid}** with an average fitness of {valid_performance[best_valid]:.4f}.")
    
    # Recommendations
    print("\n### Recommendations")
    print("Based on the analysis, we recommend:")
    
    if best_config:
        print(f"1. Use the **{best_config}** configuration for best results.")
    
    if 'best_algo' in locals():
        print(f"2. Focus on **{best_algo}** algorithms for this problem domain.")
    
    if 'best_selection' in locals() and 'best_crossover' in locals():
        print(f"3. For genetic algorithms, use **{best_selection}** selection with **{best_crossover}** crossover.")
    
    if 'best_mutation' in locals() and 'best_elitism' in locals() and 'best_population' in locals():
        print(f"4. Optimize GA parameters: **{best_mutation}** mutation rate, **{best_elitism}** elitism, and **{best_population}** population size.")
    
    if 'best_valid' in locals():
        print(f"5. For handling valid solutions, the **{best_valid}** approach is recommended.")
    
    print("\n6. Consider further parameter tuning and hybrid approaches for even better results.")

# Generate conclusions and recommendations
if results_df is not None and history_data is not None:
    generate_conclusions(results_df, history_data)

# %% [markdown]
# ## 9. Save Results and Visualizations

# %%
# Save the figures to files
def save_visualizations():
    # Create a directory for visualizations if it doesn't exist
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # Save all open figures
    figures = [plt.figure(i) for i in plt.get_fignums()]
    for i, fig in enumerate(figures):
        fig.savefig(f'visualizations/figure_{i+1}.png', dpi=300, bbox_inches='tight')
    
    print(f"Saved {len(figures)} visualizations to the 'visualizations' directory.")

# Save visualizations
save_visualizations()

# %% [markdown]
# ## 10. Summary

# %%
print("""
# Summary of Analysis

This notebook has performed a comprehensive analysis of the optimization algorithms applied to the Fantasy League Team Optimization problem. The analysis includes:

1. **Basic Performance Analysis**: Comparison of fitness, function evaluations, and execution time across all configurations.

2. **Convergence Analysis**: Visualization of how fitness improves over iterations for each algorithm and configuration.

3. **Genetic Algorithm Specific Analysis**: Detailed analysis of the impact of different GA parameters (selection method, crossover type, mutation rate, elitism, population size, and valid solutions focus).

4. **Algorithm Type Comparison**: Comparison of different algorithm types (Hill Climbing, Simulated Annealing, Genetic Algorithm, Hybrid GA).

5. **Statistical Analysis**: ANOVA and post-hoc tests to determine if there are statistically significant differences between configurations.

6. **Conclusions and Recommendations**: Summary of the best performing configurations and recommendations for future optimization.

The visualizations have been saved to the 'visualizations' directory for reference.
""")
