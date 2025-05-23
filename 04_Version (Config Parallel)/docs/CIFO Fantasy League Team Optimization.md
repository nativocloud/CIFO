# CIFO Fantasy League Team Optimization

## Project Overview

This project implements and analyzes various optimization algorithms for solving the Fantasy League Team Optimization problem. The goal is to create balanced teams of players while respecting position and budget constraints.

## Latest Updates (May 2025)

- **Integrated Pipeline**: Combined execution and visualization into a single comprehensive pipeline
- **Parallel Execution**: Added multiprocessing support for faster algorithm evaluation
- **New Algorithms**: Implemented promising algorithms including:
  - GA Memetic (with local search)
  - HC with Random Restart
  - GA Island Model
  - GA with Scramble Mutation
- **Statistical Analysis**: Added robust statistical tests to compare algorithm performance
- **Data Storage**: Implemented comprehensive data saving for all metrics and runs
- **Visualization Improvements**: Fixed convergence curves and added normalized comparisons
- **Team Solution Display**: Added detailed visualization of the optimal team composition

## Problem Description

The Fantasy League Team Optimization problem involves:
- Creating balanced teams from a pool of players
- Respecting position constraints (1 GK, 2 DEF, 2 MID, 2 FWD per team)
- Staying within budget constraints
- Maximizing skill balance across teams

The objective function minimizes the standard deviation of average skills across teams, creating balanced competition.

## Algorithms Implemented

### Base Algorithms
1. **Hill Climbing (HC_Standard)**
   - Simple local search that accepts only improving moves
   - Prone to getting stuck in local optima

2. **Simulated Annealing (SA_Standard)**
   - Probabilistically accepts worse solutions based on temperature
   - Helps escape local optima through controlled exploration

3. **Genetic Algorithm Variants**
   - **GA_Tournament_OnePoint**: Tournament selection with one-point crossover
   - **GA_Tournament_TwoPoint**: Tournament selection with two-point crossover
   - **GA_Rank_Uniform**: Rank-based selection with uniform crossover
   - **GA_Boltzmann_TwoPoint**: Boltzmann selection with two-point crossover
   - **GA_Hybrid**: GA with occasional local search

### Promising Algorithms
4. **GA Memetic**
   - Combines genetic algorithms with local search
   - Applies hill climbing to individuals with a certain probability
   - Balances global exploration with local exploitation

5. **Hill Climbing with Random Restart**
   - Restarts search from a new random point when stuck
   - Helps escape local optima through diversification

6. **GA Island Model**
   - Maintains multiple isolated populations (islands)
   - Periodically migrates individuals between islands
   - Preserves genetic diversity and prevents premature convergence

7. **GA with Scramble Mutation**
   - Uses a mutation operator that shuffles a subsequence
   - Provides better exploration of the search space

## Project Structure

```
CIFO-24-25/
├── CIFO_Complete_Pipeline_Final.ipynb  # Complete pipeline notebook
├── CIFO_Complete_Pipeline_Final.py     # Python script version
├── solution.py                         # Solution representation
├── evolution.py                        # Evolutionary algorithms
├── operators.py                        # Genetic operators
├── fitness_counter.py                  # Fitness evaluation counter
├── players.csv                         # Player dataset
├── requirements.txt                    # Dependencies
└── experiment_results/                 # Results directory
    └── experiment_TIMESTAMP/           # Experiment results
        ├── results.csv                 # Performance metrics
        ├── history_data.npy            # Convergence history
        └── stats_results.json          # Statistical analysis
```

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/nativocloud/CIFO.git
   cd CIFO-24-25
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv cifo-env
   source cifo-env/bin/activate  # On Windows: cifo-env\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install additional packages for parallel execution:
   ```
   pip install multiprocessing jupytext scikit-posthocs
   ```

## Running the Pipeline

### Option 1: Using Jupyter Notebook

1. Start Jupyter:
   ```
   jupyter notebook
   ```

2. Open `CIFO_Complete_Pipeline_Final.ipynb`

3. Run all cells (Cell > Run All)

### Option 2: Using Python Script

Run the Python script version:
```
python CIFO_Complete_Pipeline_Final.py
```

## Configuration Options

The pipeline includes a centralized configuration dictionary `EXPERIMENT_CONFIG` where you can adjust:

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

## Parallel Execution

The pipeline supports parallel execution using Python's multiprocessing library. You can configure:

- **Execution Mode**: `ExecutionMode.SINGLE_PROCESSOR` for sequential execution or `ExecutionMode.MULTI_PROCESSOR` for parallel execution
- **Number of Processes**: Set `num_processes` to control CPU utilization (default: all available cores minus one)

## Data Storage

All experiment data is saved in a structured directory:

- **Results CSV**: Contains performance metrics for each algorithm run
- **History NPY**: Contains convergence history data for visualization
- **Statistics JSON**: Contains results of statistical tests

## Visualization Features

The pipeline includes several visualization types:

1. **Performance Comparison**:
   - Bar charts comparing fitness, function evaluations, and runtime
   - Error bars showing standard deviation

2. **Convergence Analysis**:
   - Individual convergence curves for each run
   - Average convergence curves with standard deviation bands
   - Normalized convergence curves for fair comparison

3. **Statistical Visualization**:
   - Boxplots with statistical significance groupings
   - P-value matrices for pairwise comparisons

4. **Team Solution Display**:
   - Detailed visualization of the optimal team composition
   - Player statistics and position distribution

## Statistical Analysis

The pipeline performs comprehensive statistical analysis:

1. **Normality Testing**: Shapiro-Wilk test to check distribution
2. **Homogeneity of Variances**: Levene's test
3. **Algorithm Comparison**:
   - ANOVA (for normal data) or Kruskal-Wallis (for non-normal data)
   - Effect size calculation and interpretation
4. **Post-hoc Tests**:
   - Tukey HSD (after ANOVA) or Dunn's test (after Kruskal-Wallis)
   - P-value matrix for pairwise comparisons

## Results and Findings

Based on our experiments, we found:

1. **Algorithm Performance**:
   - Genetic Algorithms generally outperformed Hill Climbing and Simulated Annealing
   - The Memetic GA approach showed the best balance between solution quality and computational cost
   - GA with Island Model demonstrated superior performance in maintaining population diversity
   - HC with Random Restart significantly improved over standard HC by escaping local optima

2. **Parameter Impact**:
   - **Selection Methods**: Tournament selection provided the best balance between exploration and exploitation
   - **Crossover Types**: Two-Point crossover preserved important building blocks better than other methods
   - **Mutation Operators**: Scramble mutation improved exploration compared to standard swap mutation
   - **Elitism**: Some elitism (10%) improved performance by preserving good solutions
   - **Population Size**: Larger populations found better solutions but required more computational resources

3. **Statistical Significance**:
   - Statistical tests confirmed significant differences between algorithms
   - The effect size was large, indicating that algorithm choice has substantial impact on performance
   - Post-hoc tests identified groups of algorithms with statistically similar performance

## Future Work

Potential areas for future improvement:

1. **Adaptive Parameter Control**: Dynamically adjust mutation and crossover rates
2. **Multi-objective Optimization**: Balance team skill and budget constraints
3. **Advanced Repair Operators**: More sophisticated handling of constraints
4. **Hybrid Approaches**: Combine strengths of different algorithms
5. **Niching Techniques**: Explore multiple good solutions simultaneously

## Contributors

- CIFO Team 2024-2025

## License

This project is licensed under the MIT License - see the LICENSE file for details.
