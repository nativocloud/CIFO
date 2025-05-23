# CIFO Fantasy League Team Optimization Project

## Overview

This project implements and compares various optimization algorithms for solving the Fantasy League team selection problem. The goal is to create balanced teams of players while respecting constraints such as budget limits, team size, and position requirements.

## Problem Description

The Fantasy League problem involves:
- Selecting players to form multiple balanced teams
- Respecting a maximum budget per team
- Ensuring each team has the required number of players
- Maintaining position balance within teams
- Maximizing overall team skill while minimizing skill variance between teams

## Algorithms Implemented

The project implements and compares the following algorithms:

### Local Search Algorithms
- **Hill Climbing (HC_Standard)**: Basic hill climbing with random neighborhood exploration
- **Hill Climbing with Random Restart (HC_Random_Restart)**: Hill climbing that periodically restarts from a new random solution
- **Simulated Annealing (SA_Standard)**: Simulated annealing with exponential cooling schedule

### Genetic Algorithms
- **GA with Tournament Selection and OnePoint Crossover**
- **GA with Tournament Selection and TwoPoint Crossover**
- **GA with Rank Selection and Uniform Crossover**
- **GA with Boltzmann Selection and TwoPoint Crossover**
- **GA Memetic**: Genetic algorithm with local search applied to individuals
- **GA with Island Model**: Multiple populations with periodic migration
- **GA with Scramble Mutation**: Uses scramble mutation operator for better exploration

### Hybrid Approaches
- **GA Hybrid**: Combines genetic algorithm with hill climbing

## Project Structure

- `solution.py`: Solution representation and fitness calculation
- `operators.py`: Genetic operators (selection, crossover, mutation)
- `evolution.py`: Evolutionary algorithm implementations
- `fitness_counter.py`: Fitness evaluation counter
- `CIFO_Complete_Pipeline_Parallel_Option.py/ipynb`: Complete pipeline with parallel execution option
- `players.csv`: Player dataset with positions, skills, and salaries

## Key Features

1. **Comprehensive Algorithm Comparison**: Implements and compares multiple optimization algorithms
2. **Statistical Analysis**: Includes robust statistical tests to determine significant performance differences
3. **Parallel Execution**: Supports parallel processing for faster experimentation
4. **Visualization**: Provides detailed visualizations of algorithm performance and convergence
5. **Optimal Team Display**: Shows the best team solution with detailed player information

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages: numpy, pandas, matplotlib, scipy, scikit-posthocs

### Installation

1. Clone the repository or extract the provided package
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Pipeline

The project provides two ways to run the pipeline:

1. **Jupyter Notebook**:
   ```
   jupyter notebook CIFO_Complete_Pipeline_Parallel_Option.ipynb
   ```

2. **Python Script**:
   ```
   python CIFO_Complete_Pipeline_Parallel_Option.py
   ```

The pipeline includes a fix for module imports, so it will work regardless of which directory you run it from.

## Configuration

The pipeline is highly configurable through the `EXPERIMENT_CONFIG` dictionary:

```python
EXPERIMENT_CONFIG = {
    'num_runs': 30,                          # Number of runs per algorithm
    'max_evaluations': 10000,                # Maximum fitness evaluations per run
    'use_parallel': True,                    # Whether to use parallel processing
    'execution_mode': ExecutionMode.MULTI_PROCESSOR,  # Set automatically based on use_parallel
    'num_processes': max(1, multiprocessing.cpu_count() - 1),  # Number of parallel processes
    'load_existing': False,                  # Whether to load existing results
    'results_dir': 'experiment_results',     # Directory for saving results
    'save_results': True,                    # Whether to save results
    'random_seed': 42,                       # Random seed for reproducibility
    'significance_level': 0.05,              # Significance level for statistical tests
}
```

## Latest Updates

1. **Parallel Processing**: Added configurable parallel execution option (set `use_parallel = True`)
2. **New Algorithms**: Implemented promising algorithms (GA Memetic, HC Random Restart, GA Island Model, GA Scramble Mutation)
3. **Robust Statistical Analysis**: Enhanced statistical tests with effect size calculation
4. **Improved Visualization**: Added convergence curves and algorithm comparison plots
5. **Operator Interface Consistency**: Standardized mutation and crossover operator interfaces
6. **Team Solution Display**: Added detailed visualization of the optimal team solution
7. **Module Import Fix**: Added automatic path configuration to ensure modules are found regardless of execution directory

## Results and Analysis

The pipeline generates comprehensive results including:

1. **Performance Comparison**: Bar charts comparing algorithm performance
2. **Convergence Analysis**: Plots showing how fitness improves over iterations
3. **Statistical Significance**: ANOVA/Kruskal-Wallis tests with post-hoc analysis
4. **Best Team Solution**: Detailed breakdown of the optimal team composition

## Troubleshooting

If you encounter any issues:

1. **Module Import Errors**: The pipeline automatically adds its directory to the Python path. If you still encounter import errors, make sure all files are in the same directory.

2. **Parallel Processing Errors**: If you encounter errors with parallel processing, try setting `use_parallel = False` in the `EXPERIMENT_CONFIG` dictionary.

3. **Missing Operators**: The package includes a complete `operators.py` file with all necessary functions. If you get errors about missing operators, make sure you're using this version.

## Contributing

To extend this project:

1. Add new algorithms in `evolution.py`
2. Implement new operators in `operators.py`
3. Add new algorithm configurations to the pipeline

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was developed for the Computational Intelligence for Optimization course
- Special thanks to all contributors and researchers in the field of evolutionary computation
