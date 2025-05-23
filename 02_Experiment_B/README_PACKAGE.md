# CIFO Fantasy League Optimization Package

This package contains all the necessary files to run the complete optimization pipeline for the Fantasy League Team Optimization problem.

## Contents

- `CIFO_Complete_Pipeline_Final.ipynb` - Jupyter notebook with the complete pipeline
- `CIFO_Complete_Pipeline_Final.py` - Python script version of the notebook
- `solution.py` - Implementation of the solution representation
- `evolution.py` - Implementation of evolutionary algorithms
- `operators.py` - Implementation of genetic operators
- `fitness_counter.py` - Implementation of fitness evaluation counter
- `players.csv` - Dataset with player information
- `requirements.txt` - Python package dependencies
- `README.md` - Original project README
- `README_PACKAGE.md` - This file

## Setup Instructions

1. Create a virtual environment (recommended):
   ```
   python -m venv cifo-env
   source cifo-env/bin/activate  # On Windows: cifo-env\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install additional packages for parallel execution:
   ```
   pip install multiprocessing jupytext scikit-posthocs
   ```

## Running the Pipeline

You can run the pipeline in two ways:

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

## Configuration

The pipeline includes a centralized configuration dictionary `EXPERIMENT_CONFIG` where you can adjust:

- Number of runs per algorithm (`num_runs`)
- Execution mode (parallel or sequential)
- Number of processes for parallel execution
- Statistical analysis parameters
- Data storage options

## Output

The pipeline will create a directory structure in `experiment_results/` containing:

- CSV files with performance metrics
- NPY files with convergence history
- JSON files with statistical analysis results

## Algorithms Included

1. **Existing Algorithms**:
   - Hill Climbing (HC_Standard)
   - Simulated Annealing (SA_Standard)
   - GA with Tournament Selection and OnePoint Crossover
   - GA with Tournament Selection and TwoPoint Crossover
   - GA with Rank Selection and Uniform Crossover
   - GA with Boltzmann Selection and TwoPoint Crossover
   - GA Hybrid

2. **Promising Algorithms**:
   - GA Memetic (with local search applied to each individual)
   - HC with Random Restart (to escape local optima)
   - GA with Island Model (for maintaining genetic diversity)
   - GA with Scramble Mutation (for better exploration)

## Troubleshooting

- If you encounter memory issues during parallel execution, reduce the number of processes in `EXPERIMENT_CONFIG['num_processes']`
- For visualization issues, ensure matplotlib is properly configured for your environment
