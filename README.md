# Sports League Optimization

This project implements optimization algorithms for the Sports League problem, where players need to be assigned to teams in a balanced way while respecting position and budget constraints.

## Problem Definition

In the Sports League problem:
- 35 players must be assigned to 5 teams (7 players per team)
- Each team must have exactly 1 GK, 2 DEF, 2 MID, and 2 FWD
- Each team's total salary must not exceed 750M €
- The goal is to create balanced teams (minimize standard deviation of average team skills)

## Project Structure

The project follows a modular design with clear separation of concerns:

- `solution.py`: Contains the abstract `Solution` base class and concrete implementations for the Sports League problem
- `evolution.py`: Implements the optimization algorithms (Hill Climbing, Simulated Annealing, Genetic Algorithm)
- `operators.py`: Implements genetic operators (mutation, crossover, selection) and utility functions
- `main.ipynb`: Jupyter notebook for running experiments and visualizing results

## Usage

### Basic Usage

1. Clone the repository
2. Ensure you have the required dependencies: `numpy`, `pandas`, `matplotlib`
3. Open and run the `main.ipynb` notebook

### Creating a Solution

```python
from solution import LeagueSolution

# Create a random solution
solution = LeagueSolution(players=players_data)

# Check if the solution is valid
is_valid = solution.is_valid()

# Calculate the fitness (lower is better)
fitness = solution.fitness()
```

### Running Optimization Algorithms

#### Hill Climbing

```python
from solution import LeagueHillClimbingSolution
from evolution import hill_climbing

# Create an initial solution for Hill Climbing
hc_solution = LeagueHillClimbingSolution(players=players_data)

# Run Hill Climbing
best_solution, best_fitness, history = hill_climbing(
    hc_solution,
    max_iterations=500,
    max_no_improvement=100,
    verbose=True
)
```

#### Simulated Annealing

```python
from solution import LeagueSASolution
from evolution import simulated_annealing

# Create an initial solution for Simulated Annealing
sa_solution = LeagueSASolution(players=players_data)

# Run Simulated Annealing
best_solution, best_fitness, history = simulated_annealing(
    sa_solution,
    initial_temperature=200.0,
    cooling_rate=0.95,
    min_temperature=1e-5,
    iterations_per_temp=20,
    verbose=True
)
```

#### Genetic Algorithm

```python
from evolution import genetic_algorithm
from evolution import (
    selection_tournament_variable_k,
    crossover_one_point_prefer_valid,
    mutate_targeted_player_exchange
)

# Run Genetic Algorithm
best_solution, best_fitness, history = genetic_algorithm(
    players_data,
    population_size=100,
    max_generations=50,
    selection_operator=selection_tournament_variable_k,
    selection_params={"k": 3},
    crossover_operator=crossover_one_point_prefer_valid,
    crossover_rate=0.8,
    mutation_operator=mutate_targeted_player_exchange,
    mutation_rate=0.1,
    elitism=True,
    elitism_size=2,
    verbose=True
)
```

### Analyzing Results

```python
# Get detailed team statistics
team_stats = best_solution.get_team_stats()

# Display team compositions
for i, stats in enumerate(team_stats):
    print(f"Team {i}:")
    print(f"Average Skill: {stats['avg_skill']:.2f}")
    print(f"Total Salary: {stats['total_salary']} M €")
    print("Players:", stats['players'])
```

## Extending the Project

### Adding New Mutation Operators

To add a new mutation operator, define a function in `operators.py`:

```python
def mutate_new_operator(solution):
    """
    Description of the new mutation operator.
    
    Args:
        solution (LeagueSolution): The solution to mutate
        
    Returns:
        LeagueSolution: A new solution with the mutation applied
    """
    # Create a new representation based on the original
    new_repr = solution.repr[:]
    
    # Apply your mutation logic here
    # ...
    
    # Return a new solution with the mutated representation
    return LeagueSolution(
        repr=new_repr,
        num_teams=solution.num_teams,
        team_size=solution.team_size,
        max_budget=solution.max_budget,
        players=solution.players
    )
```

### Adding New Crossover Operators

To add a new crossover operator, define a function in `operators.py`:

```python
def crossover_new_operator(parent1, parent2):
    """
    Description of the new crossover operator.
    
    Args:
        parent1 (LeagueSolution): First parent solution
        parent2 (LeagueSolution): Second parent solution
        
    Returns:
        LeagueSolution: A new solution created by crossover
    """
    # Create a new representation based on the parents
    child_repr = []
    
    # Apply your crossover logic here
    # ...
    
    # Return a new solution with the child representation
    return LeagueSolution(
        repr=child_repr,
        num_teams=parent1.num_teams,
        team_size=parent1.team_size,
        max_budget=parent1.max_budget,
        players=parent1.players
    )
```

### Adding New Selection Operators

To add a new selection operator, define a function in `operators.py`:

```python
def selection_new_operator(population):
    """
    Description of the new selection operator.
    
    Args:
        population (list): List of LeagueSolution objects
        
    Returns:
        LeagueSolution: The selected solution
    """
    # Apply your selection logic here
    # ...
    
    # Return the selected solution
    return selected_solution
```

## Requirements

- Python 3.6+
- NumPy
- Pandas
- Matplotlib
