import random
import numpy as np
from copy import deepcopy

from solution import LeagueSolution, LeagueHillClimbingSolution, LeagueSASolution

# Hill Climbing algorithm
def hill_climbing(initial_solution, max_iterations=1000, max_no_improvement=100, verbose=False):
    """
    Implementation of the Hill Climbing optimization algorithm for the Sports League problem.

    Args:
        initial_solution (LeagueHillClimbingSolution): Initial solution to the optimization problem
        max_iterations (int): Maximum number of iterations
        max_no_improvement (int): Maximum number of iterations without improvement
        verbose (bool): Whether to print progress information

    Returns:
        tuple: (best_solution, best_fitness, history)
    """
    current = initial_solution
    current_fitness = current.fitness()
    history = [current_fitness]
    
    iterations_without_improvement = 0
    
    for iteration in range(max_iterations):
        neighbors = current.get_neighbors()
        
        if not neighbors:
            if verbose:
                print(f"No valid neighbors found at iteration {iteration}")
            break
        
        neighbor = min(neighbors, key=lambda x: x.fitness())
        neighbor_fitness = neighbor.fitness()
        
        if neighbor_fitness < current_fitness:
            current = neighbor
            current_fitness = neighbor_fitness
            history.append(current_fitness)
            iterations_without_improvement = 0
            
            if verbose:
                print(f"Iteration {iteration}: fitness = {current_fitness}")
        else:
            iterations_without_improvement += 1
            
            if iterations_without_improvement >= max_no_improvement:
                if verbose:
                    print(f"Stopping after {iteration} iterations with no improvement")
                break
    
    return current, current_fitness, history

# Simulated Annealing algorithm
def simulated_annealing(
    initial_solution,
    initial_temperature=100.0,
    cooling_rate=0.95,
    min_temperature=0.1,
    iterations_per_temp=20,
    maximization=False,
    verbose=False
):
    """
    Implementation of the Simulated Annealing optimization algorithm for the Sports League problem.

    Args:
        initial_solution (LeagueSASolution): Initial solution to the optimization problem
        initial_temperature (float): Starting temperature
        cooling_rate (float): Rate at which temperature decreases (0-1)
        min_temperature (float): Minimum temperature at which the algorithm stops
        iterations_per_temp (int): Number of iterations at each temperature
        maximization (bool): Whether this is a maximization problem (default: False)
        verbose (bool): Whether to print progress information (default: False)

    Returns:
        tuple: (best_solution, best_fitness, history)
    """
    # Initialize
    current_solution = initial_solution
    current_fitness = current_solution.fitness()
    
    best_solution = deepcopy(current_solution)
    best_fitness = current_fitness
    
    temperature = initial_temperature
    history = [current_fitness]
    
    if verbose:
        print(f"Initial solution fitness: {current_fitness}")
    
    # Main loop
    while temperature > min_temperature:
        for _ in range(iterations_per_temp):
            # Generate random neighbor
            neighbor = current_solution.get_random_neighbor()
            neighbor_fitness = neighbor.fitness()
            
            # Calculate fitness difference (delta)
            delta = neighbor_fitness - current_fitness
            
            # For minimization problems, we accept if delta < 0 (neighbor is better)
            # For maximization problems, we accept if delta > 0 (neighbor is better)
            accept = False
            
            if (maximization and delta > 0) or (not maximization and delta < 0):
                # Always accept better solutions
                accept = True
            else:
                # Accept worse solutions with a probability based on temperature
                # For minimization: e^(-|delta|/temperature)
                # For maximization: e^(-|delta|/temperature)
                probability = np.exp(-abs(delta) / temperature)
                if random.random() < probability:
                    accept = True
            
            # Update current solution if accepted
            if accept:
                current_solution = deepcopy(neighbor)
                current_fitness = neighbor_fitness
                
                # Update best solution if needed
                if (maximization and current_fitness > best_fitness) or \
                   (not maximization and current_fitness < best_fitness):
                    best_solution = deepcopy(current_solution)
                    best_fitness = current_fitness
                    
                    if verbose:
                        print(f"New best solution found: {best_fitness}")
            
            history.append(best_fitness)
        
        # Cool down
        temperature *= cooling_rate
        
        if verbose:
            print(f"Temperature: {temperature:.2f}, Current fitness: {current_fitness:.4f}, Best fitness: {best_fitness:.4f}")
    
    if verbose:
        print(f"Final solution fitness: {best_fitness}")
    
    return best_solution, best_fitness, history

# Import genetic algorithm functions from operators.py
from operators import (
    generate_population,
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
