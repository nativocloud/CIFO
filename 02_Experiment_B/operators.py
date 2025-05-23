import random
import numpy as np
from copy import deepcopy

from solution import LeagueSolution, LeagueHillClimbingSolution

# MUTATION OPERATORS --------

def mutate_swap(solution, mutation_rate=0.1):
    """
    Basic swap mutation: randomly swaps two players between teams.
    
    Args:
        solution (LeagueSolution): The solution to mutate
        mutation_rate (float): Probability of mutation (default: 0.1)
        
    Returns:
        LeagueSolution: A new solution with the mutation applied
    """
    # Ensure mutation_rate is not None
    if mutation_rate is None:
        mutation_rate = 0.1
        
    # Check if mutation should occur
    if random.random() > mutation_rate:
        return deepcopy(solution)
    
    new_repr = solution.repr[:]
    i, j = random.sample(range(len(new_repr)), 2)
    new_repr[i], new_repr[j] = new_repr[j], new_repr[i]
    
    mutated = deepcopy(solution)
    mutated.repr = new_repr
    return mutated

def mutate_scramble(solution, mutation_rate=0.1):
    """
    Scramble mutation: randomly selects a subsequence and shuffles it.
    
    Args:
        solution (LeagueSolution): The solution to mutate
        mutation_rate (float): Probability of mutation (default: 0.1)
        
    Returns:
        LeagueSolution: A new solution with the mutation applied
    """
    # Ensure mutation_rate is not None
    if mutation_rate is None:
        mutation_rate = 0.1
        
    # Check if mutation should occur
    if random.random() > mutation_rate:
        return deepcopy(solution)
    
    mutated = deepcopy(solution)
    
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

def mutate_swap_constrained(solution, mutation_rate=0.1):
    """
    Position-constrained swap mutation: randomly swaps two players of the same position.
    
    Args:
        solution (LeagueSolution): The solution to mutate
        mutation_rate (float): Probability of mutation (default: 0.1)
        
    Returns:
        LeagueSolution: A new solution with the mutation applied
    """
    # Ensure mutation_rate is not None
    if mutation_rate is None:
        mutation_rate = 0.1
        
    # Check if mutation should occur
    if random.random() > mutation_rate:
        return deepcopy(solution)
    
    # Create a position map for efficient mutation
    position_map = {}
    for idx, player in enumerate(solution.players):
        pos = player["Position"]
        if pos not in position_map:
            position_map[pos] = []
        position_map[pos].append(idx)
    
    # Randomly select a position
    pos = random.choice(list(position_map.keys()))
    
    # Need at least 2 players of this position to swap
    if len(position_map[pos]) < 2:
        return deepcopy(solution)  # Return original if no swap is possible
    
    # Select two random players of this position
    idx1, idx2 = random.sample(position_map[pos], 2)
    
    # Create new solution with swapped players
    mutated = deepcopy(solution)
    mutated.repr[idx1], mutated.repr[idx2] = mutated.repr[idx2], mutated.repr[idx1]
    
    return mutated

def mutate_team_shift(solution, mutation_rate=0.1):
    """
    Team shift mutation: shifts all player assignments by a random number.
    
    Args:
        solution (LeagueSolution): The solution to mutate
        mutation_rate (float): Probability of mutation (default: 0.1)
        
    Returns:
        LeagueSolution: A new solution with the mutation applied
    """
    # Ensure mutation_rate is not None
    if mutation_rate is None:
        mutation_rate = 0.1
        
    # Check if mutation should occur
    if random.random() > mutation_rate:
        return deepcopy(solution)
    
    shift = random.randint(1, solution.num_teams - 1)
    new_repr = [(team_id + shift) % solution.num_teams for team_id in solution.repr]
    
    mutated = deepcopy(solution)
    mutated.repr = new_repr
    return mutated

# CROSSOVER OPERATORS --------

def crossover_one_point(parent1, parent2):
    """
    One-point crossover: creates a child by taking a portion from each parent.
    
    Args:
        parent1 (LeagueSolution): First parent solution
        parent2 (LeagueSolution): Second parent solution
        
    Returns:
        LeagueSolution: A new solution created by crossover
    """
    cut = random.randint(1, len(parent1.repr) - 2)
    child_repr = parent1.repr[:cut] + parent2.repr[cut:]
    
    child = deepcopy(parent1)
    child.repr = child_repr
    return child

def crossover_two_point(parent1, parent2):
    """
    Two-point crossover: creates a child by taking portions from each parent.
    
    Args:
        parent1 (LeagueSolution): First parent solution
        parent2 (LeagueSolution): Second parent solution
        
    Returns:
        LeagueSolution: A new solution created by crossover
    """
    length = len(parent1.repr)
    cut1 = random.randint(1, length - 3)
    cut2 = random.randint(cut1 + 1, length - 2)
    
    child_repr = parent1.repr[:cut1] + parent2.repr[cut1:cut2] + parent1.repr[cut2:]
    
    child = deepcopy(parent1)
    child.repr = child_repr
    return child

def crossover_uniform(parent1, parent2):
    """
    Uniform crossover: creates a child by randomly selecting genes from either parent.
    
    Args:
        parent1 (LeagueSolution): First parent solution
        parent2 (LeagueSolution): Second parent solution
        
    Returns:
        LeagueSolution: A new solution created by crossover
    """
    child_repr = [
        parent1.repr[i] if random.random() < 0.5 else parent2.repr[i]
        for i in range(len(parent1.repr))
    ]
    
    child = deepcopy(parent1)
    child.repr = child_repr
    return child

# SELECTION OPERATORS --------

def selection_tournament(fitness_values, tournament_size=3):
    """
    Tournament selection: selects the best solution from k random candidates.
    
    Args:
        fitness_values (list): List of fitness values
        tournament_size (int): Tournament size
        
    Returns:
        int: Index of the selected solution
    """
    # Ensure tournament size is valid
    tournament_size = min(tournament_size, len(fitness_values))
    tournament_size = max(tournament_size, 1)
    
    # Select random candidates
    candidates = random.sample(range(len(fitness_values)), tournament_size)
    
    # Return the best candidate
    return min(candidates, key=lambda i: fitness_values[i])

def selection_ranking(fitness_values):
    """
    Ranking selection: selects solutions with probability proportional to their rank.
    
    Args:
        fitness_values (list): List of fitness values
        
    Returns:
        int: Index of the selected solution
    """
    # Create pairs of (index, fitness)
    indexed_fitness = list(enumerate(fitness_values))
    
    # Sort by fitness (ascending for minimization)
    sorted_pairs = sorted(indexed_fitness, key=lambda x: x[1])
    
    # Extract indices in sorted order
    sorted_indices = [idx for idx, _ in sorted_pairs]
    
    # Calculate ranks (best gets highest rank)
    ranks = list(range(1, len(sorted_indices) + 1))
    ranks.reverse()  # Reverse so best gets highest rank
    
    # Calculate total rank sum
    total = sum(ranks)
    
    # Calculate selection probabilities
    probs = [r / total for r in ranks]
    
    # Select based on probabilities
    return random.choices(sorted_indices, weights=probs, k=1)[0]

def selection_boltzmann(fitness_values, temperature=1.0):
    """
    Boltzmann selection: uses Boltzmann distribution to select solutions.
    
    Args:
        fitness_values (list): List of fitness values
        temperature (float): Temperature parameter controlling selection pressure
        
    Returns:
        int: Index of the selected solution
    """
    # For minimization problems, we need to invert the fitness
    # (lower fitness should have higher probability)
    inverted_fitness = [1.0 / (f + 1e-10) for f in fitness_values]  # Add small constant to avoid division by zero
    
    # Calculate Boltzmann probabilities
    boltzmann_values = [np.exp(f / temperature) for f in inverted_fitness]
    total = sum(boltzmann_values)
    probabilities = [b / total for b in boltzmann_values]
    
    # Select based on probabilities
    return random.choices(range(len(fitness_values)), weights=probabilities, k=1)[0]

# POPULATION GENERATION --------

def generate_population(players, size, num_teams=5, team_size=7, max_budget=750):
    """
    Generate a random initial population of valid solutions.
    
    Args:
        players (list): List of player dictionaries
        size (int): Population size
        num_teams (int): Number of teams
        team_size (int): Number of players per team
        max_budget (int): Maximum budget per team
        
    Returns:
        list: List of LeagueSolution objects
    """
    population = []
    
    while len(population) < size:
        solution = LeagueSolution(
            num_teams=num_teams,
            team_size=team_size,
            max_budget=max_budget,
            players=players
        )
        
        if solution.is_valid():
            population.append(solution)
    
    return population

# GENETIC ALGORITHM --------

def genetic_algorithm(
    players,
    population_size=50,
    max_generations=30,
    selection_operator=selection_tournament,
    selection_params=None,
    crossover_operator=crossover_one_point,
    crossover_rate=0.8,
    mutation_operator=mutate_swap,
    mutation_rate=0.2,
    elitism=True,
    elitism_size=1,
    local_search=None,
    verbose=False
):
    """
    Genetic Algorithm implementation for the Sports League problem.
    
    Args:
        players (list): List of player dictionaries
        population_size (int): Size of the population
        max_generations (int): Maximum number of generations
        selection_operator (function): Function to select parents
        selection_params (dict): Parameters for selection operator
        crossover_operator (function): Function to perform crossover
        crossover_rate (float): Probability of crossover
        mutation_operator (function): Function to perform mutation
        mutation_rate (float): Probability of mutation
        elitism (bool): Whether to use elitism
        elitism_size (int): Number of elite solutions to preserve
        local_search (dict): Local search configuration (if using hybrid GA)
        verbose (bool): Whether to print progress information
        
    Returns:
        tuple: (best_solution, best_fitness, fitness_history)
    """
    # Initialize selection parameters
    if selection_params is None:
        selection_params = {}
    
    # Generate initial population
    population = generate_population(
        players=players,
        size=population_size,
        num_teams=5,
        team_size=7,
        max_budget=750
    )
    
    # Evaluate initial population
    fitness_values = [solution.fitness() for solution in population]
    
    # Find best solution
    best_idx = fitness_values.index(min(fitness_values))
    best_solution = deepcopy(population[best_idx])
    best_fitness = fitness_values[best_idx]
    
    # Initialize fitness history
    fitness_history = [best_fitness]
    
    # Main loop
    for generation in range(max_generations):
        # Create new population
        new_population = []
        
        # Elitism: keep best solutions
        if elitism:
            elite_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])[:elitism_size]
            elite = [deepcopy(population[i]) for i in elite_indices]
            new_population.extend(elite)
        
        # Create offspring
        while len(new_population) < population_size:
            # Selection
            if selection_operator == selection_tournament:
                tournament_size = selection_params.get('tournament_size', 3)
                parent1_idx = selection_operator(fitness_values, tournament_size)
                parent2_idx = selection_operator(fitness_values, tournament_size)
            elif selection_operator == selection_boltzmann:
                temperature = selection_params.get('temperature', 1.0)
                parent1_idx = selection_operator(fitness_values, temperature)
                parent2_idx = selection_operator(fitness_values, temperature)
            else:
                parent1_idx = selection_operator(fitness_values)
                parent2_idx = selection_operator(fitness_values)
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover
            if random.random() < crossover_rate:
                child = crossover_operator(parent1, parent2)
            else:
                child = deepcopy(parent1)
            
            # Mutation
            child = mutation_operator(child, mutation_rate)
            
            # Local search (if enabled)
            if local_search is not None and random.random() < local_search.get('probability', 0.1):
                # Simple hill climbing
                for _ in range(local_search.get('iterations', 5)):
                    neighbor = deepcopy(child)
                    idx = random.randint(0, len(neighbor.repr) - 1)
                    neighbor.repr[idx] = random.randint(0, neighbor.num_teams - 1)
                    
                    if neighbor.fitness() < child.fitness():
                        child = neighbor
            
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
        
        # Update fitness history
        fitness_history.append(best_fitness)
        
        # Print progress
        if verbose and generation % 10 == 0:
            print(f"Generation {generation}: Best fitness = {best_fitness}")
    
    return best_solution, best_fitness, fitness_history
