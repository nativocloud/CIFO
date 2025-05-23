import random
import numpy as np
from copy import deepcopy

from solution import LeagueSolution, LeagueHillClimbingSolution

# MUTATION OPERATORS --------

def mutate_swap(solution):
    """
    Basic swap mutation: randomly swaps two players between teams.
    
    Args:
        solution (LeagueSolution): The solution to mutate
        
    Returns:
        LeagueSolution: A new solution with the mutation applied
    """
    new_repr = solution.repr[:]
    i, j = random.sample(range(len(new_repr)), 2)
    new_repr[i], new_repr[j] = new_repr[j], new_repr[i]
    
    return LeagueSolution(
        repr=new_repr,
        num_teams=solution.num_teams,
        team_size=solution.team_size,
        max_budget=solution.max_budget,
        players=solution.players
    )

def mutate_swap_constrained(solution):
    """
    Position-constrained swap mutation: randomly swaps two players of the same position.
    
    Args:
        solution (LeagueSolution): The solution to mutate
        
    Returns:
        LeagueSolution: A new solution with the mutation applied
    """
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
        return solution  # Return original if no swap is possible
    
    # Select two random players of this position
    idx1, idx2 = random.sample(position_map[pos], 2)
    
    # Create new solution with swapped players
    new_repr = solution.repr[:]
    new_repr[idx1], new_repr[idx2] = new_repr[idx2], new_repr[idx1]
    
    return LeagueSolution(
        repr=new_repr,
        num_teams=solution.num_teams,
        team_size=solution.team_size,
        max_budget=solution.max_budget,
        players=solution.players
    )

def mutate_team_shift(solution):
    """
    Team shift mutation: shifts all player assignments by a random number.
    
    Args:
        solution (LeagueSolution): The solution to mutate
        
    Returns:
        LeagueSolution: A new solution with the mutation applied
    """
    shift = random.randint(1, solution.num_teams - 1)
    new_repr = [(team_id + shift) % solution.num_teams for team_id in solution.repr]
    
    return LeagueSolution(
        repr=new_repr,
        num_teams=solution.num_teams,
        team_size=solution.team_size,
        max_budget=solution.max_budget,
        players=solution.players
    )

def mutate_targeted_player_exchange(solution):
    """
    Targeted player exchange: swaps players between teams to improve balance.
    Identifies the team with highest average skill and the team with lowest,
    then swaps a random player between them.
    
    Args:
        solution (LeagueSolution): The solution to mutate
        
    Returns:
        LeagueSolution: A new solution with the mutation applied
    """
    # Get team statistics
    teams = [[] for _ in range(solution.num_teams)]
    for idx, team_id in enumerate(solution.repr):
        teams[team_id].append({
            "idx": idx,
            "player": solution.players[idx]
        })
    
    # Calculate average skill for each team
    avg_skills = []
    for team in teams:
        avg_skill = np.mean([p["player"]["Skill"] for p in team])
        avg_skills.append(avg_skill)
    
    # Find highest and lowest skill teams
    highest_team = np.argmax(avg_skills)
    lowest_team = np.argmin(avg_skills)
    
    # If they're the same (unlikely), return original solution
    if highest_team == lowest_team:
        return solution
    
    # Create position maps for both teams
    high_team_by_pos = {}
    low_team_by_pos = {}
    
    for player in teams[highest_team]:
        pos = player["player"]["Position"]
        if pos not in high_team_by_pos:
            high_team_by_pos[pos] = []
        high_team_by_pos[pos].append(player)
    
    for player in teams[lowest_team]:
        pos = player["player"]["Position"]
        if pos not in low_team_by_pos:
            low_team_by_pos[pos] = []
        low_team_by_pos[pos].append(player)
    
    # Find a position that exists in both teams
    common_positions = set(high_team_by_pos.keys()) & set(low_team_by_pos.keys())
    if not common_positions:
        return solution  # No common positions, return original
    
    # Select a random common position
    pos = random.choice(list(common_positions))
    
    # Select a random player from each team with this position
    high_player = random.choice(high_team_by_pos[pos])
    low_player = random.choice(low_team_by_pos[pos])
    
    # Swap the players
    new_repr = solution.repr[:]
    new_repr[high_player["idx"]], new_repr[low_player["idx"]] = new_repr[low_player["idx"]], new_repr[high_player["idx"]]
    
    return LeagueSolution(
        repr=new_repr,
        num_teams=solution.num_teams,
        team_size=solution.team_size,
        max_budget=solution.max_budget,
        players=solution.players
    )

def mutate_shuffle_within_team_constrained(solution):
    """
    Shuffle within team: randomly selects a team and shuffles its players with other teams,
    while maintaining position constraints.
    
    Args:
        solution (LeagueSolution): The solution to mutate
        
    Returns:
        LeagueSolution: A new solution with the mutation applied
    """
    # Select a random team
    chosen_team = random.randint(0, solution.num_teams - 1)
    
    # Get indices of players in the chosen team
    team_indices = [i for i, team in enumerate(solution.repr) if team == chosen_team]
    
    # Create a new representation
    new_repr = solution.repr[:]
    
    # For each player in the team
    for idx in team_indices:
        # Get the player's position
        pos = solution.players[idx]["Position"]
        
        # Find all players with the same position in other teams
        other_team_players = [
            i for i, p in enumerate(solution.players) 
            if p["Position"] == pos and solution.repr[i] != chosen_team
        ]
        
        # If there are other players with the same position, swap with a random one
        if other_team_players:
            swap_idx = random.choice(other_team_players)
            new_repr[idx], new_repr[swap_idx] = new_repr[swap_idx], new_repr[idx]
    
    return LeagueSolution(
        repr=new_repr,
        num_teams=solution.num_teams,
        team_size=solution.team_size,
        max_budget=solution.max_budget,
        players=solution.players
    )

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
    
    return LeagueSolution(
        repr=child_repr,
        num_teams=parent1.num_teams,
        team_size=parent1.team_size,
        max_budget=parent1.max_budget,
        players=parent1.players
    )

def crossover_one_point_prefer_valid(parent1, parent2, max_attempts=10):
    """
    One-point crossover with validity preference: tries multiple cut points to find a valid solution.
    
    Args:
        parent1 (LeagueSolution): First parent solution
        parent2 (LeagueSolution): Second parent solution
        max_attempts (int): Maximum number of attempts to find a valid solution
        
    Returns:
        LeagueSolution: A new solution created by crossover
    """
    for _ in range(max_attempts):
        cut = random.randint(1, len(parent1.repr) - 2)
        child_repr = parent1.repr[:cut] + parent2.repr[cut:]
        
        child = LeagueSolution(
            repr=child_repr,
            num_teams=parent1.num_teams,
            team_size=parent1.team_size,
            max_budget=parent1.max_budget,
            players=parent1.players
        )
        
        if child.is_valid():
            return child
    
    # If no valid child found, return a copy of the better parent
    if parent1.fitness() <= parent2.fitness():
        return LeagueSolution(
            repr=parent1.repr[:],
            num_teams=parent1.num_teams,
            team_size=parent1.team_size,
            max_budget=parent1.max_budget,
            players=parent1.players
        )
    else:
        return LeagueSolution(
            repr=parent2.repr[:],
            num_teams=parent2.num_teams,
            team_size=parent2.team_size,
            max_budget=parent2.max_budget,
            players=parent2.players
        )

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
    
    return LeagueSolution(
        repr=child_repr,
        num_teams=parent1.num_teams,
        team_size=parent1.team_size,
        max_budget=parent1.max_budget,
        players=parent1.players
    )

def crossover_uniform_prefer_valid(parent1, parent2, max_attempts=10):
    """
    Uniform crossover with validity preference: tries multiple combinations to find a valid solution.
    
    Args:
        parent1 (LeagueSolution): First parent solution
        parent2 (LeagueSolution): Second parent solution
        max_attempts (int): Maximum number of attempts to find a valid solution
        
    Returns:
        LeagueSolution: A new solution created by crossover
    """
    for _ in range(max_attempts):
        child_repr = [
            parent1.repr[i] if random.random() < 0.5 else parent2.repr[i]
            for i in range(len(parent1.repr))
        ]
        
        child = LeagueSolution(
            repr=child_repr,
            num_teams=parent1.num_teams,
            team_size=parent1.team_size,
            max_budget=parent1.max_budget,
            players=parent1.players
        )
        
        if child.is_valid():
            return child
    
    # If no valid child found, return a copy of the better parent
    if parent1.fitness() <= parent2.fitness():
        return LeagueSolution(
            repr=parent1.repr[:],
            num_teams=parent1.num_teams,
            team_size=parent1.team_size,
            max_budget=parent1.max_budget,
            players=parent1.players
        )
    else:
        return LeagueSolution(
            repr=parent2.repr[:],
            num_teams=parent2.num_teams,
            team_size=parent2.team_size,
            max_budget=parent2.max_budget,
            players=parent2.players
        )

# SELECTION OPERATORS --------

def selection_tournament(population, k=3):
    """
    Tournament selection: selects the best solution from k random candidates.
    
    Args:
        population (list): List of LeagueSolution objects
        k (int): Tournament size
        
    Returns:
        LeagueSolution: The selected solution
    """
    selected = random.sample(population, k)
    selected.sort(key=lambda sol: sol.fitness()) 
    return selected[0]

def selection_tournament_variable_k(population, k=3):
    """
    Tournament selection with variable size: allows specifying tournament size.
    
    Args:
        population (list): List of LeagueSolution objects
        k (int): Tournament size
        
    Returns:
        LeagueSolution: The selected solution
    """
    # Ensure k is within valid range
    k = min(k, len(population))
    k = max(k, 1)
    
    selected = random.sample(population, k)
    selected.sort(key=lambda sol: sol.fitness()) 
    return selected[0]

def selection_ranking(population):
    """
    Ranking selection: selects solutions with probability proportional to their rank.
    
    Args:
        population (list): List of LeagueSolution objects
        
    Returns:
        LeagueSolution: The selected solution
    """
    sorted_pop = sorted(population, key=lambda s: s.fitness()) 
    ranks = list(range(1, len(sorted_pop)+1))
    total = sum(ranks)
    probs = [r / total for r in ranks[::-1]]  # Best gets highest prob
    return random.choices(sorted_pop, weights=probs, k=1)[0]

def selection_boltzmann(population, temperature=1.0):
    """
    Boltzmann selection: uses Boltzmann distribution to select solutions.
    
    Args:
        population (list): List of LeagueSolution objects
        temperature (float): Temperature parameter controlling selection pressure
        
    Returns:
        LeagueSolution: The selected solution
    """
    # Get all fitness values
    fitness_values = [sol.fitness() for sol in population]
    
    # For minimization problems, we need to invert the fitness
    # (lower fitness should have higher probability)
    inverted_fitness = [1.0 / (f + 1e-10) for f in fitness_values]  # Add small constant to avoid division by zero
    
    # Calculate Boltzmann probabilities
    boltzmann_values = [np.exp(f / temperature) for f in inverted_fitness]
    total = sum(boltzmann_values)
    probabilities = [b / total for b in boltzmann_values]
    
    # Select based on probabilities
    return random.choices(population, weights=probabilities, k=1)[0]

# This import is now at the top of the file
# from solution import LeagueHillClimbingSolution

# GENETIC ALGORITHM --------

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
        tuple: (best_solution, best_fitness, history)
    """
    # Initialize population
    population = generate_population(players, population_size)
    
    # Initialize tracking variables
    best_solution = min(population, key=lambda s: s.fitness())
    best_fitness = best_solution.fitness()
    history = [best_fitness]
    
    if verbose:
        print(f"Initial best fitness: {best_fitness}")
    
    # Main loop
    for generation in range(max_generations):
        # Sort population by fitness (ascending for minimization)
        population.sort(key=lambda x: x.fitness())
        
        # Create new population
        new_population = []
        
        # Apply elitism if enabled
        if elitism:
            new_population.extend(population[:elitism_size])
        
        # Fill the rest of the population
        while len(new_population) < population_size:
            # Selection
            if selection_params:
                parent1 = selection_operator(population, **selection_params)
                parent2 = selection_operator(population, **selection_params)
            else:
                parent1 = selection_operator(population)
                parent2 = selection_operator(population)
            
            # Crossover
            if random.random() < crossover_rate:
                child = crossover_operator(parent1, parent2)
            else:
                # No crossover, clone a parent
                child = LeagueSolution(
                    repr=parent1.repr[:],
                    num_teams=parent1.num_teams,
                    team_size=parent1.team_size,
                    max_budget=parent1.max_budget,
                    players=parent1.players
                )
            
            # Mutation
            if random.random() < mutation_rate:
                child = mutation_operator(child)
            
            # Add to new population if valid
            if child.is_valid():
                new_population.append(child)
        
        # Replace old population
        population = new_population
        
        # Apply local search if configured (hybrid GA)
        if local_search and generation % local_search.get("frequency", 5) == 0:
            # Apply hill climbing to the best solution
            if local_search.get("algorithm") == "hill_climbing":
                from evolution import hill_climbing
                
                # Convert to hill climbing solution
                hc_solution = LeagueHillClimbingSolution(
                    repr=population[0].repr[:],
                    num_teams=population[0].num_teams,
                    team_size=population[0].team_size,
                    max_budget=population[0].max_budget,
                    players=population[0].players
                )
                
                # Run hill climbing
                improved_solution, improved_fitness, _ = hill_climbing(
                    hc_solution,
                    max_iterations=local_search.get("iterations", 50),
                    verbose=False
                )
                
                # Replace the best solution if improved
                if improved_fitness < population[0].fitness():
                    population[0] = LeagueSolution(
                        repr=improved_solution.repr[:],
                        num_teams=improved_solution.num_teams,
                        team_size=improved_solution.team_size,
                        max_budget=improved_solution.max_budget,
                        players=improved_solution.players
                    )
        
        # Update best solution
        current_best = min(population, key=lambda s: s.fitness())
        current_best_fitness = current_best.fitness()
        
        if current_best_fitness < best_fitness:
            best_solution = LeagueSolution(
                repr=current_best.repr[:],
                num_teams=current_best.num_teams,
                team_size=current_best.team_size,
                max_budget=current_best.max_budget,
                players=current_best.players
            )
            best_fitness = current_best_fitness
            
            if verbose:
                print(f"Generation {generation}: New best fitness: {best_fitness}")
        elif verbose and generation % 5 == 0:
            print(f"Generation {generation}: Current best fitness: {current_best_fitness}")
        
        history.append(best_fitness)
    
    if verbose:
        print(f"Final best fitness: {best_fitness}")
    
    return best_solution, best_fitness, history
