# Genetic Operators Documentation

This document provides a detailed explanation of the genetic operators used in the Fantasy League Team Optimization project.

## 1. Selection Operators

Selection operators determine which individuals from the current population will be chosen as parents for the next generation.

### 1.1 Tournament Selection

**Implementation**: `selection_tournament` in `operators.py`

**Description**:
Tournament selection works by randomly selecting a small subset of individuals (tournament) from the population and choosing the best one as a parent. This process is repeated to select multiple parents.

**Parameters**:
- `tournament_size`: Number of individuals in each tournament (default: 3)

**Algorithm**:
1. Randomly select `tournament_size` individuals from the population
2. Choose the individual with the best fitness from the tournament
3. Return the index of the selected individual

**Characteristics**:
- Selection pressure can be adjusted by changing the tournament size
- Larger tournaments increase selection pressure toward fitter individuals
- Smaller tournaments allow more exploration and diversity
- Does not require sorting the entire population
- Computationally efficient

**Code Example**:
```python
def selection_tournament(fitness_values, tournament_size=3):
    """
    Tournament selection: randomly select tournament_size individuals and choose the best.
    
    Args:
        fitness_values: List of fitness values for the population
        tournament_size: Number of individuals in each tournament
        
    Returns:
        Index of the selected individual
    """
    population_size = len(fitness_values)
    tournament_indices = random.sample(range(population_size), tournament_size)
    tournament_fitness = [fitness_values[i] for i in tournament_indices]
    
    # Find the best individual in the tournament (minimum fitness for minimization)
    winner_idx = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
    return winner_idx
```

### 1.2 Rank-Based Selection

**Implementation**: `selection_ranking` in `operators.py`

**Description**:
Rank-based selection assigns selection probabilities based on the rank of individuals in the population, rather than their absolute fitness values. This helps maintain selection pressure even when fitness values are very close.

**Algorithm**:
1. Sort the population by fitness
2. Assign ranks to each individual (best = highest rank)
3. Calculate selection probabilities based on ranks
4. Select an individual using these probabilities

**Characteristics**:
- Reduces the effect of outliers in fitness values
- Maintains selection pressure even when fitness differences are small
- More computationally intensive than tournament selection due to sorting
- Helps prevent premature convergence

**Code Example**:
```python
def selection_ranking(fitness_values):
    """
    Rank-based selection: selection probability based on rank rather than absolute fitness.
    
    Args:
        fitness_values: List of fitness values for the population
        
    Returns:
        Index of the selected individual
    """
    population_size = len(fitness_values)
    
    # Create (index, fitness) pairs and sort by fitness (ascending for minimization)
    indexed_fitness = [(i, fitness) for i, fitness in enumerate(fitness_values)]
    indexed_fitness.sort(key=lambda x: x[1])
    
    # Assign ranks (higher rank = better individual)
    ranks = [population_size - i for i in range(population_size)]
    
    # Calculate total rank sum for normalization
    total_rank = sum(ranks)
    
    # Calculate selection probabilities
    probabilities = [rank / total_rank for rank in ranks]
    
    # Cumulative probabilities for selection
    cum_prob = [sum(probabilities[:i+1]) for i in range(population_size)]
    
    # Select based on probabilities
    r = random.random()
    for i, cp in enumerate(cum_prob):
        if r <= cp:
            return indexed_fitness[i][0]
    
    # Fallback (should not reach here)
    return indexed_fitness[-1][0]
```

### 1.3 Boltzmann Selection

**Implementation**: `selection_boltzmann` in `operators.py`

**Description**:
Boltzmann selection uses a temperature parameter to control selection pressure over time. At high temperatures, selection is almost random, while at low temperatures, selection strongly favors fitter individuals.

**Parameters**:
- `temperature`: Controls selection pressure (default: decreases over time)

**Algorithm**:
1. Calculate a Boltzmann factor for each individual: e^(-fitness/temperature)
2. Normalize these factors to create selection probabilities
3. Select an individual based on these probabilities

**Characteristics**:
- Dynamically adjusts selection pressure during evolution
- Starts with exploration (high temperature) and gradually shifts to exploitation (low temperature)
- Helps prevent premature convergence
- Computationally more intensive than tournament selection

**Code Example**:
```python
def selection_boltzmann(fitness_values, temperature=1.0):
    """
    Boltzmann selection: selection probability based on Boltzmann distribution.
    
    Args:
        fitness_values: List of fitness values for the population
        temperature: Temperature parameter (controls selection pressure)
        
    Returns:
        Index of the selected individual
    """
    population_size = len(fitness_values)
    
    # Calculate Boltzmann factors (e^(-fitness/T))
    # Note: For minimization problems, smaller fitness is better
    boltzmann_factors = [math.exp(-fitness / temperature) for fitness in fitness_values]
    
    # Calculate total for normalization
    total = sum(boltzmann_factors)
    
    # Calculate selection probabilities
    probabilities = [factor / total for factor in boltzmann_factors]
    
    # Cumulative probabilities for selection
    cum_prob = [sum(probabilities[:i+1]) for i in range(population_size)]
    
    # Select based on probabilities
    r = random.random()
    for i, cp in enumerate(cum_prob):
        if r <= cp:
            return i
    
    # Fallback (should not reach here)
    return population_size - 1
```

## 2. Crossover Operators

Crossover operators combine genetic material from two parent solutions to create offspring.

### 2.1 One-Point Crossover

**Implementation**: `crossover_one_point` in `operators.py`

**Description**:
One-point crossover selects a single random point in the chromosome, and exchanges the genetic material between the two parents at that point to create a child.

**Algorithm**:
1. Select a random crossover point
2. Create a child by taking genes from the first parent up to the crossover point
3. Take the remaining genes from the second parent

**Characteristics**:
- Simple and efficient
- Preserves segments of genetic material
- May disrupt building blocks if they span the crossover point
- Good for problems where gene order matters

**Code Example**:
```python
def crossover_one_point(parent1, parent2):
    """
    One-point crossover: creates a child by taking portions from each parent.
    
    Args:
        parent1: First parent solution
        parent2: Second parent solution
        
    Returns:
        A new solution created by crossover
    """
    cut_point = random.randint(1, len(parent1.repr) - 1)
    child_repr = parent1.repr[:cut_point] + parent2.repr[cut_point:]
    
    return LeagueSolution(
        repr=child_repr,
        num_teams=parent1.num_teams,
        team_size=parent1.team_size,
        max_budget=parent1.max_budget,
        players=parent1.players
    )
```

### 2.2 Two-Point Crossover

**Implementation**: `two_point_crossover` in `CIFO_Complete_Pipeline_Final.py`

**Description**:
Two-point crossover selects two random points in the chromosome, and exchanges the genetic material between the two parents between these points.

**Algorithm**:
1. Select two random crossover points
2. Create a child by taking genes from the first parent up to the first crossover point
3. Take genes from the second parent between the two crossover points
4. Take the remaining genes from the first parent

**Characteristics**:
- More flexible than one-point crossover
- Better preserves building blocks at the ends of the chromosome
- Allows for more recombination possibilities
- Good for problems with complex gene interactions

**Code Example**:
```python
def two_point_crossover(parent1, parent2):
    """
    Two-point crossover: creates a child by taking portions from each parent.
    
    Args:
        parent1: First parent solution
        parent2: Second parent solution
        
    Returns:
        A new solution created by crossover
    """
    cut1 = random.randint(1, len(parent1.repr) - 3)
    cut2 = random.randint(cut1 + 1, len(parent1.repr) - 2)
    child_repr = parent1.repr[:cut1] + parent2.repr[cut1:cut2] + parent1.repr[cut2:]
    
    return LeagueSolution(
        repr=child_repr,
        num_teams=parent1.num_teams,
        team_size=parent1.team_size,
        max_budget=parent1.max_budget,
        players=parent1.players
    )
```

### 2.3 Uniform Crossover

**Implementation**: `crossover_uniform` in `operators.py`

**Description**:
Uniform crossover considers each gene independently and randomly selects which parent contributes the gene to the child.

**Parameters**:
- `swap_probability`: Probability of swapping genes between parents (default: 0.5)

**Algorithm**:
1. For each gene position, generate a random number
2. If the number is less than the swap probability, take the gene from the second parent
3. Otherwise, take the gene from the first parent

**Characteristics**:
- Highly disruptive, breaking up building blocks
- Provides maximum exploration of the search space
- Good for problems where gene order doesn't matter
- Useful when building blocks are unknown or not important

**Code Example**:
```python
def crossover_uniform(parent1, parent2, swap_probability=0.5):
    """
    Uniform crossover: creates a child by randomly selecting genes from either parent.
    
    Args:
        parent1: First parent solution
        parent2: Second parent solution
        swap_probability: Probability of taking a gene from the second parent
        
    Returns:
        A new solution created by crossover
    """
    child_repr = []
    
    for i in range(len(parent1.repr)):
        if random.random() < swap_probability:
            child_repr.append(parent2.repr[i])
        else:
            child_repr.append(parent1.repr[i])
    
    return LeagueSolution(
        repr=child_repr,
        num_teams=parent1.num_teams,
        team_size=parent1.team_size,
        max_budget=parent1.max_budget,
        players=parent1.players
    )
```

## 3. Mutation Operators

Mutation operators introduce small random changes to individuals to maintain genetic diversity.

### 3.1 Swap Mutation

**Implementation**: `mutate_swap` in `operators.py`

**Description**:
Swap mutation randomly changes the team assignment of a player in the solution.

**Parameters**:
- `mutation_rate`: Probability of mutation for each position (default: 1/chromosome_length)

**Algorithm**:
1. For each position in the chromosome, generate a random number
2. If the number is less than the mutation rate, change the team assignment to a random value

**Characteristics**:
- Simple and effective
- Maintains the number of players
- Allows exploration of the search space
- May create invalid solutions that need repair

**Code Example**:
```python
def mutate_swap(solution, mutation_rate=None):
    """
    Swap mutation: randomly changes team assignments.
    
    Args:
        solution: Solution to mutate
        mutation_rate: Probability of mutation for each position
        
    Returns:
        Mutated solution
    """
    mutated = deepcopy(solution)
    
    if mutation_rate is None:
        mutation_rate = 1.0 / len(mutated.repr)
    
    for i in range(len(mutated.repr)):
        if random.random() < mutation_rate:
            mutated.repr[i] = random.randint(0, mutated.num_teams - 1)
    
    return mutated
```

### 3.2 Constrained Swap Mutation

**Implementation**: `mutate_swap_constrained` in `operators.py`

**Description**:
Constrained swap mutation is similar to swap mutation but ensures that the resulting solution respects certain constraints.

**Parameters**:
- `mutation_rate`: Probability of mutation for each position (default: 1/chromosome_length)

**Algorithm**:
1. For each position in the chromosome, generate a random number
2. If the number is less than the mutation rate, change the team assignment
3. Check if the change violates constraints
4. If constraints are violated, revert the change or try a different team

**Characteristics**:
- More complex than simple swap mutation
- Ensures valid solutions
- May limit exploration of the search space
- Computationally more intensive

**Code Example**:
```python
def mutate_swap_constrained(solution, mutation_rate=None):
    """
    Constrained swap mutation: randomly changes team assignments while respecting constraints.
    
    Args:
        solution: Solution to mutate
        mutation_rate: Probability of mutation for each position
        
    Returns:
        Mutated solution
    """
    mutated = deepcopy(solution)
    
    if mutation_rate is None:
        mutation_rate = 1.0 / len(mutated.repr)
    
    for i in range(len(mutated.repr)):
        if random.random() < mutation_rate:
            # Store original team
            original_team = mutated.repr[i]
            
            # Try different teams
            valid_teams = list(range(mutated.num_teams))
            random.shuffle(valid_teams)
            
            for team in valid_teams:
                if team != original_team:
                    mutated.repr[i] = team
                    
                    # Check if solution is valid
                    if mutated.is_valid():
                        break
                    
                    # Revert if invalid
                    mutated.repr[i] = original_team
    
    return mutated
```

### 3.3 Scramble Mutation

**Implementation**: `mutate_scramble` in `CIFO_Complete_Pipeline_Final.py`

**Description**:
Scramble mutation selects a random subsequence of the chromosome and randomly shuffles the genes within that subsequence.

**Parameters**:
- `mutation_rate`: Probability of applying the mutation (default: 0.1)

**Algorithm**:
1. Determine if mutation occurs based on mutation rate
2. If mutation occurs, select a random subsequence
3. Shuffle the genes within the subsequence
4. Replace the original subsequence with the shuffled one

**Characteristics**:
- Preserves the gene values but changes their positions
- Good for problems where the order of genes matters
- Provides more exploration than simple swap mutation
- Maintains the number of players in each team

**Code Example**:
```python
def mutate_scramble(solution, mutation_rate=0.1):
    """
    Scramble mutation: randomly selects a subsequence and shuffles it.
    
    Args:
        solution: Solution to mutate
        mutation_rate: Probability of mutation
        
    Returns:
        Mutated solution
    """
    mutated = deepcopy(solution)
    
    # Determine if mutation occurs based on rate
    if random.random() < mutation_rate:
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
```

## 4. Advanced Operators

### 4.1 Local Search in Memetic Algorithm

**Implementation**: `local_search` in `run_memetic_algorithm` function

**Description**:
Local search is used in memetic algorithms to improve individuals through hill climbing.

**Parameters**:
- `iterations`: Number of local search iterations
- `local_search_prob`: Probability of applying local search to an individual

**Algorithm**:
1. Start with an individual from the population
2. Generate a neighbor by making a small change
3. If the neighbor is better, replace the current solution
4. Repeat for a specified number of iterations

**Characteristics**:
- Combines global exploration (GA) with local exploitation (hill climbing)
- Improves convergence speed
- Computationally more intensive
- Very effective for complex optimization problems

**Code Example**:
```python
def local_search(solution, iterations):
    """
    Local search to improve a solution through hill climbing.
    
    Args:
        solution: Solution to improve
        iterations: Number of local search iterations
        
    Returns:
        Improved solution
    """
    best_sol = deepcopy(solution)
    best_fitness = solution.fitness()
    
    for _ in range(iterations):
        # Generate neighbor
        neighbor = deepcopy(best_sol)
        idx = random.randint(0, len(neighbor.repr) - 1)
        neighbor.repr[idx] = random.randint(0, neighbor.num_teams - 1)
        
        neighbor_fitness = neighbor.fitness()
        
        # Accept if better
        if neighbor_fitness < best_fitness:
            best_sol = neighbor
            best_fitness = neighbor_fitness
    
    return best_sol
```

### 4.2 Migration in Island Model

**Implementation**: Migration logic in `run_island_model_ga` function

**Description**:
Migration in the island model involves periodically exchanging individuals between isolated populations (islands).

**Parameters**:
- `migration_interval`: Number of generations between migrations
- `migration_size`: Number of individuals to migrate
- `num_islands`: Number of isolated populations

**Algorithm**:
1. Evolve each island independently for a specified number of generations
2. Select the best individuals from each island as migrants
3. Replace the worst individuals in the next island with these migrants
4. Continue evolution with the updated populations

**Characteristics**:
- Maintains genetic diversity by isolating populations
- Prevents premature convergence
- Allows different islands to explore different regions of the search space
- Periodic migration shares good solutions between islands

**Code Example**:
```python
# Migration between islands
if generation % config['migration_interval'] == 0 and generation > 0:
    for i in range(num_islands):
        # Select migrants (best individuals)
        migrant_indices = np.argsort(island_fitness[i])[:config['migration_size']]
        migrants = [deepcopy(islands[i][idx]) for idx in migrant_indices]
        
        # Send to next island (ring topology)
        next_island = (i + 1) % num_islands
        
        # Replace worst individuals in next island
        worst_indices = np.argsort(island_fitness[next_island])[-config['migration_size']:]
        for j, idx in enumerate(worst_indices):
            islands[next_island][idx] = migrants[j]
        
        # Re-evaluate fitness of next island
        island_fitness[next_island] = [solution.fitness() for solution in islands[next_island]]
```

## 5. Operator Selection Guidelines

### 5.1 Selection Operators

- **Tournament Selection**: Use when you want a simple, efficient selection method with adjustable selection pressure.
- **Rank-Based Selection**: Use when fitness values are close together or when you want to reduce the impact of outliers.
- **Boltzmann Selection**: Use when you want to dynamically adjust selection pressure during evolution.

### 5.2 Crossover Operators

- **One-Point Crossover**: Use for simple problems or when gene order matters.
- **Two-Point Crossover**: Use when you want to preserve building blocks at the ends of the chromosome.
- **Uniform Crossover**: Use when you want maximum exploration or when gene order doesn't matter.

### 5.3 Mutation Operators

- **Swap Mutation**: Use as a general-purpose mutation operator.
- **Constrained Swap Mutation**: Use when you need to ensure valid solutions.
- **Scramble Mutation**: Use when you want to preserve gene values but change their positions.

### 5.4 Advanced Operators

- **Local Search**: Use in memetic algorithms to improve convergence speed.
- **Migration**: Use in island models to maintain genetic diversity.

## 6. Performance Considerations

### 6.1 Computational Complexity

- **Tournament Selection**: O(tournament_size)
- **Rank-Based Selection**: O(n log n) due to sorting
- **Boltzmann Selection**: O(n) for probability calculation
- **One-Point/Two-Point Crossover**: O(n) where n is chromosome length
- **Uniform Crossover**: O(n) where n is chromosome length
- **Swap Mutation**: O(n) where n is chromosome length
- **Local Search**: O(iterations * n) where n is chromosome length
- **Island Model**: O(islands * population_size * generations)

### 6.2 Memory Usage

- Most operators have minimal memory overhead
- Island Model requires more memory to maintain multiple populations
- Memetic algorithms may require additional memory for local search

### 6.3 Parallelization

- Island Model is naturally parallelizable (each island can evolve independently)
- Selection, crossover, and mutation operators can be parallelized for large populations
- Local search in memetic algorithms can be parallelized across individuals

## 7. Conclusion

The choice of genetic operators significantly impacts the performance of evolutionary algorithms. Understanding the characteristics and trade-offs of different operators allows for better algorithm design and problem-solving.

For the Fantasy League Team Optimization problem, our experiments showed that:

1. Tournament selection provided the best balance between exploration and exploitation
2. Two-Point crossover preserved important building blocks better than other methods
3. Scramble mutation improved exploration compared to standard swap mutation
4. The combination of these operators in a memetic algorithm or island model produced the best results
