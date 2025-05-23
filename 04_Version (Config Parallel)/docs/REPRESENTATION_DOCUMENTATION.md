# Solution Representation Documentation

This document provides a detailed explanation of the solution representation used in the Fantasy League Team Optimization project.

## 1. Problem Definition

The Fantasy League Team Optimization problem involves creating balanced teams of players while respecting position and budget constraints.

### 1.1 Objective

The primary objective is to minimize the standard deviation of average skills across teams, creating balanced competition.

### 1.2 Constraints

1. **Position Constraints**: Each team must have:
   - 1 Goalkeeper (GK)
   - 2 Defenders (DEF)
   - 2 Midfielders (MID)
   - 2 Forwards (FWD)

2. **Budget Constraints**: Each team must stay within a maximum budget.

3. **Player Assignment**: Each player must be assigned to exactly one team.

## 2. Solution Representation

### 2.1 Chromosome Structure

The solution is represented as a list of integers, where:
- Each position in the list corresponds to a player
- The value at each position indicates the team to which the player is assigned

For example, with 35 players and 5 teams:
```
[0, 2, 1, 4, 3, 0, 2, 1, 4, 3, 0, 2, 1, 4, 3, 0, 2, 1, 4, 3, 0, 2, 1, 4, 3, 0, 2, 1, 4, 3, 0, 2, 1, 4, 3]
```

In this representation:
- Player 0 is assigned to team 0
- Player 1 is assigned to team 2
- Player 2 is assigned to team 1
- And so on...

### 2.2 Implementation

The solution representation is implemented in the `LeagueSolution` class in `solution.py`:

```python
class LeagueSolution:
    def __init__(self, players, repr=None, num_teams=5, team_size=7, max_budget=750):
        self.players = players
        self.num_teams = num_teams
        self.team_size = team_size
        self.max_budget = max_budget
        self.fitness_counter = None
        
        # Initialize representation
        if repr is None:
            # Random initialization
            self.repr = [random.randint(0, num_teams - 1) for _ in range(len(players))]
        else:
            # Use provided representation
            self.repr = repr
```

### 2.3 Advantages of this Representation

1. **Simplicity**: The representation is straightforward and easy to understand.
2. **Fixed Length**: The chromosome length is fixed (equal to the number of players).
3. **Efficiency**: Team assignments can be quickly modified and evaluated.
4. **Compatibility**: Works well with standard genetic operators (selection, crossover, mutation).

### 2.4 Disadvantages of this Representation

1. **Invalid Solutions**: Can easily generate invalid solutions that violate constraints.
2. **Redundancy**: Multiple representations can map to the same team composition.
3. **Indirect Mapping**: Requires additional processing to extract team compositions.

## 3. Solution Validation

### 3.1 Validity Checks

A solution is considered valid if it satisfies all constraints:

```python
def is_valid(self):
    """
    Check if the solution is valid (respects all constraints).
    
    Returns:
        bool: True if valid, False otherwise
    """
    teams = self.get_teams()
    
    for team in teams:
        # Check team size
        if len(team) != self.team_size:
            return False
        
        # Check position constraints
        positions = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
        for player in team:
            positions[player["Position"]] += 1
        
        if positions["GK"] != 1 or positions["DEF"] != 2 or positions["MID"] != 2 or positions["FWD"] != 2:
            return False
        
        # Check budget constraint
        total_salary = sum(player["Salary"] for player in team)
        if total_salary > self.max_budget:
            return False
    
    return True
```

### 3.2 Team Extraction

The `get_teams` method extracts the team compositions from the representation:

```python
def get_teams(self):
    """
    Extract teams from the representation.
    
    Returns:
        list: List of teams, where each team is a list of player dictionaries
    """
    teams = [[] for _ in range(self.num_teams)]
    
    for i, team_id in enumerate(self.repr):
        if 0 <= team_id < self.num_teams:
            teams[team_id].append(self.players[i])
    
    return teams
```

### 3.3 Team Statistics

The `get_team_stats` method calculates statistics for each team:

```python
def get_team_stats(self):
    """
    Calculate statistics for each team.
    
    Returns:
        list: List of team statistics dictionaries
    """
    teams = self.get_teams()
    stats = []
    
    for i, team in enumerate(teams):
        positions = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
        for player in team:
            positions[player["Position"]] += 1
        
        avg_skill = sum(player["Skill"] for player in team) / len(team) if team else 0
        total_salary = sum(player["Salary"] for player in team)
        
        stats.append({
            "team_id": i,
            "players": team,
            "avg_skill": avg_skill,
            "total_salary": total_salary,
            "positions": positions
        })
    
    return stats
```

## 4. Fitness Function

### 4.1 Objective Function

The fitness function measures the quality of a solution. For the Fantasy League Team Optimization problem, the objective is to minimize the standard deviation of average skills across teams:

```python
def fitness(self):
    """
    Calculate fitness of the solution (lower is better).
    
    Returns:
        float: Fitness value
    """
    # Count fitness evaluation if counter is set
    if self.fitness_counter is not None:
        self.fitness_counter.increment()
    
    # Get team statistics
    team_stats = self.get_team_stats()
    
    # Extract average skills
    avg_skills = [stat["avg_skill"] for stat in team_stats]
    
    # Calculate standard deviation of average skills
    if len(avg_skills) <= 1:
        return float('inf')  # Invalid solution
    
    std_dev = np.std(avg_skills)
    
    # Check if solution is valid
    if not self.is_valid():
        return float('inf')  # Invalid solution
    
    return std_dev
```

### 4.2 Penalty Function

In some variants, a penalty function is used to handle constraint violations:

```python
def fitness_with_penalty(self, penalty_factor=10.0):
    """
    Calculate fitness with penalty for constraint violations.
    
    Args:
        penalty_factor: Weight for penalty term
        
    Returns:
        float: Fitness value with penalty
    """
    # Count fitness evaluation if counter is set
    if self.fitness_counter is not None:
        self.fitness_counter.increment()
    
    # Get team statistics
    team_stats = self.get_team_stats()
    
    # Extract average skills
    avg_skills = [stat["avg_skill"] for stat in team_stats]
    
    # Calculate standard deviation of average skills
    if len(avg_skills) <= 1:
        return float('inf')  # Invalid solution
    
    std_dev = np.std(avg_skills)
    
    # Calculate penalty for constraint violations
    penalty = 0.0
    
    # Position constraint violations
    for stat in team_stats:
        positions = stat["positions"]
        penalty += abs(positions["GK"] - 1)
        penalty += abs(positions["DEF"] - 2)
        penalty += abs(positions["MID"] - 2)
        penalty += abs(positions["FWD"] - 2)
    
    # Budget constraint violations
    for stat in team_stats:
        if stat["total_salary"] > self.max_budget:
            penalty += (stat["total_salary"] - self.max_budget) / 100.0
    
    # Apply penalty
    return std_dev + penalty_factor * penalty
```

### 4.3 Fitness Counter

To track the number of fitness evaluations, a `FitnessCounter` is used:

```python
def set_fitness_counter(self, counter):
    """
    Set fitness counter for tracking evaluations.
    
    Args:
        counter: FitnessCounter instance
    """
    self.fitness_counter = counter
```

## 5. Solution Initialization

### 5.1 Random Initialization

By default, solutions are initialized randomly:

```python
# Random initialization
self.repr = [random.randint(0, num_teams - 1) for _ in range(len(players))]
```

### 5.2 Valid Initialization

For some algorithms, valid initialization is used to ensure that the initial solution respects all constraints:

```python
def initialize_valid_solution(players, num_teams=5, team_size=7, max_budget=750):
    """
    Initialize a valid solution that respects all constraints.
    
    Args:
        players: List of player dictionaries
        num_teams: Number of teams
        team_size: Number of players per team
        max_budget: Maximum budget per team
        
    Returns:
        LeagueSolution: Valid initial solution
    """
    # Group players by position
    positions = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for i, player in enumerate(players):
        positions[player["Position"]].append(i)
    
    # Initialize representation with -1 (unassigned)
    repr = [-1] * len(players)
    
    # Assign players to teams
    for team_id in range(num_teams):
        # Select 1 GK, 2 DEF, 2 MID, 2 FWD for each team
        team_players = []
        team_players.extend(random.sample(positions["GK"], 1))
        team_players.extend(random.sample(positions["DEF"], 2))
        team_players.extend(random.sample(positions["MID"], 2))
        team_players.extend(random.sample(positions["FWD"], 2))
        
        # Assign team_id to selected players
        for player_idx in team_players:
            repr[player_idx] = team_id
        
        # Remove assigned players from available pools
        for player_idx in team_players:
            position = players[player_idx]["Position"]
            positions[position].remove(player_idx)
    
    # Assign remaining players randomly
    for i in range(len(repr)):
        if repr[i] == -1:
            repr[i] = random.randint(0, num_teams - 1)
    
    return LeagueSolution(
        repr=repr,
        num_teams=num_teams,
        team_size=team_size,
        max_budget=max_budget,
        players=players
    )
```

## 6. Solution Manipulation

### 6.1 Neighbor Generation

For local search algorithms like Hill Climbing and Simulated Annealing, neighbors are generated by changing the team assignment of a random player:

```python
# Generate neighbor
neighbor = deepcopy(solution)
idx = random.randint(0, len(neighbor.repr) - 1)
neighbor.repr[idx] = random.randint(0, neighbor.num_teams - 1)
```

### 6.2 Crossover

Crossover operators combine genetic material from two parent solutions to create offspring. See the Operators Documentation for details on crossover operators.

### 6.3 Mutation

Mutation operators introduce small random changes to individuals to maintain genetic diversity. See the Operators Documentation for details on mutation operators.

## 7. Solution Repair

### 7.1 Repair Operator

For some algorithms, a repair operator is used to fix invalid solutions:

```python
def repair_solution(solution):
    """
    Repair an invalid solution to make it valid.
    
    Args:
        solution: Solution to repair
        
    Returns:
        LeagueSolution: Repaired solution
    """
    repaired = deepcopy(solution)
    teams = repaired.get_teams()
    
    # Fix position constraints
    for team_id, team in enumerate(teams):
        positions = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
        for player in team:
            positions[player["Position"]] += 1
        
        # Find players to remove or add
        excess = {}
        deficit = {}
        for pos, count in positions.items():
            required = 1 if pos == "GK" else 2
            if count > required:
                excess[pos] = count - required
            elif count < required:
                deficit[pos] = required - count
        
        # Remove excess players
        for pos, count in excess.items():
            pos_players = [i for i, p in enumerate(repaired.players) 
                          if p["Position"] == pos and repaired.repr[i] == team_id]
            
            # Sort by skill (remove lowest skill first)
            pos_players.sort(key=lambda i: repaired.players[i]["Skill"])
            
            # Remove excess players
            for i in range(min(count, len(pos_players))):
                # Assign to random other team
                new_team = random.randint(0, repaired.num_teams - 1)
                while new_team == team_id:
                    new_team = random.randint(0, repaired.num_teams - 1)
                
                repaired.repr[pos_players[i]] = new_team
        
        # Add deficit players
        for pos, count in deficit.items():
            available = [i for i, p in enumerate(repaired.players) 
                        if p["Position"] == pos and repaired.repr[i] != team_id]
            
            # Sort by skill (add highest skill first)
            available.sort(key=lambda i: repaired.players[i]["Skill"], reverse=True)
            
            # Add deficit players
            for i in range(min(count, len(available))):
                repaired.repr[available[i]] = team_id
    
    # Fix budget constraints
    for team_id, team in enumerate(teams):
        total_salary = sum(player["Salary"] for player in team)
        
        if total_salary > repaired.max_budget:
            # Sort players by salary (remove highest salary first)
            team_players = [(i, repaired.players[i]) 
                           for i in range(len(repaired.repr)) 
                           if repaired.repr[i] == team_id]
            
            team_players.sort(key=lambda x: x[1]["Salary"], reverse=True)
            
            # Remove players until budget constraint is satisfied
            for i, player in team_players:
                # Assign to random other team
                new_team = random.randint(0, repaired.num_teams - 1)
                while new_team == team_id:
                    new_team = random.randint(0, repaired.num_teams - 1)
                
                repaired.repr[i] = new_team
                
                # Recalculate total salary
                team = [repaired.players[j] for j in range(len(repaired.repr)) 
                       if repaired.repr[j] == team_id]
                
                total_salary = sum(player["Salary"] for player in team)
                
                if total_salary <= repaired.max_budget:
                    break
    
    return repaired
```

## 8. Solution Visualization

### 8.1 Team Composition Display

The solution can be visualized by displaying the team compositions:

```python
def display_team_solution(solution):
    """
    Display the team composition of a solution.
    
    Args:
        solution: Solution to display
    """
    team_stats = solution.get_team_stats()
    
    print("\nTeam Statistics:")
    print(f"{'Team':<10} {'Avg Skill':<15} {'Total Salary':<15} {'GK':<5} {'DEF':<5} {'MID':<5} {'FWD':<5}")
    print("-" * 65)
    
    for stat in team_stats:
        positions = stat["positions"]
        print(f"Team {stat['team_id']+1:<5} {stat['avg_skill']:<15.2f} {stat['total_salary']:<15.2f} "
              f"{positions['GK']:<5} {positions['DEF']:<5} {positions['MID']:<5} {positions['FWD']:<5}")
    
    print("\nDetailed Team Composition:")
    
    for stat in team_stats:
        print(f"\nTeam {stat['team_id']+1}:")
        print(f"{'Name':<20} {'Position':<10} {'Skill':<10} {'Salary':<10}")
        print("-" * 50)
        
        for player in stat["players"]:
            print(f"{player['Name']:<20} {player['Position']:<10} {player['Skill']:<10.2f} {player['Salary']:<10.2f}")
        
        print(f"Average Skill: {stat['avg_skill']:.2f}")
        print(f"Total Salary: {stat['total_salary']:.2f}")
    
    # Calculate overall statistics
    avg_skills = [stat["avg_skill"] for stat in team_stats]
    overall_std = np.std(avg_skills)
    
    print("\nOverall Team Balance:")
    print(f"Standard Deviation of Average Skills: {overall_std:.4f}")
    print(f"This matches the fitness value: {solution.fitness():.4f}")
```

### 8.2 Graphical Visualization

For more advanced visualization, matplotlib can be used to create graphical representations of the solution:

```python
def plot_team_solution(solution):
    """
    Create a graphical visualization of the team solution.
    
    Args:
        solution: Solution to visualize
    """
    team_stats = solution.get_team_stats()
    
    # Plot average skills
    plt.figure(figsize=(10, 6))
    teams = [f"Team {stat['team_id']+1}" for stat in team_stats]
    avg_skills = [stat["avg_skill"] for stat in team_stats]
    
    plt.bar(teams, avg_skills)
    plt.title("Average Skill by Team")
    plt.xlabel("Team")
    plt.ylabel("Average Skill")
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(avg_skills):
        plt.text(i, v + 0.1, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.show()
    
    # Plot position distribution
    plt.figure(figsize=(12, 8))
    positions = ["GK", "DEF", "MID", "FWD"]
    
    for i, stat in enumerate(team_stats):
        pos_counts = [stat["positions"][pos] for pos in positions]
        plt.subplot(2, 3, i+1)
        plt.bar(positions, pos_counts)
        plt.title(f"Team {stat['team_id']+1}")
        plt.ylim(0, 3)
        
        # Add value labels
        for j, v in enumerate(pos_counts):
            plt.text(j, v + 0.1, str(v), ha='center')
    
    plt.tight_layout()
    plt.show()
    
    # Plot salary distribution
    plt.figure(figsize=(10, 6))
    total_salaries = [stat["total_salary"] for stat in team_stats]
    
    plt.bar(teams, total_salaries)
    plt.title("Total Salary by Team")
    plt.xlabel("Team")
    plt.ylabel("Total Salary (€M)")
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(total_salaries):
        plt.text(i, v + 10, f"{v:.2f}", ha='center')
    
    # Add budget line
    plt.axhline(y=solution.max_budget, color='r', linestyle='--', label=f"Budget Limit ({solution.max_budget})")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
```

## 9. Specialized Solution Classes

### 9.1 Hill Climbing Solution

For Hill Climbing, a specialized solution class is used:

```python
class LeagueHillClimbingSolution(LeagueSolution):
    """
    Specialized solution class for Hill Climbing algorithm.
    """
    
    def get_neighbors(self, num_neighbors=1):
        """
        Generate random neighbors of the current solution.
        
        Args:
            num_neighbors: Number of neighbors to generate
            
        Returns:
            list: List of neighbor solutions
        """
        neighbors = []
        
        for _ in range(num_neighbors):
            neighbor = deepcopy(self)
            idx = random.randint(0, len(neighbor.repr) - 1)
            neighbor.repr[idx] = random.randint(0, neighbor.num_teams - 1)
            neighbors.append(neighbor)
        
        return neighbors
    
    def get_best_neighbor(self, num_neighbors=10):
        """
        Generate multiple neighbors and return the best one.
        
        Args:
            num_neighbors: Number of neighbors to generate
            
        Returns:
            LeagueHillClimbingSolution: Best neighbor
        """
        neighbors = self.get_neighbors(num_neighbors)
        
        # Find best neighbor
        best_neighbor = None
        best_fitness = float('inf')
        
        for neighbor in neighbors:
            fitness = neighbor.fitness()
            if fitness < best_fitness:
                best_neighbor = neighbor
                best_fitness = fitness
        
        return best_neighbor
```

## 10. Conclusion

The solution representation is a critical component of the Fantasy League Team Optimization project. The integer-based representation used in this project offers a good balance between simplicity and effectiveness, allowing for efficient manipulation and evaluation of solutions.

Key aspects of the solution representation include:
1. **Integer-based encoding**: Each player is assigned a team ID
2. **Constraint handling**: Position and budget constraints are checked during validation
3. **Fitness calculation**: Standard deviation of average skills is minimized
4. **Solution manipulation**: Neighbors, crossover, and mutation operators modify the representation
5. **Visualization**: Team compositions and statistics can be displayed

This representation has proven effective for various optimization algorithms, including Hill Climbing, Simulated Annealing, and Genetic Algorithms, allowing for fair comparison between different approaches.
