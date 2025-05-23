from abc import ABC, abstractmethod
import random
import numpy as np
from copy import deepcopy

class Solution(ABC):
    """
    Abstract base class for all solutions.
    
    This class defines the interface that all solution classes must implement.
    """
    def __init__(self, repr=None):
        # To initialize a solution we need to know its representation. 
        # If no representation is given, a solution is randomly initialized.
        if repr == None:
            repr = self.random_initial_representation()
        # Attributes
        self.repr = repr

    # Method that is called when we run print(object of the class)
    def __repr__(self):
        return str(self.repr)

    # Other methods that must be implemented in subclasses
    @abstractmethod
    def fitness(self):
        pass

    @abstractmethod
    def random_initial_representation(self):
        pass


class LeagueSolution(Solution):
    """
    Solution class for the Sports League optimization problem.
    
    A solution is represented as a list of team assignments for each player.
    For example, repr[0] = 2 means player 0 is assigned to team 2.
    """
    def __init__(self, repr=None, num_teams=5, team_size=7, max_budget=750, players=None):
        self.num_teams = num_teams
        self.team_size = team_size
        self.max_budget = max_budget
        self.players = players
        super().__init__(repr=repr)

    def random_initial_representation(self):
        """Generate a random assignment of players to teams."""
        team_ids = [i % self.num_teams for i in range(self.num_teams * self.team_size)]
        random.shuffle(team_ids)
        return team_ids

    def is_valid(self):
        """
        Check if the solution respects all constraints:
        - Each team has exactly team_size players
        - Each team has the correct position distribution (1 GK, 2 DEF, 2 MID, 2 FWD)
        - Each team's total salary is within budget
        """
        if not self.players:
            return False
            
        teams = [[] for _ in range(self.num_teams)]
        for idx, team_id in enumerate(self.repr):
            teams[team_id].append(self.players[idx])
        
        for team in teams:
            if len(team) != self.team_size:
                return False
            
            roles = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
            budget = 0
            for p in team:
                roles[p["Position"]] += 1
                # Handle different salary field names
                if "Salary" in p:
                    budget += p["Salary"]
                elif "Salary (€M)" in p:
                    budget += p["Salary (€M)"]
                else:
                    # Default to 0 if no salary field is found
                    print(f"Warning: No salary field found for player {p}")
                    budget += 0
            
            if roles != {"GK": 1, "DEF": 2, "MID": 2, "FWD": 2}:
                return False
            
            if budget > self.max_budget:
                return False
        
        return True

    def fitness(self):
        """
        Calculate the fitness of the solution.
        Lower fitness is better (minimization problem).
        
        Returns infinity for invalid solutions.
        """
        if not self.is_valid():
            return float("inf")
        
        team_skills = [[] for _ in range(self.num_teams)]
        for idx, team_id in enumerate(self.repr):
            team_skills[team_id].append(self.players[idx]["Skill"])
        
        avg_skills = [np.mean(team) for team in team_skills]
        return np.std(avg_skills)

    def get_teams(self):
        """
        Get the teams formed by the current assignment.
        Returns a list of teams, where each team is a list of player dictionaries.
        """
        teams = [[] for _ in range(self.num_teams)]
        for idx, team_id in enumerate(self.repr):
            teams[team_id].append(self.players[idx])
        return teams
    
    def get_team_stats(self):
        """
        Calculate statistics for each team.
        Returns a list of dictionaries with team statistics.
        """
        teams = self.get_teams()
        stats = []
        
        for i, team in enumerate(teams):
            avg_skill = np.mean([p["Skill"] for p in team])
            
            # Handle different salary field names
            total_salary = 0
            for p in team:
                if "Salary" in p:
                    total_salary += p["Salary"]
                elif "Salary (€M)" in p:
                    total_salary += p["Salary (€M)"]
                else:
                    # Default to 0 if no salary field is found
                    print(f"Warning: No salary field found for player {p}")
            
            positions = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
            for p in team:
                positions[p["Position"]] += 1
                
            stats.append({
                "team_id": i,
                "avg_skill": avg_skill,
                "total_salary": total_salary,
                "positions": positions,
                "players": team
            })
            
        return stats


class LeagueHillClimbingSolution(LeagueSolution):
    """
    Extension of LeagueSolution for Hill Climbing algorithm.
    Adds methods for generating and evaluating neighbors.
    """
    def get_neighbors(self):
        """
        Generate valid neighboring solutions by swapping players between teams.
        Returns a list of valid neighbor solutions.
        """
        neighbors = []
        
        # Create a position map for efficient neighbor generation
        position_map = {}
        for idx, player in enumerate(self.players):
            pos = player["Position"]
            if pos not in position_map:
                position_map[pos] = []
            position_map[pos].append(idx)
        
        # Generate neighbors by swapping players of the same position
        for pos, indices in position_map.items():
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx1, idx2 = indices[i], indices[j]
                    
                    # Only swap players from different teams
                    if self.repr[idx1] != self.repr[idx2]:
                        new_repr = self.repr[:]
                        new_repr[idx1], new_repr[idx2] = new_repr[idx2], new_repr[idx1]
                        
                        neighbor = LeagueHillClimbingSolution(
                            repr=new_repr, 
                            num_teams=self.num_teams, 
                            team_size=self.team_size, 
                            max_budget=self.max_budget,
                            players=self.players
                        )
                        
                        if neighbor.is_valid():
                            neighbors.append(neighbor)
        
        return neighbors


class LeagueSASolution(LeagueSolution):
    """
    Extension of LeagueSolution for Simulated Annealing algorithm.
    Adds methods for generating random neighbors.
    """
    def get_random_neighbor(self):
        """
        Generate a random valid neighboring solution by swapping two players.
        Returns a new LeagueSASolution instance.
        """
        # Create a position map for efficient neighbor generation
        position_map = {}
        for idx, player in enumerate(self.players):
            pos = player["Position"]
            if pos not in position_map:
                position_map[pos] = []
            position_map[pos].append(idx)
        
        # Try to find a valid neighbor by swapping players of the same position
        max_attempts = 50  # Limit attempts to avoid infinite loops
        
        for _ in range(max_attempts):
            # Randomly select a position
            pos = random.choice(list(position_map.keys()))
            
            # Need at least 2 players of this position to swap
            if len(position_map[pos]) < 2:
                continue
                
            # Select two random players of this position
            idx1, idx2 = random.sample(position_map[pos], 2)
            
            # Only swap players from different teams
            if self.repr[idx1] != self.repr[idx2]:
                new_repr = self.repr[:]
                new_repr[idx1], new_repr[idx2] = new_repr[idx2], new_repr[idx1]
                
                neighbor = LeagueSASolution(
                    repr=new_repr, 
                    num_teams=self.num_teams, 
                    team_size=self.team_size, 
                    max_budget=self.max_budget,
                    players=self.players
                )
                
                if neighbor.is_valid():
                    return neighbor
        
        # If we couldn't find a valid neighbor, return a copy of self
        return LeagueSASolution(
            repr=self.repr[:], 
            num_teams=self.num_teams, 
            team_size=self.team_size, 
            max_budget=self.max_budget,
            players=self.players
        )
