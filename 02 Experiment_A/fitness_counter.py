"""
Fitness counter module for tracking function evaluations across algorithms.

This module provides a robust way to count fitness function evaluations
regardless of how the fitness function is called in different algorithms.
"""

class FitnessCounter:
    """
    A class to count fitness function evaluations.
    
    This implementation uses a decorator pattern to wrap the original
    fitness method rather than monkey-patching it, ensuring compatibility
    with all algorithms and preventing the 'missing argument' error.
    """
    def __init__(self):
        self.count = 0
        self.original_fitness = None
        self.active = False
    
    def reset(self):
        """
        Reset the fitness evaluation counter to zero.
        
        Returns:
            None
        """
        self.count = 0
    
    def start_counting(self, solution_class):
        """
        Start counting fitness evaluations for the given solution class.
        
        Args:
            solution_class: The solution class whose fitness method will be counted
        """
        self.count = 0
        self.original_fitness = solution_class.fitness
        solution_class.fitness = self._create_counting_wrapper(solution_class)
        self.active = True
    
    def stop_counting(self, solution_class):
        """
        Stop counting fitness evaluations and restore the original fitness method.
        
        Args:
            solution_class: The solution class whose fitness method was being counted
            
        Returns:
            int: The number of fitness evaluations counted
        """
        if self.active and self.original_fitness:
            solution_class.fitness = self.original_fitness
            self.active = False
        return self.count
    
    def _create_counting_wrapper(self, solution_class):
        """
        Create a wrapper function that counts calls to the fitness method.
        
        Args:
            solution_class: The solution class whose fitness method is being wrapped
            
        Returns:
            function: A wrapped version of the fitness method that counts calls
        """
        original_fitness = self.original_fitness
        counter = self
        
        def counting_fitness(self):
            counter.count += 1
            return original_fitness(self)
        
        return counting_fitness
    
    def get_count(self):
        """
        Get the current count of fitness evaluations.
        
        Returns:
            int: The number of fitness evaluations counted
        """
        return self.count

# Create a global instance for convenience
fitness_counter = FitnessCounter()
