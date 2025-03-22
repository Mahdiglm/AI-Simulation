"""
Population class for managing a collection of dots.
"""
import random
from typing import List, Tuple, Dict, Any

from dot import Dot
from tf_neural_network import TFNeuralNetwork
from neural_network import NeuralNetwork  # The pure NumPy implementation
from obstacle import Obstacle
from constants import YELLOW

class Population:
    """Manages a population of AI dots."""
    
    def __init__(self, size: int, screen_width: int, screen_height: int, config: Dict[str, Any] = None):
        """Initialize population of dots."""
        self.size = size
        self.dots = []
        self.generation = 1
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.best_dot_index = 0
        self.min_steps = float('inf')
        
        # Configuration parameters
        self.config = config or {}
        self.crossover_rate = self.config.get('crossover_rate', 0.75)  # Probability of crossover vs. cloning
        self.mutation_rate = self.config.get('mutation_rate', 0.05)    # Probability of mutation per weight/bias
        self.elitism_count = self.config.get('elitism_count', 2)       # Number of top performers to preserve unchanged
        self.use_neural_network_fast_path = self.config.get('use_neural_network_fast_path', True)
        self.use_tensorflow = self.config.get('use_tensorflow', True)
        
        # Create initial population
        for _ in range(size):
            x = screen_width // 10
            y = screen_height // 2
            
            # Create a brain based on configuration
            if self.use_tensorflow:
                # Use TensorFlow-based neural network
                brain = TFNeuralNetwork(
                    15, 16, 3, 
                    hidden_layers=2, 
                    use_numpy_fast_path=self.use_neural_network_fast_path
                )
            else:
                # Use pure NumPy-based neural network
                brain = NeuralNetwork(15, 16, 3, hidden_layers=2)
                
            self.dots.append(Dot(x, y, brain=brain))
    
    def update(self, target: Tuple[float, float], obstacles: List[Obstacle]) -> bool:
        """Update all dots in the population.
        
        Returns:
            bool: True if all dots are dead or have reached the target.
        """
        all_dead_or_reached = True
        
        for dot in self.dots:
            if not dot.dead and not dot.reached_goal:
                all_dead_or_reached = False
                dot.update(target, obstacles)
        
        return all_dead_or_reached
        
    def calculate_fitnesses(self, target: Tuple[float, float]) -> None:
        """Calculate fitness for all dots."""
        for dot in self.dots:
            dot.calculate_fitness(target)
    
    def select_parent(self) -> Dot:
        """Select a parent based on fitness (tournament selection)."""
        # Tournament selection (select the best from a random subset)
        tournament_size = min(5, len(self.dots))
        best_dot = random.choice(self.dots)
        best_fitness = best_dot.fitness
        
        # Pick a few random dots and select the one with highest fitness
        for _ in range(tournament_size - 1):
            dot = random.choice(self.dots)
            if dot.fitness > best_fitness:
                best_dot = dot
                best_fitness = dot.fitness
                
        return best_dot
    
    def natural_selection(self) -> None:
        """Select best dots for reproduction."""
        new_dots = []
        self.calculate_fitnesses(target=(self.screen_width - 100, self.screen_height // 2))
        
        # Find and sort dots by fitness
        sorted_dots = sorted(self.dots, key=lambda d: d.fitness, reverse=True)
        self.best_dot_index = self.dots.index(sorted_dots[0])
        
        # Save best dot's steps if it reached the goal
        if sorted_dots[0].reached_goal:
            self.min_steps = min(self.min_steps, sorted_dots[0].steps)
        
        # Elite selection: Top performers go directly to next generation
        for i in range(min(self.elitism_count, len(sorted_dots))):
            elite_brain = sorted_dots[i].brain.copy()
            elite_dot = Dot(
                self.screen_width // 10, 
                self.screen_height // 2,
                elite_brain,
                sorted_dots[i].max_steps
            )
            
            # Mark the best dot for visualization (handled in the drawing code)
            new_dots.append(elite_dot)
        
        # Create the rest of the population through selection and crossover
        while len(new_dots) < self.size:
            # Select two parents based on fitness
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            
            # Create a child through crossover or cloning
            if random.random() < self.crossover_rate and parent1 != parent2:
                # Use the neural network's crossover method
                child_brain = parent1.brain.crossover(parent2.brain)
            else:
                # Cloning (with preference for the fitter parent)
                if parent1.fitness > parent2.fitness:
                    child_brain = parent1.brain.copy()
                else:
                    child_brain = parent2.brain.copy()
            
            # Create a new dot with the child brain
            child = Dot(
                self.screen_width // 10,
                self.screen_height // 2,
                child_brain,
                parent1.max_steps
            )
            
            # Apply mutation
            child.brain.mutate(self.mutation_rate)
            
            # Add to new population
            new_dots.append(child)
        
        self.dots = new_dots
        self.generation += 1
        
    def draw(self, surface) -> None:
        """Draw all dots on the given surface."""
        for dot in self.dots:
            dot.draw(surface) 