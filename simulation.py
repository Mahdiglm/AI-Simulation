"""
Simulation module for managing the AI simulation environment.

This module provides the central simulation manager that integrates various
components (dots, population, neural networks, obstacles) and handles the
overall simulation logic.
"""
import os
import time
import json
import datetime
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pygame
import argparse

from dot import Dot
from population import Population
from tf_neural_network import TFNeuralNetwork
from neural_network import NeuralNetwork
from obstacle import Obstacle
from constants import GREEN, RED, YELLOW, GRAY


class Simulation:
    """
    Main simulation class that manages the AI simulation environment.
    
    This class handles the setup, execution, and management of the simulation,
    including creation of the environment, management of the population,
    visualization, and data collection.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the simulation with the given configuration.
        
        Args:
            config: Dictionary containing simulation configuration parameters.
                   If None, default configuration will be used.
        """
        # Default configuration
        self.default_config = {
            'screen_width': 800,
            'screen_height': 600,
            'population_size': 100,
            'dot_count': 50,  # Number of dots to display (for visual clarity)
            'max_steps': 400,
            'obstacle_count': 5,
            'memory_usage': 'medium',  # low, medium, high
            'mutation_rate': 0.05,
            'max_generations': 100,
            'fps': 60,
            'auto_checkpoint_interval': 10,  # Save checkpoint every N generations
            'use_neural_network_fast_path': True,  # Use fast numpy path for neural network
            'use_tensorflow': True,  # If False, will use pure NumPy implementation instead
        }
        
        # Use provided config or default
        self.config = config or self.default_config
        
        # Initialize pygame if not already initialized
        if not pygame.get_init():
            pygame.init()
        
        # Setup screen and clock
        self.screen_width = self.config.get('screen_width', 800)
        self.screen_height = self.config.get('screen_height', 600)
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
        # Initialize game state
        self.running = False
        self.paused = False
        self.generation = 1
        self.speed_multiplier = 1
        self.show_sensors = True
        self.show_all_dots = True
        self.show_graph = True
        
        # Set up simulation objects
        self.population = None
        self.obstacles = []
        self.target_pos = (self.screen_width - 100, self.screen_height // 2)
        
        # Statistics tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.generation_history = []
        
        # Setup directories for saving data
        self.setup_directories()
        
    def setup_directories(self) -> None:
        """
        Create necessary directories for saving simulation data.
        """
        # Create 'runs' directory for saving run data
        self.runs_dir = os.path.join(os.getcwd(), 'runs')
        os.makedirs(self.runs_dir, exist_ok=True)
        
        # Create directory for current run based on timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_run_dir = os.path.join(self.runs_dir, f"run_{timestamp}")
        os.makedirs(self.current_run_dir, exist_ok=True)
        
        # Create subdirectories for checkpoints and stats
        self.checkpoints_dir = os.path.join(self.current_run_dir, 'checkpoints')
        self.stats_dir = os.path.join(self.current_run_dir, 'stats')
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)

    def initialize_population(self) -> None:
        """
        Initialize the population of dots.
        """
        population_size = self.config.get('population_size', 100)
        max_steps = self.config.get('max_steps', 400)
        
        self.population = Population(
            population_size, 
            self.screen_width, 
            self.screen_height,
            config=self.config
        )
        
        # Update max steps for all dots
        for dot in self.population.dots:
            dot.max_steps = max_steps
    
    def create_obstacles(self) -> None:
        """
        Create obstacles in the environment.
        """
        # Clear existing obstacles
        self.obstacles = []
        
        # Get obstacle count from config
        obstacle_count = self.config.get('obstacle_count', 5)
        
        # Create fixed obstacles for the environment
        # These create a maze-like environment that the dots must navigate
        
        # Border walls (optional)
        border_thickness = 10
        
        # Add some border walls (not completely enclosing)
        self.obstacles.append(Obstacle(0, 0, self.screen_width, border_thickness))  # Top wall
        self.obstacles.append(Obstacle(0, self.screen_height - border_thickness, 
                                      self.screen_width, border_thickness))  # Bottom wall
        
        # Internal obstacles - create based on count in config
        # Leave space near start and goal positions
        start_x = self.screen_width // 10
        start_y = self.screen_height // 2
        safe_radius = 50  # Safe radius around start and goal
        
        # Calculate free area
        free_area = (self.screen_width - 2 * safe_radius) * (self.screen_height - 2 * safe_radius)
        
        # Calculate obstacle size range based on screen dimensions
        min_size = min(self.screen_width, self.screen_height) // 10
        max_size = min(self.screen_width, self.screen_height) // 5
        
        # Add internal obstacles
        for _ in range(obstacle_count):
            width = np.random.randint(min_size, max_size)
            height = np.random.randint(min_size, max_size)
            
            # Ensure obstacles don't block the start or target completely
            valid_position = False
            max_attempts = 20
            attempts = 0
            
            while not valid_position and attempts < max_attempts:
                # Generate random position
                x = np.random.randint(border_thickness, self.screen_width - width - border_thickness)
                y = np.random.randint(border_thickness, self.screen_height - height - border_thickness)
                
                # Check if too close to start or target
                start_dist = np.sqrt((x - start_x)**2 + (y - start_y)**2)
                target_dist = np.sqrt((x - self.target_pos[0])**2 + (y - self.target_pos[1])**2)
                
                if start_dist > safe_radius and target_dist > safe_radius:
                    valid_position = True
                
                attempts += 1
            
            if valid_position:
                self.obstacles.append(Obstacle(x, y, width, height))
        
        # Optionally, add some fixed challenging obstacles
        # Add a divider in the middle with a gap
        gap_size = 150
        gap_y = self.screen_height // 2 - gap_size // 2
        divider_x = self.screen_width // 2
        divider_width = 20
        
        # Top part of divider
        self.obstacles.append(Obstacle(divider_x, 0, divider_width, gap_y))
        
        # Bottom part of divider  
        self.obstacles.append(Obstacle(divider_x, gap_y + gap_size, 
                                     divider_width, self.screen_height - (gap_y + gap_size)))
    
    def initialize_environment(self) -> None:
        """
        Initialize the complete simulation environment.
        """
        self.initialize_population()
        self.create_obstacles()
        
        # Reset statistics
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.generation_history = []
        
        # Reset generation counter
        self.generation = 1
        
    def update(self) -> None:
        """
        Update the simulation state.
        """
        if self.paused:
            return
            
        # Run multiple updates if speed is increased
        for _ in range(self.speed_multiplier):
            # Check if all dots are dead or have reached the target
            if self.population.update(self.target_pos, self.obstacles):
                # Calculate fitness of dots
                self.population.calculate_fitnesses(self.target_pos)
                
                # Track statistics
                self.track_statistics()
                
                # Create new generation
                self.population.natural_selection()
                self.generation += 1
                
                # Auto-save checkpoint if interval reached
                if self.generation % self.config.get('auto_checkpoint_interval', 10) == 0:
                    self.save_checkpoint()
                
                # Check if maximum generations reached
                if self.generation > self.config.get('max_generations', 100):
                    self.running = False
    
    def track_statistics(self) -> None:
        """
        Track and record statistics for the current generation.
        """
        # Calculate best and average fitness
        fitnesses = [dot.fitness for dot in self.population.dots]
        best_fitness = max(fitnesses) if fitnesses else 0
        avg_fitness = sum(fitnesses) / len(fitnesses) if fitnesses else 0
        
        # Record statistics
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        self.generation_history.append(self.generation)
        
        # Save statistics to file
        self.save_statistics()
        
        # Log progress to console
        print(f"Generation {self.generation}: Best fitness = {best_fitness:.4f}, Average fitness = {avg_fitness:.4f}")
    
    def save_statistics(self) -> None:
        """
        Save current statistics to a file.
        """
        stats = {
            'generation': self.generation,
            'best_fitness': self.best_fitness_history[-1],
            'avg_fitness': self.avg_fitness_history[-1],
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'generation_history': self.generation_history,
            'config': self.config
        }
        
        # Save as JSON
        stats_file = os.path.join(self.stats_dir, f"stats_gen_{self.generation}.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def save_checkpoint(self) -> None:
        """
        Save a checkpoint of the current simulation state.
        """
        # Create checkpoint data
        checkpoint = {
            'generation': self.generation,
            'population_size': len(self.population.dots),
            'config': self.config,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'generation_history': self.generation_history,
            # Save the brain weights of the best dot
            'best_dot_brain': self.population.dots[self.population.best_dot_index].brain.serialize()
        }
        
        # Save as JSON
        checkpoint_file = os.path.join(self.checkpoints_dir, f"checkpoint_gen_{self.generation}.json")
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        print(f"Checkpoint saved at generation {self.generation}")
    
    def load_checkpoint(self, checkpoint_file: str) -> None:
        """
        Load a checkpoint from a file.
        
        Args:
            checkpoint_file: Path to the checkpoint file.
        """
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            # Update generation and config
            self.generation = checkpoint.get('generation', 1)
            self.config.update(checkpoint.get('config', {}))
            
            # Load statistics
            self.best_fitness_history = checkpoint.get('best_fitness_history', [])
            self.avg_fitness_history = checkpoint.get('avg_fitness_history', [])
            self.generation_history = checkpoint.get('generation_history', [])
            
            # Initialize population 
            self.initialize_environment()
            
            # Load the best brain if available
            if 'best_dot_brain' in checkpoint:
                brain_data = checkpoint['best_dot_brain']
                
                # Determine which type of neural network to create
                if self.config.get('use_tensorflow', True):
                    best_brain = TFNeuralNetwork.deserialize(brain_data)
                else:
                    # Try to load as NumPy neural network, or convert
                    try:
                        best_brain = NeuralNetwork.deserialize(brain_data)
                    except KeyError:
                        # Might be in TensorFlow format, attempt conversion
                        print("Converting TensorFlow brain to NumPy format...")
                        tf_brain = TFNeuralNetwork.deserialize(brain_data)
                        # Create a NumPy version (simplified conversion)
                        brain = NeuralNetwork(
                            brain_data['input_size'],
                            brain_data['hidden_size'],
                            brain_data['output_size'],
                            brain_data.get('hidden_layers', 1)
                        )
                        # Properly convert TensorFlow weights to NumPy format
                        weights = tf_brain.model.get_weights()
                        converted_weights = []
                        converted_biases = []
                        
                        # Extract weights and biases from TensorFlow format
                        for i in range(0, len(weights), 2):
                            converted_weights.append(weights[i])
                            converted_biases.append(weights[i+1])
                        
                        # Assign converted weights to NumPy neural network
                        brain.weights = converted_weights
                        brain.biases = converted_biases
                        
                        print("Successfully converted TensorFlow brain to NumPy format")
                
                # Replace the brain of the first dot
                self.population.dots[0].brain = best_brain
                
            print(f"Checkpoint loaded: Generation {self.generation}")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    def draw(self, window: pygame.Surface) -> None:
        """
        Draw the current simulation state to the screen.
        
        Args:
            window: Pygame surface to draw on.
        """
        # Clear screen
        self.screen.fill((0, 0, 0))  # Black background
        
        # Draw target
        target_radius = 15
        pygame.draw.circle(self.screen, GREEN, self.target_pos, target_radius)
        pygame.draw.circle(self.screen, (255, 255, 255), self.target_pos, target_radius - 5)
        
        # Draw obstacles
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)
        
        # Draw population
        # Display all dots or just the best
        if self.show_all_dots:
            self.population.draw(self.screen)
        else:
            # Only draw the best dot
            if self.population.dots:
                self.population.dots[self.population.best_dot_index].draw(self.screen)
        
        # Draw statistics
        self.draw_stats(self.screen)
        
        # Draw graphs if enabled
        if self.show_graph:
            self.draw_graph(self.screen)
        
        # Copy to main window 
        window.blit(self.screen, (0, 0))
        
        # Update display
        pygame.display.flip()
    
    def draw_stats(self, surface: pygame.Surface) -> None:
        """
        Draw statistics on the screen.
        
        Args:
            surface: Surface to draw on.
        """
        font = pygame.font.SysFont('Arial', 16)
        
        # Statistics to display
        stats = [
            f"Generation: {self.generation}",
            f"Population: {len(self.population.dots)}",
            f"Speed: {self.speed_multiplier}x",
            f"Paused: {'Yes' if self.paused else 'No'}",
        ]
        
        # Add best fitness if available
        if self.best_fitness_history:
            stats.append(f"Best Fitness: {self.best_fitness_history[-1]:.4f}")
        
        # Add average fitness if available
        if self.avg_fitness_history:
            stats.append(f"Avg Fitness: {self.avg_fitness_history[-1]:.4f}")
        
        # Render and draw stats
        y_pos = 10
        for stat in stats:
            text = font.render(stat, True, (255, 255, 255))
            surface.blit(text, (10, y_pos))
            y_pos += 25
    
    def draw_graph(self, surface: pygame.Surface) -> None:
        """
        Draw a graph of fitness history.
        
        Args:
            surface: Surface to draw on.
        """
        if not self.best_fitness_history:
            return
            
        # Graph dimensions and position
        graph_width = 200
        graph_height = 100
        graph_x = self.screen_width - graph_width - 10
        graph_y = 10
        
        # Draw graph background
        pygame.draw.rect(surface, (50, 50, 50), (graph_x, graph_y, graph_width, graph_height))
        pygame.draw.rect(surface, (100, 100, 100), (graph_x, graph_y, graph_width, graph_height), 1)
        
        # Draw axis labels
        font = pygame.font.SysFont('Arial', 12)
        title = font.render("Fitness History", True, (255, 255, 255))
        surface.blit(title, (graph_x + 10, graph_y - 15))
        
        # Draw fitness history lines
        if len(self.best_fitness_history) > 1:
            # Calculate scale factors
            x_scale = graph_width / (len(self.best_fitness_history) - 1)
            
            # Find maximum fitness for scaling
            max_fitness = max(max(self.best_fitness_history), max(self.avg_fitness_history))
            if max_fitness == 0:
                max_fitness = 1.0  # Avoid division by zero
                
            y_scale = graph_height / max_fitness
            
            # Draw best fitness line (green)
            for i in range(len(self.best_fitness_history) - 1):
                pygame.draw.line(
                    surface,
                    GREEN,
                    (graph_x + i * x_scale, graph_y + graph_height - self.best_fitness_history[i] * y_scale),
                    (graph_x + (i + 1) * x_scale, graph_y + graph_height - self.best_fitness_history[i + 1] * y_scale),
                    2
                )
            
            # Draw average fitness line (yellow)
            for i in range(len(self.avg_fitness_history) - 1):
                pygame.draw.line(
                    surface,
                    YELLOW,
                    (graph_x + i * x_scale, graph_y + graph_height - self.avg_fitness_history[i] * y_scale),
                    (graph_x + (i + 1) * x_scale, graph_y + graph_height - self.avg_fitness_history[i + 1] * y_scale),
                    2
                )
    
    def handle_events(self) -> None:
        """
        Handle input events from the user.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                # Handle key press events
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                
                elif event.key == pygame.K_SPACE:
                    # Toggle pause
                    self.paused = not self.paused
                
                elif event.key == pygame.K_r:
                    # Reset simulation
                    self.initialize_environment()
                
                elif event.key == pygame.K_s:
                    # Save checkpoint
                    self.save_checkpoint()
                    
                elif event.key == pygame.K_1:
                    # Normal speed
                    self.speed_multiplier = 1
                
                elif event.key == pygame.K_2:
                    # 2x speed
                    self.speed_multiplier = 2
                
                elif event.key == pygame.K_5:
                    # 5x speed
                    self.speed_multiplier = 5
                
                elif event.key == pygame.K_0:
                    # 10x speed
                    self.speed_multiplier = 10
                    
                elif event.key == pygame.K_d:
                    # Toggle show all dots
                    self.show_all_dots = not self.show_all_dots
                    
                elif event.key == pygame.K_g:
                    # Toggle show graph
                    self.show_graph = not self.show_graph
    
    def handle_mouse_events(self) -> None:
        """
        Handle mouse interactions.
        """
        # Get mouse button states
        mouse_buttons = pygame.mouse.get_pressed()
        mouse_pos = pygame.mouse.get_pos()
        
        # Left mouse button: drag target
        if mouse_buttons[0]:  # Left button
            self.target_pos = mouse_pos
        
        # Right mouse button: add obstacle
        if mouse_buttons[2]:  # Right button
            # Create a small obstacle at mouse position
            obstacle_size = 30
            x = mouse_pos[0] - obstacle_size // 2
            y = mouse_pos[1] - obstacle_size // 2
            
            # Add a new obstacle if not too close to existing obstacles
            min_distance = 20
            too_close = False
            for obstacle in self.obstacles:
                ox, oy, width, height = obstacle.rect
                center_x = ox + width // 2
                center_y = oy + height // 2
                dist = ((center_x - mouse_pos[0])**2 + (center_y - mouse_pos[1])**2) ** 0.5
                if dist < min_distance:
                    too_close = True
                    break
                    
            if not too_close:
                self.obstacles.append(Obstacle(x, y, obstacle_size, obstacle_size))
                
    def run(self, headless: bool = False) -> None:
        """
        Run the simulation.
        
        Args:
            headless: If True, run without GUI window (for training and performance benchmarking).
        """
        # Initialize screen if not headless
        if not headless:
            pygame.display.init()
            window = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("AI Dot Collector Simulation")
            
            # Set icon
            icon = pygame.Surface((32, 32))
            icon.fill((0, 0, 0))
            pygame.draw.circle(icon, GREEN, (16, 16), 8)
            pygame.display.set_icon(icon)
        
        # Initialize the environment
        self.initialize_environment()
        
        # Start simulation loop
        self.running = True
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            if not headless:
                # Handle events
                self.handle_events()
                
                # Handle mouse events
                self.handle_mouse_events()
                
                # Update and draw at specified FPS
                self.update()
                self.draw(window)
                
                # Cap frame rate
                self.clock.tick(self.config.get('fps', 60))
                
                # Update FPS counter every second
                frame_count += 1
                if frame_count >= self.config.get('fps', 60):
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    pygame.display.set_caption(f"AI Dot Collector Simulation - FPS: {fps:.1f}")
                    frame_count = 0
                    start_time = time.time()
            else:
                # Headless mode - just update simulation as fast as possible
                self.update()
                
                # Print progress every 10 generations
                if self.generation % 10 == 0 and self.best_fitness_history:
                    print(f"Generation {self.generation}: Best fitness = {self.best_fitness_history[-1]:.4f}")
        
        # Clean up
        if not headless:
            pygame.quit()
            
    def train(self, generations: int = 100) -> None:
        """
        Train the simulation for a specified number of generations.
        
        Args:
            generations: Number of generations to train for.
        """
        # Configure for training
        self.config['max_generations'] = generations
        self.config['auto_checkpoint_interval'] = min(10, generations // 10)
        
        # Run in headless mode
        print(f"Starting training for {generations} generations...")
        self.run(headless=True)
        
        # Save final checkpoint
        self.save_checkpoint()
        print(f"Training completed after {self.generation} generations")
    
    def run_benchmark(self) -> float:
        """
        Run a performance benchmark.
        
        Returns:
            float: Average time (in seconds) per generation.
        """
        # Configure for benchmark
        benchmark_generations = 10
        self.config['max_generations'] = benchmark_generations
        self.config['auto_checkpoint_interval'] = benchmark_generations + 1  # Disable auto-checkpoints
        
        # Identify implementation type
        impl_name = "TensorFlow" if self.config.get('use_tensorflow', True) else "Pure NumPy"
        fast_path = " with NumPy fast path" if self.config.get('use_tensorflow', True) and self.config.get('use_neural_network_fast_path', True) else ""
        
        # Start timing
        start_time = time.time()
        
        # Print benchmark header
        print(f"Running benchmark with {impl_name}{fast_path} implementation for {benchmark_generations} generations...")
        
        # Run in headless mode
        self.run(headless=True)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        avg_time_per_gen = elapsed_time / benchmark_generations
        
        # Get final fitness values
        final_best_fitness = self.best_fitness_history[-1] if self.best_fitness_history else 0
        final_avg_fitness = self.avg_fitness_history[-1] if self.avg_fitness_history else 0
        
        # Print benchmark results
        print(f"\nBenchmark Results ({impl_name}{fast_path}):")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Average time per generation: {avg_time_per_gen:.4f} seconds")
        print(f"Final best fitness: {final_best_fitness:.4f}")
        print(f"Final average fitness: {final_avg_fitness:.4f}")
        print(f"Performance: {1/avg_time_per_gen:.2f} generations per second")
        
        return avg_time_per_gen
    
    def export_best_brain(self, output_file: str = None) -> str:
        """
        Export the best neural network to a file.
        
        Args:
            output_file: File path to save the neural network. If None, a default name will be used.
            
        Returns:
            str: Path to the exported file.
        """
        if not self.population or not self.population.dots:
            print("No population available to export")
            return None
            
        # Get best dot
        best_dot = self.population.dots[self.population.best_dot_index]
        
        # Serialize brain
        brain_data = best_dot.brain.serialize()
        
        # Create export data
        export_data = {
            'generation': self.generation,
            'fitness': best_dot.fitness,
            'timestamp': datetime.datetime.now().isoformat(),
            'config': self.config,
            'neural_network': brain_data
        }
        
        # Generate filename if not provided
        if output_file is None:
            output_file = os.path.join(
                self.current_run_dir, 
                f"best_brain_gen_{self.generation}.json"
            )
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        print(f"Best brain exported to: {output_file}")
        return output_file
    
    def import_brain(self, input_file: str) -> bool:
        """
        Import a neural network from a file.
        
        Args:
            input_file: Path to the file containing the neural network.
            
        Returns:
            bool: True if import was successful.
        """
        try:
            with open(input_file, 'r') as f:
                import_data = json.load(f)
            
            # Extract neural network data
            if 'neural_network' in import_data:
                brain_data = import_data['neural_network']
                
                # Create neural network based on configuration
                if self.config.get('use_tensorflow', True):
                    brain = TFNeuralNetwork.deserialize(brain_data)
                else:
                    # Try to load as NumPy neural network, or convert
                    try:
                        brain = NeuralNetwork.deserialize(brain_data)
                    except KeyError:
                        # Might be in TensorFlow format, attempt conversion
                        print("Converting TensorFlow brain to NumPy format...")
                        tf_brain = TFNeuralNetwork.deserialize(brain_data)
                        # Create a NumPy version (simplified conversion)
                        brain = NeuralNetwork(
                            brain_data['input_size'],
                            brain_data['hidden_size'],
                            brain_data['output_size'],
                            brain_data.get('hidden_layers', 1)
                        )
                        # Properly convert TensorFlow weights to NumPy format
                        weights = tf_brain.model.get_weights()
                        converted_weights = []
                        converted_biases = []
                        
                        # Extract weights and biases from TensorFlow format
                        for i in range(0, len(weights), 2):
                            converted_weights.append(weights[i])
                            converted_biases.append(weights[i+1])
                        
                        # Assign converted weights to NumPy neural network
                        brain.weights = converted_weights
                        brain.biases = converted_biases
                        
                        print("Successfully converted TensorFlow brain to NumPy format")
                
                # Initialize population if not already done
                if self.population is None:
                    self.initialize_environment()
                
                # Replace the brain of the first dot
                self.population.dots[0].brain = brain
                
                print(f"Brain imported from: {input_file}")
                print(f"Original fitness: {import_data.get('fitness', 'N/A')}")
                print(f"From generation: {import_data.get('generation', 'N/A')}")
                
                return True
                
            else:
                print("Invalid import file: Neural network data not found")
                return False
                
        except Exception as e:
            print(f"Error importing brain: {e}")
            return False


def main():
    """
    Main entry point for the simulation.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AI Dot Collector Simulation')
    parser.add_argument('--no-tensorflow', action='store_true', help='Use pure NumPy neural networks instead of TensorFlow')
    parser.add_argument('--no-fast-path', action='store_true', help='Disable NumPy fast path optimization for TensorFlow')
    parser.add_argument('--no-gui', action='store_true', help='Run without GUI (headless mode)')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--train', type=int, metavar='N', help='Train for N generations in headless mode')
    parser.add_argument('--load', type=str, metavar='FILE', help='Load checkpoint from FILE')
    args = parser.parse_args()
    
    # Create configuration based on arguments
    config = {
        'use_tensorflow': not args.no_tensorflow,
        'use_neural_network_fast_path': not args.no_fast_path,
        'headless': args.no_gui,
    }
    
    # Create simulation with configuration
    simulation = Simulation(config)
    
    # Load checkpoint if specified
    if args.load:
        simulation.load_checkpoint(args.load)
    
    # Run benchmark if requested
    if args.benchmark:
        simulation.run_benchmark()
        return
        
    # Train if requested
    if args.train:
        simulation.train(args.train)
        return
    
    # Run simulation normally
    simulation.run(headless=config.get('headless', False))


if __name__ == "__main__":
    main() 