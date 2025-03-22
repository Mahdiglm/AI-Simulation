"""
Dot class for the AI Dot Collector simulation.
"""
import math
import pygame
import numpy as np
from typing import List, Tuple, Optional, Union, Any

from tf_neural_network import TFNeuralNetwork
from neural_network import NeuralNetwork
from obstacle import Obstacle
from constants import GREEN, RED

# Type alias for either neural network implementation
BrainType = Union[TFNeuralNetwork, NeuralNetwork]

class Dot:
    """A dot in the simulation that learns with a neural network."""
    
    def __init__(self, x: float, y: float, brain: Optional[BrainType] = None, 
                max_steps: int = 400) -> None:
        """Initialize a dot with position and neural network."""
        self.x = x
        self.y = y
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.acceleration_x = 0.0
        self.acceleration_y = 0.0
        self.max_velocity = 7.0  # Increased from 5.0 for better movement range
        self.min_velocity = 0.5  # Minimum velocity
        self.max_accel = 0.5
        self.friction = 0.98  # Slight friction for more realistic movement
        
        # Brain is a neural network with 15 inputs:
        # - 8 sensor inputs for 8 directions
        # - 2 for normalized target position (x, y)
        # - 2 for current velocity (x, y)
        # - 2 for distance to nearest obstacle and target
        # - 1 for current speed value
        # 
        # 3 outputs:
        # - direction radians (0-2π) 
        # - acceleration multiplier (-1 to 1)
        # - speed control multiplier (0 to 1)
        # Default to TensorFlow neural network if none provided
        self.brain = brain or TFNeuralNetwork(15, 16, 3, hidden_layers=2)
        
        # Simulation state
        self.dead = False
        self.reached_goal = False
        self.fitness = 0.0
        self.max_steps = max_steps
        self.steps = 0
        self.goal_reach_time = None  # Time (in steps) to reach the goal
        
        # Tracking variables for improved fitness calculation
        self.prev_distance_to_goal = float('inf')
        self.start_distance_to_goal = float('inf')
        self.wall_hits = 0
        
        # Track position history for stagnation detection
        self.position_history = []  # Will store (x, y) tuples
        self.stagnation_threshold = 10  # N frames without significant movement
        self.stagnation_distance_threshold = 20  # Minimum distance to consider as not stagnant
        self.stagnation_count = 0
        
        # Sensor readings for visualization
        self.sensor_distances = [0.0] * 8
        
    def reset_state(self, x: float = None, y: float = None) -> None:
        """Reset the dot's state but keep its brain intact."""
        # Update position if provided, otherwise keep current position
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
            
        # Reset movement
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.acceleration_x = 0.0
        self.acceleration_y = 0.0
        
        # Reset state
        self.dead = False
        self.reached_goal = False
        self.steps = 0
        self.goal_reach_time = None
        
        # Reset fitness tracking
        self.fitness = 0.0
        self.prev_distance_to_goal = float('inf')
        self.start_distance_to_goal = float('inf')
        self.wall_hits = 0
        
        # Reset stagnation detection
        self.position_history = []
        self.stagnation_count = 0
        
        # Reset sensor readings
        self.sensor_distances = [0.0] * 8

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the dot on the given surface."""
        if self.dead:
            # Draw dead dot in gray
            pygame.draw.circle(surface, (100, 100, 100), (int(float(self.x)), int(float(self.y))), 6)
        elif self.reached_goal:
            # Draw successful dot in green
            pygame.draw.circle(surface, GREEN, (int(float(self.x)), int(float(self.y))), 7)
        else:
            # Draw regular dot with a color based on its fitness
            fitness_color = min(255, int(self.fitness * 255))
            color = (255 - fitness_color, fitness_color, 200)
            
            # Draw the dot with size proportional to speed
            size = 5 + int(self.get_speed() / self.max_velocity * 3)
            pygame.draw.circle(surface, color, (int(float(self.x)), int(float(self.y))), size)
            
            # Draw a line showing direction of movement
            if self.get_speed() > 0.1:
                direction_x = float(self.x) + (float(self.velocity_x) / self.get_speed() * 10)
                direction_y = float(self.y) + (float(self.velocity_y) / self.get_speed() * 10)
                pygame.draw.line(surface, (200, 200, 255), 
                             (int(float(self.x)), int(float(self.y))), 
                             (int(direction_x), int(direction_y)), 2)
                
            # Optionally draw sensors (can be toggled)
            self.draw_sensors(surface)

    def draw_sensors(self, surface: pygame.Surface) -> None:
        """Draw sensor lines for visualization."""
        # Draw each sensor as a line in direction of sensor
        sensor_colors = [
            (100, 100, 255),  # Right
            (130, 130, 255),  # Upper right
            (160, 160, 255),  # Up
            (180, 180, 255),  # Upper left
            (200, 200, 255),  # Left
            (220, 220, 255),  # Lower left
            (240, 240, 255),  # Down
            (255, 255, 255),  # Lower right
        ]
        
        for i, distance in enumerate(self.sensor_distances):
            if distance > 0:  # Only draw if sensor detected something
                angle = i * (2 * math.pi / 8)  # 8 directions
                end_x = float(self.x) + math.cos(angle) * float(distance) * 0.5  # Scale down for visual
                end_y = float(self.y) + math.sin(angle) * float(distance) * 0.5
                pygame.draw.line(surface, sensor_colors[i], 
                             (int(float(self.x)), int(float(self.y))), 
                             (int(end_x), int(end_y)), 1)

    def get_speed(self) -> float:
        """Calculate the current speed."""
        return math.sqrt(float(self.velocity_x)**2 + float(self.velocity_y)**2)
            
    def think(self, target_pos: Tuple[float, float], obstacles: List[Obstacle]) -> None:
        """Use the neural network to think about the next move."""
        if self.dead or self.reached_goal:
            return
        
        # Get sensor data for 8 directions
        sensor_data = self.get_sensor_data(obstacles)
        self.sensor_distances = sensor_data.copy()  # Store for visualization
        
        # Calculate directions and distances
        target_x, target_y = target_pos
        dx = target_x - self.x
        dy = target_y - self.y
        distance_to_target = math.sqrt(dx**2 + dy**2)
        
        # Store initial distance for fitness calculation if not already set
        if self.start_distance_to_goal == float('inf'):
            self.start_distance_to_goal = distance_to_target
        
        # Calculate normalized direction to target
        norm_dx = dx / (distance_to_target if distance_to_target > 0 else 1)
        norm_dy = dy / (distance_to_target if distance_to_target > 0 else 1)
        
        # Find closest obstacle
        closest_obstacle_distance = min(sensor_data) if sensor_data else 1000
        
        # Normalize velocities
        norm_vx = self.velocity_x / self.max_velocity
        norm_vy = self.velocity_y / self.max_velocity
        
        # Normalize distances
        norm_target_dist = min(1.0, distance_to_target / 500)  # Normalized to 0-1
        norm_obstacle_dist = min(1.0, closest_obstacle_distance / 500)
        
        # Current speed normalized
        norm_speed = self.get_speed() / self.max_velocity
        
        # Prepare inputs for neural network
        inputs = sensor_data + [
            norm_dx,
            norm_dy,
            norm_vx,
            norm_vy,
            norm_obstacle_dist,
            norm_target_dist,
            norm_speed
        ]
        
        # Get output from the neural network
        outputs = self.brain.feedforward(inputs)
        
        # Interpret outputs (extract scalars from numpy array)
        # Output 0: Direction (0 to 2π)
        direction = float(outputs[0]) * math.pi * 2  # 0 to 2π
        
        # Output 1: Acceleration magnitude (-1 to 1)
        accel_amount = float(outputs[1]) * 2 - 1  # Map from 0-1 to -1 to 1
        
        # Output 2: Speed control (0 to 1)
        speed_control = float(outputs[2])  # 0 to 1
        
        # Apply acceleration based on direction and acceleration amount
        self.acceleration_x = math.cos(direction) * self.max_accel * accel_amount
        self.acceleration_y = math.sin(direction) * self.max_accel * accel_amount
        
        # Update velocity
        self.velocity_x += self.acceleration_x
        self.velocity_y += self.acceleration_y
        
        # Apply velocity constraints based on speed control
        target_max_velocity = self.min_velocity + speed_control * (self.max_velocity - self.min_velocity)
        current_speed = self.get_speed()
        
        if current_speed > target_max_velocity:
            # Scale down velocity to meet target max
            scale_factor = target_max_velocity / current_speed
            self.velocity_x *= scale_factor
            self.velocity_y *= scale_factor
        
        # Apply friction
        self.velocity_x *= self.friction
        self.velocity_y *= self.friction
        
        # Update the previous distance to goal
        self.prev_distance_to_goal = distance_to_target
        
    def move(self) -> None:
        """Move the dot based on velocity."""
        if self.dead or self.reached_goal:
            return
            
        # Update position
        new_x = float(self.x) + float(self.velocity_x)
        new_y = float(self.y) + float(self.velocity_y)
        
        # Track position for stagnation detection
        # Only store positions if we need them for stagnation detection
        # Keep the history size strictly limited to exactly what we need
        pos = (float(self.x), float(self.y))
        
        # Use a fixed-size deque instead of an unbounded list
        # Automatically removes oldest items as new ones are added
        if len(self.position_history) >= self.stagnation_threshold:
            self.position_history.pop(0)  # Remove oldest position
        
        self.position_history.append(pos)
            
        # Check for stagnation
        self.check_stagnation()
        
        # Update position
        self.x = new_x
        self.y = new_y
        
        # Increment steps
        self.steps += 1
        
        # Check if we've exceeded the maximum number of steps
        if self.steps >= self.max_steps:
            self.dead = True
            
    def check_stagnation(self) -> None:
        """Check if the dot is stuck or not making progress."""
        if len(self.position_history) >= self.stagnation_threshold:
            # Calculate the total distance moved in the last N frames
            total_distance = 0
            for i in range(1, len(self.position_history)):
                x1, y1 = self.position_history[i-1]
                x2, y2 = self.position_history[i]
                distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                total_distance += distance
                
            # If the total distance is below threshold, increment stagnation counter
            if total_distance < self.stagnation_distance_threshold:
                self.stagnation_count += 1
            else:
                # Reset stagnation counter if movement occurred
                self.stagnation_count = max(0, self.stagnation_count - 1)
            
    def get_sensor_data(self, obstacles: List[Obstacle]) -> List[float]:
        """
        Get distance readings from sensors in 8 directions.
        Returns a list of sensor distances (0 to 1, 1 = nothing detected).
        """
        # Define 8 directions (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
        max_sensor_distance = 250.0
        sensor_distances = [1.0] * 8  # Initialize all sensors to max distance (normalized)
        
        if not obstacles:  # Early return if no obstacles
            return sensor_distances
            
        # Pre-calculate direction vectors for all 8 directions
        directions = []
        for i in range(8):
            angle = i * (2 * math.pi / 8)
            # Calculate direction vector
            directions.append((math.cos(angle), math.sin(angle)))
        
        # Pre-extract dot position once
        x1, y1 = self.x, self.y
        
        # Process each obstacle only once, checking all sensors against it
        for obstacle in obstacles:
            # Convert obstacle rect to sides only once per obstacle
            left, top, width, height = obstacle.rect
            right = left + width
            bottom = top + height
            
            # Define the four sides of the rectangle - compute only once per obstacle
            sides = [
                ((left, top), (right, top)),     # Top
                ((right, top), (right, bottom)), # Right
                ((left, bottom), (right, bottom)), # Bottom
                ((left, top), (left, bottom))    # Left
            ]
            
            # Check each direction
            for i, (dx, dy) in enumerate(directions):
                # Only process if we haven't found a closer intersection
                if sensor_distances[i] == 1.0:
                    # Calculate potential endpoint once
                    x2 = x1 + dx * max_sensor_distance
                    y2 = y1 + dy * max_sensor_distance
                    
                    min_dist = max_sensor_distance
                    
                    # Check intersection with each side of the obstacle
                    for (sx1, sy1), (sx2, sy2) in sides:
                        # Calculate denominator only once per side
                        denom = (x1 - x2) * (sy1 - sy2) - (y1 - y2) * (sx1 - sx2)
                        if abs(denom) < 1e-10:  # Avoid division by near-zero
                            continue
                            
                        t = ((x1 - sx1) * (sy1 - sy2) - (y1 - sy1) * (sx1 - sx2)) / denom
                        
                        # Early skip if t is out of range
                        if t < 0 or t > 1:
                            continue
                            
                        u = -((x1 - x2) * (y1 - sy1) - (y1 - y2) * (x1 - sx1)) / denom
                        
                        if 0 <= u <= 1:  # Intersection found
                            # Calculate intersection point
                            ix = x1 + t * (x2 - x1)
                            iy = y1 + t * (y2 - y1)
                            
                            # Calculate distance
                            dist = math.sqrt((ix - x1)**2 + (iy - y1)**2)
                            if dist < min_dist:
                                min_dist = dist
                                # Update sensor distance (normalized)
                                sensor_distances[i] = min_dist / max_sensor_distance
        
        return sensor_distances
        
    def update(self, target_pos: Tuple[float, float], obstacles: List[Obstacle]) -> bool:
        """Update the dot's state."""
        if self.dead or self.reached_goal:
            return True
        
        self.think(target_pos, obstacles)
        self.move()
        
        # Get current position for calculations
        x, y = float(self.x), float(self.y)
        
        # Check if dot has reached the target
        target_x, target_y = target_pos
        distance_squared = (x - target_x)**2 + (y - target_y)**2
        if distance_squared < 15**2:  # Square of target radius (15) - avoids sqrt for speed
            self.reached_goal = True
            self.goal_reach_time = self.steps  # Record how many steps it took
            return True
        
        # Check for collision with obstacles using rectangle collision
        # This is more efficient than checking each side
        for obstacle in obstacles:
            rect_x, rect_y, width, height = obstacle.rect
            
            # Check if dot is within the rectangle bounds
            if (rect_x <= x <= rect_x + width and
                rect_y <= y <= rect_y + height):
                self.dead = True
                self.wall_hits += 1
                return True
        
        # Check for hitting screen boundaries
        screen_width, screen_height = 800, 600  # Match simulation dimensions
        if x < 0 or x > screen_width or y < 0 or y > screen_height:
            self.dead = True
            self.wall_hits += 1
            return True
        
        return False

    def calculate_fitness(self, target_pos: Tuple[float, float]) -> float:
        """Calculate the fitness of this dot."""
        target_x, target_y = target_pos
        distance = math.sqrt((float(self.x) - target_x)**2 + (float(self.y) - target_y)**2)
        
        # Base fitness calculation: inverse of distance to target
        if self.reached_goal:
            # If reached goal, major reward plus bonus for doing it quickly
            time_bonus = 1.0 + (self.max_steps - self.goal_reach_time) / self.max_steps * 5.0
            fitness = 1.0 + time_bonus
        else:
            # Distance-based fitness for dots that didn't reach the goal
            distance_factor = 1.0 - (distance / self.start_distance_to_goal)
            distance_factor = max(0.1, distance_factor)  # Ensure minimum fitness
            
            # Progress factor - how much closer did it get compared to start
            progress_factor = (self.start_distance_to_goal - distance) / self.start_distance_to_goal
            progress_factor = max(0.0, progress_factor)  # Can't be negative
            
            # Steps factor - reward using fewer steps
            steps_factor = 1.0 - (self.steps / self.max_steps)
            
            # Base fitness
            fitness = (distance_factor * 0.4) + (progress_factor * 0.5) + (steps_factor * 0.1)
        
        # Penalties
        # Wall hit penalty
        wall_penalty = 0.05 * self.wall_hits
        
        # Stagnation penalty
        stagnation_penalty = 0.03 * self.stagnation_count
        
        # Apply penalties
        fitness = max(0.01, fitness - wall_penalty - stagnation_penalty)
        
        # Store and return
        self.fitness = fitness
        return fitness 