# AI Simulation Project

## Overview

This project provides a simulation environment for AI models. It allows users to configure various simulation parameters and observe how different AI models behave under various conditions.

## Features

- Interactive menu-driven interface with enhanced visual effects
- Customizable simulation parameters
- Visualization of simulation results
- Particle effects and animations for a modern UI experience
- AI Dot Collector simulation with visual representation
- Support for different AI models (more coming soon)

### AI Dot Collector

The AI Dot Collector simulation demonstrates a simple evolutionary AI model where dots learn to navigate to a target while avoiding obstacles. Features include:

- Population-based evolution across multiple generations
- Neural network-based decision making for each dot
- Full 360° movement control with speed adjustment
- Enhanced fitness calculation with rewards for efficiency
- Stagnation detection and wall collision penalties
- Comprehensive visualization with fitness history graph
- Configurable parameters (population size, mutation rate, etc.)
- Visual representation of AI learning progress
- Ability to save and load AI brain models
- Real-time statistics display with average and best fitness tracking
- Automatic and manual checkpoints to save progress at key generations
- Optimized sensor calculations for improved performance
- Enhanced numerical stability in neural network computations
- Robust error handling and file management

## Simulations

### AI Dot Collector

The AI Dot Collector simulation demonstrates a simple evolutionary AI model where dots learn to navigate to a target while avoiding obstacles. Features include:

- Population-based evolution across multiple generations
- Neural network-based decision making for each dot
- Full 360° movement control with speed adjustment
- Enhanced fitness calculation with rewards for efficiency
- Stagnation detection and wall collision penalties
- Comprehensive visualization with fitness history graph
- Configurable parameters (population size, mutation rate, etc.)
- Visual representation of AI learning progress
- Ability to save and load AI brain models
- Real-time statistics display with average and best fitness tracking

The simulation uses a genetic algorithm approach with neural networks controlling each dot's movement. The dots evolve over generations, with the most successful ones (those that reach the target quickly or get closest to it) passing their genes to the next generation.

## Installation

```
git clone https://github.com/yourusername/ai-simulation.git
cd ai-simulation
pip install -r requirements.txt
```

## Usage

Run the main script to start the simulation:

```
python main.py
```

Follow the on-screen prompts to configure and run your simulation.

For the AI Dot Collector simulation:

1. Select "Start Simulation" from the main menu
2. Choose "AI Dot Collector"
3. Configure the simulation parameters if desired
4. Select "Start Simulation"
5. The simulation will open in a separate window using Pygame
6. Use the keyboard controls shown in the simulation window:
   - Space: Pause/Resume
   - R: Reset simulation
   - +/-: Change speed
   - M: Generate new map (keep AI)
   - S: Save best brain
   - L: Load saved brain
   - C: Create checkpoint of current simulation state
   - A: Toggle all dots/best dot
   - V: Toggle sensor visualization
   - G: Toggle fitness graph
   - Esc: Return to menu
7. Close the window when finished to return to the main menu

Checkpoints are automatically saved every 10 generations (configurable) and stored in the `checkpoints` directory. Manual checkpoints can be created at any time by pressing the 'C' key.

## Requirements

- Python 3.8 or higher
- Required packages:
  - colorama==0.4.6 (for terminal colors)
  - pygame==2.5.0 (for AI Dot Collector simulation)
  - numpy==1.24.3 (for neural network operations)

## Configuration

The simulation can be configured through the interactive menu. Options include:

### General Settings

- Simulation duration
- Environment complexity
- Random seed
- Log level
- Save results toggle

### AI Dot Collector Settings

- Population size: Number of AI dots in each generation
- Dot count: Number of dots displayed on screen (for visual clarity)
- Obstacle count: Number of obstacles in the environment
- Memory usage (low, medium, high)
- Mutation rate: Controls how much neural networks change between generations
- Number of generations: Maximum generations to run
- Options to save/load the best performing "brain" (neural network)

## Technical Details

### Neural Network Architecture

The AI dots use an enhanced neural network with:

- Input layer: 15 neurons
  - 8 distance sensors in different directions (N, NE, E, SE, S, SW, W, NW)
  - 2 normalized target position values (dx, dy)
  - 2 current velocity values (vel_x, vel_y)
  - 2 distance metrics (nearest obstacle distance, target distance)
  - 1 current speed value (normalized)
- Hidden layers: Multiple configurable layers (2 by default)
  - Each with 16 neurons and leaky ReLU activation
- Output layer: 3 neurons
  - Direction (0 to 2π)
  - Acceleration (-1 to 1)
  - Speed control (0 to 1)

The neural network uses advanced techniques like:

- He initialization for better learning with ReLU activation
- Leaky ReLU to prevent "dying neurons"
- Multiple hidden layers for more complex behavior modeling
- Numerical stabilization with clipping for overflow prevention

### Genetic Algorithm

- Selection: Tournament selection (selects the best from random subsets)
- Crossover: Single-point crossover between parent networks
  - Randomly crosses over weights and biases between parents
  - Separate crossover points for each layer in the network
- Mutation: Random weight adjustments controlled by mutation rate
  - Uses normal distribution for more natural changes
- Fitness: Enhanced calculation considering:
  - Distance to target
  - Progress toward the goal
  - Efficiency (steps taken)
  - Penalties for wall hits and stagnation
- Elitism: Best performers are preserved unchanged in each generation

### Physics Simulation

- Continuous physics model with acceleration, velocity, friction, and position
- Full 360° movement control with speed adjustment
- Collision detection with walls and obstacles
- Directional sensors that detect distances to walls and obstacles
- Position history tracking for stagnation detection
- Smooth movement with velocity-based updates

### Visualization

- Side-by-side display of simulation and statistics
- Fitness history graph showing best and average fitness
- Color coding for different dot states (dead, reached goal, moving)
- Sensor visualization to see what the AI perceives
- Direction indicators showing movement intent
- Comprehensive statistics display with generation metrics

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
