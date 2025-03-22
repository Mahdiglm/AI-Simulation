# AI Simulation Project

## Overview

This project provides a simulation environment for AI models with a focus on evolutionary neural networks. It allows users to configure various simulation parameters and observe how different AI models behave and evolve under various conditions.

## Features

- Neural network-based decision making with TensorFlow and NumPy implementations
- Evolutionary algorithms for training AI agents
- Customizable simulation parameters through command-line arguments
- Advanced visualization of AI behavior and learning progress
- Performance optimizations with NumPy fast path for TensorFlow
- Comprehensive data collection and statistics tracking
- Automatic and manual checkpoint system for saving progress
- Ability to run in headless mode for faster training
- Built-in benchmarking capabilities

### AI Dot Collector

The AI Dot Collector simulation demonstrates evolutionary AI where dots learn to navigate to a target while avoiding obstacles. Features include:

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

## Installation

```
git clone https://github.com/Mahdiglm/AI-Simulation.git
cd AI-Simulation
pip install -r requirements.txt
```

## Usage

The simulation can be run directly from the command line with various options:

```
python simulation.py [options]
```

### Command-line Options:

```
--no-tensorflow    Use pure NumPy neural networks instead of TensorFlow
--no-fast-path     Disable NumPy fast path optimization for TensorFlow
--no-gui           Run without GUI (headless mode)
--benchmark        Run performance benchmark
--train N          Train for N generations in headless mode
--load FILE        Load checkpoint from FILE
```

### Examples:

Run the simulation with the default settings:

```
python simulation.py
```

Run in headless mode for faster training:

```
python simulation.py --no-gui
```

Train for 100 generations in headless mode:

```
python simulation.py --train 100
```

Load a previous checkpoint:

```
python simulation.py --load checkpoints/checkpoint_gen_50.json
```

Run a performance benchmark:

```
python simulation.py --benchmark
```

### Interactive Controls:

While the simulation is running, you can use the following keyboard controls:

- Space: Pause/Resume simulation
- R: Reset simulation
- S: Save checkpoint
- 1: Normal speed
- 2: 2x speed
- 5: 5x speed
- 0: 10x speed
- D: Toggle showing all dots or just the best dot
- G: Toggle graph display

## Project Structure

- `simulation.py`: Main simulation manager and entry point
- `dot.py`: Implements the AI dot agents with sensors and physics
- `population.py`: Manages populations of dots and evolution
- `neural_network.py`: Pure NumPy implementation of neural networks
- `tf_neural_network.py`: TensorFlow implementation with optimizations
- `obstacle.py`: Handles obstacles and collision detection
- `constants.py`: Defines various constants used throughout the project

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

### Genetic Algorithm

- Selection: Tournament selection (selects the best from random subsets)
- Crossover: Single-point crossover between parent networks
- Mutation: Random weight adjustments controlled by mutation rate
- Fitness: Enhanced calculation considering distance, progress, efficiency
- Elitism: Best performers are preserved unchanged in each generation

### Optimization Features

- Optional TensorFlow and NumPy implementations
- NumPy fast path for efficient feedforward operations
- Optimized sensor calculations for improved performance
- Efficient collision detection with early-exit optimizations
- Configurable execution speed for visualization vs. training

## Data Management

Simulation runs are organized in the following directory structure:

- `runs/`: Main directory for all simulation runs
  - `run_[timestamp]/`: Directory for a specific run
    - `checkpoints/`: Saved model checkpoints
    - `stats/`: Statistics data in JSON format

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
