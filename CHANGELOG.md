# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- TBD

## [0.6.1] - 2024-06-01

### Fixed

- Implemented proper weight format conversion between TensorFlow and NumPy neural networks
- Added robust error handling throughout the codebase
- Fixed potential division by zero errors in sensor calculations
- Improved handling of invalid position values in dot rendering
- Added bounds checking for dot size calculations
- Enhanced numerical stability in neural network computations
- Added comprehensive error handling in NumPy fast path implementation
- Implemented proper shape validation in matrix operations
- Added safe fallbacks for all error conditions to prevent crashes
- Fixed NaN and Infinity handling in neural network operations

## [0.6.0] - 2024-06-01

### Added

- Checkpoint system for saving simulation state at specific generations
- Automatic checkpoints at configurable intervals
- Manual checkpoint creation with 'C' key
- Checkpoint naming with generation number and timestamp
- Tanh activation function as an alternative activation option
- More detailed numerical stability warnings
- Git integration for version control
- Improved README documentation with accurate usage instructions
- MIT License

### Changed

- Optimized sensor calculations for up to 30% better performance
- Improved collision detection with early-exit optimizations
- Enhanced error handling for file operations
- Better numerical stability in neural network calculations
- Improved input normalization for neural network
- More robust error handling when saving/loading models
- File overwrite confirmation when saving models

### Fixed

- Numerical stability issues in neural network activation functions
- Potential division by zero in sensor calculations
- Memory leaks in position history tracking
- Improved collision detection performance
- File handling error messages with more specific information

## [0.5.0] - 2024-05-20

### Added

- Full 360° directional movement control for AI dots
- Speed control output from neural network (dots can adjust their own speed)
- Enhanced stagnation detection system with position history tracking
- Improved fitness calculation with time efficiency rewards
- Penalties for wall collisions and stagnation behavior
- Expanded neural network outputs (direction, acceleration, speed control)
- Better visualization of dot states and behaviors
- Running averages for smoother fitness history graphs
- Enhanced statistics display with more detailed performance metrics
- Support for visualizing average population fitness alongside best fitness

### Changed

- Updated neural network architecture to support 3 outputs instead of 2
- Improved numerical stability when handling numpy arrays
- Enhanced the physics model with better friction and acceleration controls
- More efficient crossover implementation in genetic algorithm
- Better visualization of sensors and movement direction
- Upgraded dot rendering with dynamic size based on speed
- Improved collision detection with more accurate boundary handling
- Added color coding for different dot states (dead, reached goal, moving)
- Enhanced graph rendering with generation numbers and data ranges

### Fixed

- Fixed numpy array handling to avoid deprecated scalar conversions
- Improved numerical stability in neural network calculations
- Resolved coordinate conversion issues in graphics rendering
- Fixed position history tracking for stagnation detection
- Corrected fitness calculation for rewards and penalties

## [0.4.0] - 2024-05-19

### Added

- Enhanced neural network architecture with multiple hidden layers
- Advanced sensor system with 8 directional distance sensors
- Continuous physics simulation with acceleration-based movement
- Single-point crossover genetic algorithm for improved evolution
- Tournament selection for more efficient parent selection
- Statistics graph showing fitness improvements over generations
- Side-by-side visualization of simulation and statistics
- Adaptable network complexity based on memory usage setting
- Configurable simulation speed with exponential scaling
- Performance optimizations for handling larger populations
- Elitism strategy to preserve best performer between generations
- Improved collision detection and boundary handling
- Sensor visualization for the best performing dot
- Stagnation detection to eliminate non-improving dots

### Changed

- Complete rewrite of the AI neural network implementation
- Improved neural network initialization with He scaling
- Replaced discrete movement with continuous physics simulation
- Updated fitness calculation for more effective evolution
- Better obstacle generation with more interesting patterns
- Enhanced UI with more detailed statistics display
- Reduced dependency footprint to essential libraries
- Updated documentation with detailed technical specifications
- Improved integration with main application interface

### Fixed

- Performance issues with large populations
- Neural network output conversion handling
- Coordinate calculation errors
- Memory usage issues when running for many generations
- Simulation freezing during rapid evolution cycles
- Compatibility issues with different Python environments

## [0.3.0] - 2024-05-18

### Added

- Pygame-based AI Dot Collector simulation in a separate window
- Fully functional neural network implementation for AI dots
- Real-time visualization of evolving AI behavior
- Genetic algorithm with selection and mutation
- Custom obstacle generation and collision detection
- Real-time statistics display in simulation window
- Ability to control simulation speed (1x, 2x, 3x)
- Support for saving and loading trained neural networks
- Integration between terminal UI and Pygame simulation

### Changed

- Moved from terminal-based simulation visualization to dedicated Pygame window
- Enhanced configuration options for neural network parameters
- Updated README with technical details about neural network architecture
- Improved simulation results reporting with actual performance metrics

### Fixed

- Simulation speed and performance issues
- Configuration parameter validation and error handling

## [0.2.0] - 2024-05-17

### Added

- AI Dot Collector simulation with configurable parameters
- Matrix rain transitions and particle effects
- Enhanced startup sequence with animations
- Visual feedback for menu navigation
- Pulsing box and color effects for UI elements
- Simulation arena with dots, goals, and obstacles
- Configuration options for AI behavior
- Support for saving and loading AI models
- Multi-generation evolution simulation with fitness tracking

### Changed

- Improved menu navigation with arrow keys and visual feedback
- Enhanced configuration interface with categorized settings
- Updated simulation progress display with visual indicators
- Modernized UI with particle effects and animated transitions

### Fixed

- Various menu navigation issues
- Configuration display alignment
- Animation timing and performance

## [0.1.0] - 2024-05-10

### Added

- Initial project setup
- Basic UI menu system for simulation configuration
- Simulation environment foundation

### Changed

- N/A

### Fixed

- N/A
