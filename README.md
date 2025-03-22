# AI Simulation Project 🚀  

## Overview  

The **AI Simulation Project** is a cutting-edge environment designed to simulate and evolve AI models using **evolutionary neural networks**. With highly customizable parameters and advanced visualization, this project allows users to analyze AI decision-making, observe adaptive behaviors, and optimize neural architectures efficiently.  

## Key Features  

- 🧠 **Neural Network Decision Making** – Supports both **TensorFlow** and **NumPy** implementations.  
- 🔄 **Evolutionary Algorithms** – Adaptive learning through natural selection and genetic evolution.  
- ⚡ **Performance Optimization** – NumPy fast path optimizations for enhanced speed.  
- 🎨 **Advanced Visualization** – Real-time tracking of AI learning and fitness progression.  
- 🛠️ **Customizable Parameters** – Adjust settings via command-line arguments.  
- 🏆 **Benchmarking & Statistics** – Collect performance metrics and optimize models.  
- 💾 **Checkpoint System** – Save and load progress automatically or manually.  
- ⚡ **Headless Mode Support** – Run high-speed training without GUI overhead.  

---

## 🏹 AI Dot Collector  

The **AI Dot Collector** is a core simulation showcasing evolutionary AI in action. In this scenario, dots **learn to navigate** toward a target while avoiding obstacles, improving their strategy across multiple generations.  

### Features:  
✅ **Neural Network-Based Movement** – Each dot makes decisions using a trained neural model.  
✅ **Full 360° Navigation** – Control speed, direction, and acceleration dynamically.  
✅ **Enhanced Fitness Calculation** – Efficient navigation is rewarded; stagnation and collisions are penalized.  
✅ **Evolutionary Learning** – Populations evolve over multiple generations.  
✅ **Real-time Stats & Graphs** – Track AI learning curves, fitness history, and progress.  
✅ **Save & Load AI Brains** – Preserve trained models for future analysis.  
✅ **Optimized Sensor Calculations** – Fast, precise environmental detection for each AI agent.  
✅ **Comprehensive Error Handling** – Ensures smooth execution with minimal failures.  

---

## 🚀 Installation  

Clone the repository and install dependencies:  

```sh
git clone https://github.com/Mahdiglm/AI-Simulation.git
cd AI-Simulation
pip install -r requirements.txt
```

---

## 🎮 Usage  

Run the simulation with customizable options:  

```sh
python simulation.py [options]
```

### Command-line Options  

```sh
--no-tensorflow    # Use pure NumPy neural networks  
--no-fast-path     # Disable NumPy fast path optimization  
--no-gui           # Run in headless mode (no GUI)  
--benchmark        # Run performance benchmark  
--train N          # Train for N generations  
--load FILE        # Load checkpoint from a saved file  
```

### Examples  

Run the simulation with default settings:  

```sh
python simulation.py
```

Train AI for **100 generations** in headless mode:  

```sh
python simulation.py --train 100 --no-gui
```

Load a saved checkpoint:  

```sh
python simulation.py --load checkpoints/checkpoint_gen_50.json
```

Run a **performance benchmark**:  

```sh
python simulation.py --benchmark
```

---

## ⌨️ Interactive Controls  

While running, control the simulation using:  

| Key | Action |
|-----|--------|
| **Space** | Pause/Resume simulation |
| **R** | Reset simulation |
| **S** | Save checkpoint |
| **1** | Normal speed |
| **2** | 2x speed |
| **5** | 5x speed |
| **0** | 10x speed |
| **D** | Toggle between best dot & all dots |
| **G** | Show/hide fitness graph |

---

## 🏗️ Project Structure  

```
📂 AI-Simulation/
 ├── simulation.py        # Main simulation manager
 ├── dot.py               # AI-controlled dots with sensors & movement
 ├── population.py        # Evolutionary population management
 ├── neural_network.py    # NumPy-based neural network
 ├── tf_neural_network.py # TensorFlow-based neural network
 ├── obstacle.py          # Handles obstacles & collision detection
 ├── constants.py         # Global constants
 ├── runs/                # Stored simulation data & checkpoints
```

---

## 🔬 Technical Overview  

### 🧠 Neural Network Architecture  

- **Input Layer (15 neurons)**  
  - 8 distance sensors (N, NE, E, SE, S, SW, W, NW)  
  - 2 normalized target coordinates (dx, dy)  
  - 2 velocity components (vel_x, vel_y)  
  - 2 distance metrics (nearest obstacle & target distance)  
  - 1 normalized speed value  

- **Hidden Layers**  
  - Configurable (default: **2 layers, 16 neurons each**)  
  - **Activation**: Leaky ReLU  

- **Output Layer (3 neurons)**  
  - Direction (0 - 2π)  
  - Acceleration (-1 to 1)  
  - Speed Control (0 to 1)  

### 🧬 Genetic Algorithm  

- **Selection** – Tournament selection strategy  
- **Crossover** – Single-point crossover for genetic recombination  
- **Mutation** – Controlled random weight mutations  
- **Fitness Calculation** – Distance, efficiency, progress evaluation  
- **Elitism** – Best performers are preserved across generations  

### ⚡ Performance Optimizations  

- **NumPy & TensorFlow implementations** for flexibility  
- **Optimized feedforward calculations** for fast neural processing  
- **Efficient collision detection** with early-exit optimizations  
- **Configurable execution speed** for training vs. visualization  

---

## 📊 Data Management  

Simulation runs are stored in:  

```
📂 runs/
 ├── run_[timestamp]/      # Each run has its own directory
 │   ├── checkpoints/      # Saved AI models
 │   ├── stats/            # JSON-formatted performance statistics
```

---

## 📜 License  

This project is licensed under the **MIT License**.  

---

## 🤝 Contributing  

Contributions are welcome! If you'd like to improve the project, please submit a **Pull Request**.  

---

💡 **Enhance your AI simulation skills & explore the power of neural evolution!** 🚀  
