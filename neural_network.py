"""
Pure NumPy-based neural network implementation for AI dot decision making.

This module provides a complete neural network implementation using only NumPy,
without any dependencies on TensorFlow. It serves as a fallback implementation
when TensorFlow is not available or causing performance issues.

Features:
- Pure NumPy implementation for maximum compatibility and efficiency
- Multi-layer feedforward neural network with configurable hidden layers
- Leaky ReLU activation for hidden layers and sigmoid for outputs
- Genetic algorithm support with mutation and crossover
- Serialization support for saving and loading models
"""
import numpy as np
from typing import Dict, List

class NeuralNetwork:
    """
    Neural network for AI decision making using pure NumPy (no TensorFlow).
    
    This implementation provides a completely self-contained neural network
    without any dependencies on TensorFlow or other deep learning frameworks.
    It is designed to be efficient for simple feedforward operations and
    genetic algorithm-based training in simulation environments.
    
    The network includes support for multiple hidden layers, He initialization,
    leaky ReLU activation, and numerical stability protections.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, hidden_layers: int = 1):
        """Initialize neural network with random weights and biases."""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        
        # He initialization for ReLU activation
        # This scaling helps prevent vanishing/exploding gradients
        scale_ih = np.sqrt(2.0 / input_size)
        scale_ho = np.sqrt(2.0 / hidden_size)
        scale_hh = np.sqrt(2.0 / hidden_size)
        
        # Initialize weights and biases for input-to-hidden layer
        self.weights_ih = np.random.randn(hidden_size, input_size) * scale_ih
        self.bias_h = np.zeros((hidden_size, 1))
        
        # Initialize weights and biases for hidden-to-output layer
        self.weights_ho = np.random.randn(output_size, hidden_size) * scale_ho
        
        # For multiple hidden layers
        if hidden_layers > 1:
            self.hidden_weights = []
            self.hidden_biases = []
            for _ in range(hidden_layers - 1):
                self.hidden_weights.append(np.random.randn(hidden_size, hidden_size) * scale_hh)
                self.hidden_biases.append(np.zeros((hidden_size, 1)))
        
        self.bias_o = np.zeros((output_size, 1))
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def leaky_relu(self, x: np.ndarray) -> np.ndarray:
        """Leaky ReLU activation function."""
        return np.maximum(0.01 * x, x)
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function with numerical stability."""
        # Clip values to avoid overflow
        x_clipped = np.clip(x, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-x_clipped))
    
    def tanh(self, x: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent activation function with numerical stability."""
        # Clip values to avoid overflow
        x_clipped = np.clip(x, -20.0, 20.0)
        return np.tanh(x_clipped)
    
    def feedforward(self, inputs: np.ndarray) -> np.ndarray:
        """Perform feedforward computation through the network with improved stability."""
        # Ensure input is 2D array with correct shape
        if isinstance(inputs, list):
            inputs = np.array(inputs, dtype=np.float64)
            
        if inputs.ndim == 1:
            inputs = inputs.reshape(-1, 1)
        
        # Normalize inputs for better stability (prevents extreme values)
        # Skip normalization for binary inputs (0-1 range)
        if np.max(np.abs(inputs)) > 1.0:
            inputs = np.clip(inputs, -10.0, 10.0)
        
        # Input to first hidden layer
        hidden = np.dot(self.weights_ih, inputs) + self.bias_h
        hidden = self.leaky_relu(hidden)
        
        # Additional hidden layers if any
        if self.hidden_layers > 1:
            for i in range(self.hidden_layers - 1):
                hidden = np.dot(self.hidden_weights[i], hidden) + self.hidden_biases[i]
                hidden = self.leaky_relu(hidden)
        
        # Hidden to output layer
        output = np.dot(self.weights_ho, hidden) + self.bias_o
        
        # Use sigmoid for output layer (maps to 0-1 range)
        output = self.sigmoid(output)
        
        # Ensure output does not contain NaN or inf values
        if np.any(np.isnan(output)) or np.any(np.isinf(output)):
            # If numerical issues occurred, return safe values
            print("Warning: Numerical instability detected in neural network")
            return np.ones(self.output_size) * 0.5  # Safe default
        
        # Return as numpy array with shape (output_size,)
        return np.ravel(output)
    
    def copy(self):
        """Create a copy of this neural network."""
        new_nn = NeuralNetwork(self.input_size, self.hidden_size, self.output_size, self.hidden_layers)
        
        # Copy weights and biases
        new_nn.weights_ih = self.weights_ih.copy()
        new_nn.bias_h = self.bias_h.copy()
        
        if self.hidden_layers > 1:
            new_nn.hidden_weights = [w.copy() for w in self.hidden_weights]
            new_nn.hidden_biases = [b.copy() for b in self.hidden_biases]
        
        new_nn.weights_ho = self.weights_ho.copy()
        new_nn.bias_o = self.bias_o.copy()
        
        return new_nn
    
    def mutate(self, mutation_rate: float) -> None:
        """Randomly mutate weights and biases."""
        # Function to mutate a matrix
        def mutate_matrix(matrix):
            mask = np.random.random(matrix.shape) < mutation_rate
            # Use normal distribution for mutations
            changes = np.random.normal(0, 0.3, matrix.shape)
            matrix[mask] += changes[mask]
        
        # Mutate input-to-hidden weights and biases
        mutate_matrix(self.weights_ih)
        mutate_matrix(self.bias_h)
        
        # Mutate hidden layer weights and biases
        if self.hidden_layers > 1:
            for i in range(self.hidden_layers - 1):
                mutate_matrix(self.hidden_weights[i])
                mutate_matrix(self.hidden_biases[i])
        
        # Mutate hidden-to-output weights and biases
        mutate_matrix(self.weights_ho)
        mutate_matrix(self.bias_o)
    
    def crossover(self, other: 'NeuralNetwork') -> 'NeuralNetwork':
        """Perform single-point crossover with another neural network."""
        # Ensure networks have the same structure
        if (self.input_size != other.input_size or 
            self.hidden_size != other.hidden_size or 
            self.output_size != other.output_size or
            self.hidden_layers != other.hidden_layers):
            raise ValueError("Networks must have the same structure for crossover")
        
        # Create child neural network
        child = NeuralNetwork(
            self.input_size, 
            self.hidden_size, 
            self.output_size,
            self.hidden_layers
        )
        
        # Perform crossover on input-to-hidden weights
        rows, cols = self.weights_ih.shape
        for i in range(rows):
            # Choose a random crossover point
            crossover_point = np.random.randint(0, cols)
            # Take weights from first parent up to crossover point
            child.weights_ih[i, :crossover_point] = self.weights_ih[i, :crossover_point]
            # Take weights from second parent after crossover point
            child.weights_ih[i, crossover_point:] = other.weights_ih[i, crossover_point:]
        
        # Randomly select bias values from either parent
        for i in range(child.bias_h.shape[0]):
            if np.random.random() < 0.5:
                child.bias_h[i] = self.bias_h[i]
            else:
                child.bias_h[i] = other.bias_h[i]
        
        # Crossover for hidden layers
        if self.hidden_layers > 1:
            for layer in range(self.hidden_layers - 1):
                rows, cols = self.hidden_weights[layer].shape
                for i in range(rows):
                    crossover_point = np.random.randint(0, cols)
                    child.hidden_weights[layer][i, :crossover_point] = self.hidden_weights[layer][i, :crossover_point]
                    child.hidden_weights[layer][i, crossover_point:] = other.hidden_weights[layer][i, crossover_point:]
                
                # Select biases
                for i in range(child.hidden_biases[layer].shape[0]):
                    if np.random.random() < 0.5:
                        child.hidden_biases[layer][i] = self.hidden_biases[layer][i]
                    else:
                        child.hidden_biases[layer][i] = other.hidden_biases[layer][i]
        
        # Crossover for hidden-to-output weights
        rows, cols = self.weights_ho.shape
        for i in range(rows):
            crossover_point = np.random.randint(0, cols)
            child.weights_ho[i, :crossover_point] = self.weights_ho[i, :crossover_point]
            child.weights_ho[i, crossover_point:] = other.weights_ho[i, crossover_point:]
        
        # Select output biases
        for i in range(child.bias_o.shape[0]):
            if np.random.random() < 0.5:
                child.bias_o[i] = self.bias_o[i]
            else:
                child.bias_o[i] = other.bias_o[i]
        
        return child
    
    def serialize(self) -> Dict:
        """Convert neural network to serializable format."""
        data = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'hidden_layers': self.hidden_layers,
            'weights_ih': self.weights_ih.tolist(),
            'bias_h': self.bias_h.tolist(),
            'weights_ho': self.weights_ho.tolist(),
            'bias_o': self.bias_o.tolist()
        }
        
        if self.hidden_layers > 1:
            data['hidden_weights'] = [w.tolist() for w in self.hidden_weights]
            data['hidden_biases'] = [b.tolist() for b in self.hidden_biases]
        
        return data
    
    @classmethod
    def deserialize(cls, data: Dict) -> 'NeuralNetwork':
        """Create a neural network from serialized data."""
        nn = cls(
            data['input_size'],
            data['hidden_size'],
            data['output_size'],
            data.get('hidden_layers', 1)
        )
        
        nn.weights_ih = np.array(data['weights_ih'])
        nn.bias_h = np.array(data['bias_h'])
        nn.weights_ho = np.array(data['weights_ho'])
        nn.bias_o = np.array(data['bias_o'])
        
        if nn.hidden_layers > 1 and 'hidden_weights' in data:
            nn.hidden_weights = [np.array(w) for w in data['hidden_weights']]
            nn.hidden_biases = [np.array(b) for b in data['hidden_biases']]
        
        return nn 