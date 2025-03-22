"""
TensorFlow-based neural network implementation for AI dot decision making.

This module implements a neural network using TensorFlow/Keras for AI decision-making.
It includes an optimized NumPy-based fast path for feedforward operations, which provides
better performance during simulation while maintaining compatibility with TensorFlow
for advanced features.

Features:
- TensorFlow-based neural network architecture with configurable hidden layers
- NumPy-based fast path for efficient feedforward computation
- Genetic algorithm support with mutation and crossover
- Serialization support for saving and loading models
"""
import os
import json
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Any, Union

# Ensure TensorFlow logging is not too verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Instead of disabling eager execution, explicitly enable it (it's on by default in TF 2.x)
tf.compat.v1.enable_eager_execution()

class TFNeuralNetwork:
    """
    Neural network for AI decision making using TensorFlow with NumPy fast path.
    
    This implementation uses TensorFlow/Keras for the model architecture, but provides
    an optional NumPy-based fast path for feedforward operations, which is much faster
    for simple inference in simulation environments.
    
    The neural network supports genetic algorithm operations like mutation and crossover,
    making it suitable for evolving populations of AI agents.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, hidden_layers: int = 1, use_numpy_fast_path: bool = True):
        """
        Initialize neural network with TensorFlow/Keras model.
        
        Args:
            input_size: Number of input neurons
            hidden_size: Number of neurons in each hidden layer
            output_size: Number of output neurons
            hidden_layers: Number of hidden layers
            use_numpy_fast_path: If True, use numpy for feedforward (faster but less flexible)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.use_numpy_fast_path = use_numpy_fast_path
        
        # Create the Keras model
        self.model = self._build_model()
        
        # For tracking when we need to compile or rebuild the model
        self.is_compiled = True
        
        # For fast numpy-based execution
        self.weights = None
        self.biases = None
        if use_numpy_fast_path:
            self._extract_weights_to_numpy()
    
    def _build_model(self) -> tf.keras.Model:
        """
        Build and compile the Keras model.
        
        Returns:
            Compiled Keras Sequential model
        """
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(tf.keras.layers.Dense(
            self.hidden_size, 
            activation='relu',
            input_shape=(self.input_size,),
            kernel_initializer=tf.keras.initializers.HeNormal()  # He initialization for ReLU
        ))
        
        # Hidden layers
        for _ in range(self.hidden_layers - 1):
            model.add(tf.keras.layers.Dense(
                self.hidden_size, 
                activation='relu',
                kernel_initializer=tf.keras.initializers.HeNormal()
            ))
        
        # Output layer
        model.add(tf.keras.layers.Dense(
            self.output_size, 
            activation='sigmoid',  # Use sigmoid for 0-1 output range
            kernel_initializer=tf.keras.initializers.GlorotNormal()  # Xavier/Glorot for sigmoid
        ))
        
        # Compile model - we don't need training but compiling sets up the model for prediction
        model.compile(
            optimizer='adam',
            loss='mse'  # This doesn't matter much since we're not training normally
        )
        
        return model
    
    def _extract_weights_to_numpy(self):
        """Extract weights from TensorFlow model to numpy arrays for fast execution."""
        # Use model.get_weights() which returns numpy arrays directly, 
        # avoiding the need to call .numpy() on tensor variables
        all_weights = self.model.get_weights()
        
        # Weights are stored as [W1, b1, W2, b2, ...]
        self.weights = []
        self.biases = []
        
        for i in range(0, len(all_weights), 2):
            self.weights.append(all_weights[i])
            self.biases.append(all_weights[i+1])
    
    def _numpy_feedforward(self, inputs: np.ndarray) -> np.ndarray:
        """Fast numpy-based feedforward implementation."""
        # Input layer
        x = inputs
        
        # Hidden layers
        for i in range(len(self.weights) - 1):
            # Linear transformation
            # Note: TensorFlow weights have shape [input_size, output_size]
            # No need to transpose for matrix multiplication
            x = np.dot(x, self.weights[i]) + self.biases[i]
            # ReLU activation
            x = np.maximum(0.01 * x, x)  # Leaky ReLU
        
        # Output layer - sigmoid activation for final layer
        x = np.dot(x, self.weights[-1]) + self.biases[-1]
        # Use sigmoid for output (matching the model's sigmoid activation)
        x = 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))
        
        return x
    
    def feedforward(self, inputs: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Perform feedforward computation through the network.
        
        Args:
            inputs: Input values as numpy array or list
            
        Returns:
            Numpy array of outputs
        """
        # Ensure input is the right shape and type
        if isinstance(inputs, list):
            inputs = np.array(inputs, dtype=np.float32)
        
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)  # Add batch dimension
        
        # Clamp inputs for stability
        inputs = np.clip(inputs, -10.0, 10.0)
        
        # Use numpy fast path if enabled (much faster)
        if self.use_numpy_fast_path and self.weights is not None:
            return self._numpy_feedforward(inputs).flatten()
        
        # Otherwise use TensorFlow model
        with tf.device('/CPU:0'):  # Force CPU for small batches (likely faster than GPU transfer)
            outputs = self.model.predict(inputs, verbose=0)
        
        # Return flattened outputs
        return outputs.flatten()
    
    def copy(self) -> 'TFNeuralNetwork':
        """
        Create a deep copy of this neural network.
        
        Returns:
            New TFNeuralNetwork instance with the same weights
        """
        # Create new network with same architecture
        new_nn = TFNeuralNetwork(
            self.input_size,
            self.hidden_size,
            self.output_size,
            self.hidden_layers
        )
        
        # Copy weights from this model to new model
        # We first need to get all the weights
        weights = self.model.get_weights()
        
        # Then set them in the new model
        new_nn.model.set_weights(weights)
        
        return new_nn
    
    def mutate(self, mutation_rate: float) -> None:
        """
        Randomly mutate weights and biases of the network.
        
        Args:
            mutation_rate: Probability of mutation for each weight/bias
        """
        # Get current weights
        weights = self.model.get_weights()
        
        # Mutate weights
        for i in range(len(weights)):
            # Create mask of which weights to mutate
            mask = np.random.random(weights[i].shape) < mutation_rate
            
            # Create random changes for the masked elements
            changes = np.random.normal(0, 0.3, weights[i].shape)
            
            # Apply changes only where mask is True
            weights[i][mask] += changes[mask]
        
        # Set mutated weights back to model
        self.model.set_weights(weights)
        
        # Update numpy weights and biases for fast path if enabled
        if self.use_numpy_fast_path:
            self._extract_weights_to_numpy()
    
    def _single_layer_crossover(self, layer1: np.ndarray, layer2: np.ndarray) -> np.ndarray:
        """
        Perform single-point crossover on two weight matrices.
        
        Args:
            layer1: First parent's layer weights
            layer2: Second parent's layer weights
            
        Returns:
            New weight matrix with genetic material from both parents
        """
        # Create a copy to modify
        child_layer = layer1.copy()
        
        if layer1.ndim == 1:  # For bias vectors
            # Choose a random crossover point
            crossover_point = np.random.randint(0, layer1.shape[0])
            # Take values from second parent after crossover point
            child_layer[crossover_point:] = layer2[crossover_point:]
        else:  # For weight matrices
            # Crossover each row at a different point
            for i in range(layer1.shape[0]):
                crossover_point = np.random.randint(0, layer1.shape[1])
                child_layer[i, crossover_point:] = layer2[i, crossover_point:]
                
        return child_layer
    
    def crossover(self, other: 'TFNeuralNetwork') -> 'TFNeuralNetwork':
        """
        Perform crossover with another network to create a child network.
        
        Args:
            other: Another neural network to crossover with
            
        Returns:
            New neural network with genetic material from both parents
        """
        # Ensure compatible networks
        if (self.input_size != other.input_size or
            self.hidden_size != other.hidden_size or
            self.output_size != other.output_size or
            self.hidden_layers != other.hidden_layers):
            raise ValueError("Neural networks must have the same architecture for crossover")
        
        # Create child network with same architecture
        child = TFNeuralNetwork(
            self.input_size,
            self.hidden_size,
            self.output_size,
            self.hidden_layers,
            use_numpy_fast_path=self.use_numpy_fast_path
        )
        
        # Get weights from both parents
        parent1_weights = self.model.get_weights()
        parent2_weights = other.model.get_weights()
        child_weights = []
        
        # Crossover each layer
        for layer1, layer2 in zip(parent1_weights, parent2_weights):
            child_layer = self._single_layer_crossover(layer1, layer2)
            child_weights.append(child_layer)
        
        # Set child's weights
        child.model.set_weights(child_weights)
        
        # Update numpy weights and biases for fast path
        if child.use_numpy_fast_path:
            child._extract_weights_to_numpy()
        
        return child
    
    def serialize(self) -> Dict:
        """
        Convert neural network to serializable format.
        
        Returns:
            Dictionary with model parameters and weights
        """
        # Get model weights as list of numpy arrays
        weights = self.model.get_weights()
        
        # Convert weights to nested lists for JSON serialization
        weights_list = [w.tolist() for w in weights]
        
        data = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'hidden_layers': self.hidden_layers,
            'weights': weights_list
        }
        
        return data
    
    @classmethod
    def deserialize(cls, data: Dict) -> 'TFNeuralNetwork':
        """
        Create a neural network from serialized data.
        
        Args:
            data: Dictionary with model parameters and weights
            
        Returns:
            Reconstructed neural network
        """
        # Create network with architecture from data
        nn = cls(
            data['input_size'],
            data['hidden_size'],
            data['output_size'],
            data.get('hidden_layers', 1)
        )
        
        # Convert weight lists back to numpy arrays
        weights = [np.array(w, dtype=np.float32) for w in data['weights']]
        
        # Set weights in the model
        nn.model.set_weights(weights)
        
        # Update numpy weights and biases for fast path
        if nn.use_numpy_fast_path:
            nn._extract_weights_to_numpy()
        
        return nn 