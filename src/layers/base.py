from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional, Dict

class Layer(ABC):
    """Abstract base class for all neural network layers"""

    def __init__(self):
        self.input_shape: Optional[Tuple] = None
        self.output_shape: Optional[Tuple] = None
        self.trainable: bool = True
        self.training: bool = True

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Wrapper for forward pass"""
        return self.forward(inputs)
    
    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass of the layer"""
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass of the layer"""
        raise NotImplementedError
    
    def update_params(self, learning_rate: float) -> None:
        """Update layer parameters using gradients"""
        pass

    def set_training_mode(self, mode: bool) -> None:
        """Set layer to training or evaluation mode"""
        self.training = mode
    
    @abstractmethod
    def get_params(self) -> Dict:
        """Get layer parameters"""
        return {}
    
    @abstractmethod
    def set_params(self, params: Dict) -> None:
        """Set layer parameters"""
        pass

    def _validate_input_shape(self, input_shape: Tuple) -> None:
        """Validate input shape and set output shape"""
        if self.input_shape is None:
            self.input_shape = input_shape
        elif input_shape != self.input_shape:
            raise ValueError(
                f"Expected input shape {self.input_shape}, got {input_shape}"
            )
        
class ParamLayer(Layer):
    """Base class for layers with trainable parameters"""
    
    def __init__(self):
        super().__init__()
        self.gradients: Dict = {}
        self.parameters: Dict = {}

    def zero_grad(self) -> None:
        """Reset gradients to zero"""
        self.gradients = {key: np.zeros_like(value)
                          for key, value in self.parameters.items()}
        
    def get_params(self):
        """Get layer parameters"""
        return self.parameters.copy()
    
    def set_params(self, params):
        """Set layer parameters"""
        self.parameters = params.copy()
        