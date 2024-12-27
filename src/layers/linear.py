import numpy as np
from typing import Tuple, Dict
from base import ParamLayer

class Linear(ParamLayer):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize parameters (He initialization)
        self.parameters = {
            'W': np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features),
            'b': np.zeros((out_features, 1))
        }
        self.zero_grad()

    def _flatten_input(self, inputs):
        """
        Flatten input tensor for linear layer
        Returns flattened input and original shape for backward pass
        """
        original_shape = inputs.shape
        if len(original_shape) > 2:
            # For conv layer input (N, C, H, W)
            N = original_shape[0]
            flattened = inputs.reshape(N, -1).T
        else:
            # Already flattened
            flattened = inputs
        
        return flattened, original_shape
    
    def forward(self, inputs):
        self.inputs, self.original_shape = self._flatten_input(inputs)

        output = np.dot(self.parameters['W'], self.inputs) + self.parameters['b']
        return output

    def backward(self, grad_output):
        N = self.inputs.shape[1]

        # Compute gradients
        self.gradients['W'] = np.dot(grad_output, self.inputs.T) / N
        self.gradients['b'] = np.sum(grad_output, axis=1, keepdims=True) / N

        # Compute gradient with respect to input
        grad_input = np.dot(self.parameters['W'].T, grad_output)

        # reshape gradient to match input shape
        if len(self.original_shape) > 2:
            grad_input = grad_input.T.reshape(self.original_shape)

        return grad_input
    
    def update_params(self, learning_rate: float):
        for key in self.parameters:
            self.parameters[key] -= learning_rate * self.gradients[key]
        
    def get_output_shape(self, input_shape: Tuple) -> Tuple:
        """Calculate output shape"""
        batch_size = input_shape[0]
        return (batch_size, self.out_features)


if __name__ == "__main__":
    # Test case 1: Simple 3-dimensional input to 2-dimensional output
    print("\nTest Case 1: 3D to 2D transformation")
    
    # Create sample input (3 features, batch size of 2)
    simple_input = np.array([
        [1, 2],  # First sample
        [3, 4],  # Second sample
        [5, 6]   # Third sample
    ])  # Shape: (3, 2) - (in_features, batch_size)
    
    # Create Linear layer
    linear = Linear(in_features=3, out_features=2)
    
    # Set specific weights for demonstration
    linear.parameters['W'] = np.array([
        [0.1, 0.2, 0.3],  # First output neuron
        [0.4, 0.5, 0.6]   # Second output neuron
    ])  # Shape: (out_features, in_features)
    
    linear.parameters['b'] = np.array([[0.1], [0.2]])  # Shape: (out_features, 1)
    
    print("\nInput shape:", simple_input.shape)
    print("Input:\n", simple_input)
    print("\nWeights:\n", linear.parameters['W'])
    print("\nBias:\n", linear.parameters['b'])
    
    # Forward pass
    output = linear.forward(simple_input)
    print("\nOutput after forward pass:")
    print(output)
    print("Output shape:", output.shape)
    
    # Expected output calculation for verification
    expected_output = np.dot(linear.parameters['W'], simple_input) + linear.parameters['b']
    print("\nExpected output:")
    print(expected_output)
    print("Forward pass matches expected output:", np.allclose(output, expected_output))
    
    # Test backward pass
    grad_output = np.array([
        [1.0, 2.0],  # Gradient for first output neuron
        [3.0, 4.0]   # Gradient for second output neuron
    ])  # Shape: (out_features, batch_size)
    
    print("\nGradient from next layer:")
    print(grad_output)
    
    # Backward pass
    grad_input = linear.backward(grad_output)
    print("\nGradient w.r.t input:")
    print(grad_input)
    print("Gradient shape:", grad_input.shape)
    
    # Expected gradients calculation for verification
    expected_grad_input = np.dot(linear.parameters['W'].T, grad_output)
    print("\nExpected gradient w.r.t input:")
    print(expected_grad_input)
    print("Backward pass matches expected gradient:", np.allclose(grad_input, expected_grad_input))
    
    # Test case 2: 4D input (from conv layer)
    print("\n\nTest Case 2: 4D input from convolutional layer")
    conv_input = np.random.randn(2, 3, 4, 4)  # (batch_size, channels, height, width)
    print("Conv input shape:", conv_input.shape)
    
    # Create Linear layer
    linear_conv = Linear(in_features=3*4*4, out_features=10)
    
    # Forward pass with 4D input
    output_conv = linear_conv.forward(conv_input)
    print("Output shape after flattening conv input:", output_conv.shape)
    
    # Test parameter updates
    linear_conv.update_params(learning_rate=0.01)
    print("\nParameters updated successfully!")
    
    print("\nAll tests completed!")