import numpy as np
from typing import Tuple
from base import ParamLayer

class Conv2D(ParamLayer):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.parameters = {
            'W' : np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * \
            np.sqrt(2.0 / (in_channels * kernel_size * kernel_size)),
            'b' : np.zeros((out_channels, 1, 1))
        }
        self.zero_grad()

    def _pad_input(self, inputs):
        if self.padding > 0:
            return np.pad(
                inputs,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode = 'constant'
            )
        
        return inputs
    
    def _extract_windows(self, inputs):
        N, C, H, W = inputs.shape
        out_h = (H - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_w = (W - self.kernel_size + 2 * self.padding) // self.stride + 1

        # extract windows
        x_padded = self._pad_input(inputs)
        windows = np.zeros((N, C, self.kernel_size, self.kernel_size,
                            out_h, out_w))
        
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                # windows [batch, channel, kernel_size, kernel_size, out_h, out_w]
                windows[:, :, i, j, :, :] = x_padded[:, :, 
                    i*self.stride:i*self.stride + out_h*self.stride:self.stride,
                    j*self.stride:j*self.stride + out_w*self.stride:self.stride]
        
        windows = windows.transpose(0, 4, 5, 1, 2, 3).reshape(-1, C*self.kernel_size*self.kernel_size)
        # example: (N, out_h, out_w, C, kernel_size, kernel_size) -> (N*out_h*out_w, C*kernel_size*kernel_size)

        return windows, (N, out_h, out_w)
    
    def forward(self, inputs):
        self.inputs = inputs
        N, C, H, W = inputs.shape

        self.windows, (N, out_h, out_w) = self._extract_windows(inputs)
        weights = self.parameters['W'].reshape(self.out_channels, -1)

        output = np.dot(self.windows, weights.T)
        output = output.reshape(N, out_h, out_w, self.out_channels)
        output = output.transpose(0, 3, 1, 2)

        output += self.parameters['b']

        return output
    
    def backward(self, grad_output):
        N = self.inputs.shape[0]

        # Reshape gradients for computation
        grad_output_reshaped = grad_output.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        # example: (N, out_h, out_w, out_channels) -> (N*out_h*out_w, out_channels)

        # Compute gradients for weights
        self.gradients['W'] = np.dot(grad_output_reshaped.T, self.windows)
        self.gradients['W'] = self.gradients['W'].reshape(self.parameters['W'].shape)  
        self.gradients['W'] /= N
        self.gradients['b'] = np.sum(grad_output, axis = (0, 2, 3), keepdims = True) / N

        # Compute gradient for input
        grad_input = np.zeros_like(self._pad_input(self.inputs))
        _, out_h, out_w = grad_output.shape[1:]

        # Distribute gradients back to input positions
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                grad_input[:, :, i*self.stride:i*self.stride + out_h*self.stride:self.stride,
                           j*self.stride:j*self.stride + out_w*self.stride:self.stride] += \
                             grad_output_reshaped.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        
        if self.padding > 0:
            return grad_input[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return grad_input
    
    def update_params(self, learning_rate: float) -> None:
        """Update parameters using gradients"""
        for key in self.parameters:
            self.parameters[key] -= learning_rate * self.gradients[key]


if __name__ == "__main__":
    # Test case 1: Simple 3x3 input with one channel
    simple_input = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]).reshape(1, 1, 3, 3)  # (N, C, H, W)

    # Create a Conv2D layer
    conv = Conv2D(
        in_channels=1,
        out_channels=1,
        kernel_size=2,
        stride=1,
        padding=0
    )

    # Set specific weights for demonstration
    conv.parameters['W'] = np.array([
        [1, 0],
        [0, 1]
    ]).reshape(1, 1, 2, 2)  # (out_channels, in_channels, kernel_size, kernel_size)
    conv.parameters['b'] = np.zeros((1, 1, 1))

    print("\nTest Case 1: Simple 3x3 input")
    print("Input shape:", simple_input.shape)
    print("\nInput:")
    print(simple_input.squeeze())
    print("\nKernel:")
    print(conv.parameters['W'].squeeze())

    # Forward pass
    output = conv.forward(simple_input)
    print("\nOutput after forward pass:")
    print(output.squeeze())
    print("Output shape:", output.shape)

    # Let's see what windows look like
    print("\nExtracted windows:")
    print(conv.windows)
    print("Windows shape:", conv.windows.shape)

    # Test backward pass
    grad_output = np.array([
        [10, 20],
        [30, 40]
    ]).reshape(1, 1, 2, 2)
    
    print("\nGradient from next layer:")
    print(grad_output.squeeze())
    
    # Reshape grad_output for backward pass
    grad_output_reshaped = grad_output.transpose(0, 2, 3, 1).reshape(-1, 1)
    print("\nReshaped gradient:")
    print(grad_output_reshaped.squeeze())
    
    grad_input = conv.backward(grad_output)
    print("\nGradient w.r.t input:")
    print(grad_input.squeeze())
    print("Gradient shape:", grad_input.shape)

    print("\nTests completed!")