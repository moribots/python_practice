import torch
import torch.nn as nn
import torch.nn.functional as F

# Problem 1: Define a simple MLP with 2 hidden layers


class Problem1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # TODO: Define layers
        pass

    def forward(self, x):
        # TODO: Implement forward pass
        pass

# Problem 2: Compute MSE loss


def problem2(pred, target):
    # TODO: Return MSE loss
    pass

# Problem 3: Implement a convolutional layer forward (manual)


def problem3(input_tensor, weight, bias):
    # TODO: Manual conv2d
    pass

# Problem 4: Define an RNN cell


class Problem4(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # TODO: Define weights
        pass

    def forward(self, x, h):
        # TODO: Implement RNN step
        pass

# Problem 5: Compute gradients for a simple function


def problem5():
    # TODO: Use torch.autograd to compute grad of x^2
    pass
