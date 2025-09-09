import einops
import numpy as np

# Problem 1: Rearrange image tensor from (batch, height, width, channels) to (batch, channels, height, width)


def problem1(tensor):
    # TODO: Implement using einops.rearrange
    pass

# Problem 2: Flatten the last two dimensions of a tensor (batch, seq_len, hidden) to (batch, seq_len * hidden)


def problem2(tensor):
    # TODO: Implement using einops.rearrange
    pass

# Problem 3: Split a tensor along a dimension: from (batch, 2*hidden) to (batch, 2, hidden)


def problem3(tensor):
    # TODO: Implement using einops.rearrange
    pass

# Problem 4: Transpose dimensions: from (batch, seq, heads, head_dim) to (batch, heads, seq, head_dim)


def problem4(tensor):
    # TODO: Implement using einops.rearrange
    pass

# Problem 5: Repeat along a dimension: from (batch, seq) to (batch, seq, 3) by repeating


def problem5(tensor):
    # TODO: Implement using einops.repeat
    pass
