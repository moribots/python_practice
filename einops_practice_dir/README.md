# Einops Study Guide

Einops (Einstein Operations) is a library for tensor operations that makes complex rearrangements readable and efficient. It's particularly useful in deep learning for manipulating tensor dimensions, especially in computer vision, NLP, and robotics perception tasks.

## Key Concepts

### Rearrange
- Changes tensor shape by rearranging dimensions
- Syntax: `einops.rearrange(tensor, pattern)`
- Pattern uses dimension names: 'b h w c' for batch, height, width, channels

### Repeat
- Repeats tensor along specified dimensions
- Syntax: `einops.repeat(tensor, pattern)`

### Reduce
- Reduces tensor along dimensions (sum, mean, max, etc.)
- Syntax: `einops.reduce(tensor, pattern, reduction)`

## Interview-Ready Concepts

### Why Einops Matters for Robotics/ML
- **Computer Vision**: Image preprocessing, batch processing, channel manipulation
- **NLP**: Sequence processing, attention mechanisms, multi-head operations
- **Reinforcement Learning**: State representation, action spaces, experience replay
- **Performance**: More efficient than manual transpose/reshape chains

## Worked Examples

### Problem 1: Image Tensor Rearrangement
**Concept**: Convert from (batch, height, width, channels) to (batch, channels, height, width) - common for PyTorch models.

```python
import einops
import numpy as np

# Input: (2, 32, 32, 3) - 2 images, 32x32, RGB
tensor = np.random.rand(2, 32, 32, 3)

# Rearrange to (2, 3, 32, 32) - channels first
result = einops.rearrange(tensor, 'b h w c -> b c h w')
print(result.shape)  # (2, 3, 32, 32)
```

### Problem 2: Sequence Flattening
**Concept**: Flatten sequence and hidden dimensions for processing.

```python
# Input: (batch, seq_len, hidden) -> (batch, seq_len * hidden)
tensor = np.random.rand(4, 10, 64)
result = einops.rearrange(tensor, 'b s h -> b (s h)')
print(result.shape)  # (4, 640)
```

### Problem 3: Dimension Splitting
**Concept**: Split a dimension into multiple parts.

```python
# Input: (batch, 2*hidden) -> (batch, 2, hidden)
tensor = np.random.rand(4, 128)
result = einops.rearrange(tensor, 'b (h d) -> b h d', h=2)
print(result.shape)  # (4, 2, 64)
```

### Problem 4: Attention Transpose
**Concept**: Rearrange for multi-head attention.

```python
# Input: (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
tensor = np.random.rand(2, 50, 8, 64)
result = einops.rearrange(tensor, 'b s h d -> b h s d')
print(result.shape)  # (2, 8, 50, 64)
```

### Problem 5: Channel Repetition
**Concept**: Repeat along channel dimension.

```python
# Input: (batch, seq) -> (batch, seq, 3) by repeating
tensor = np.random.rand(2, 10)
result = einops.repeat(tensor, 'b s -> b s 3')
print(result.shape)  # (2, 10, 3)
```

## Advanced Interview Topics

### Reduce Operations
```python
# Global average pooling
tensor = np.random.rand(2, 32, 32, 3)
result = einops.reduce(tensor, 'b h w c -> b c', 'mean')
print(result.shape)  # (2, 3)

# Max pooling across spatial dimensions
result = einops.reduce(tensor, 'b h w c -> b c', 'max')
```

### Complex Rearrangements
```python
# From (batch, time, height, width, channels) to (batch*time, channels, height, width)
video = np.random.rand(2, 10, 32, 32, 3)
frames = einops.rearrange(video, 'b t h w c -> (b t) c h w')
print(frames.shape)  # (20, 3, 32, 32)
```

### Memory Layout Considerations
- Einops operations are often more memory-efficient than multiple transposes
- Use for preprocessing pipelines in robotics vision systems
- Critical for real-time applications with memory constraints

## Practice Tips
- Always write the pattern clearly: source -> target
- Use meaningful dimension names
- Test with small tensors first
- Einops is more readable than manual transpose/reshape chains
- Common in robotics for sensor data processing (LiDAR, camera, IMU)
