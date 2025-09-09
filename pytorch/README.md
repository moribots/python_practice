# PyTorch Study Guide

PyTorch is the primary deep learning framework for robotics research, used for neural networks, reinforcement learning, and computer vision. Robotics interviews often test deep learning fundamentals, PyTorch proficiency, and ML system design.

## Key Concepts

### Tensors
- Multi-dimensional arrays
- GPU acceleration with `.cuda()`
- Autograd for automatic differentiation

### Neural Networks
- `nn.Module`: Base class for models
- `nn.Linear`: Fully connected layers
- `nn.ReLU`: Activation functions

### Loss Functions
- `nn.MSELoss`: Mean squared error
- `nn.CrossEntropyLoss`: Classification

### Optimization
- `torch.optim.SGD`: Stochastic gradient descent
- `optimizer.step()`: Update parameters

### Autograd
- `tensor.backward()`: Compute gradients
- `tensor.grad`: Access gradients

## Interview-Ready Concepts

### Training Pipeline
- Data loading with `DataLoader`
- Model training loops
- Validation and testing
- Model serialization

### Advanced Layers
- Convolutional layers (`nn.Conv2d`)
- Recurrent layers (`nn.LSTM`, `nn.GRU`)
- Attention mechanisms
- Normalization layers

### Optimization Techniques
- Different optimizers (Adam, RMSprop)
- Learning rate scheduling
- Regularization (dropout, weight decay)
- Gradient clipping

## Worked Examples

### Problem 1: Simple MLP
```python
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Test
model = SimpleMLP(10, 5, 1)
x = torch.randn(2, 10)
output = model(x)
print(output.shape)  # torch.Size([2, 1])
```

### Problem 2: MSE Loss
```python
# Compute MSE between predictions and targets
pred = torch.tensor([1.0, 2.0, 3.0])
target = torch.tensor([1.5, 2.5, 2.8])

mse_loss = nn.MSELoss()
loss = mse_loss(pred, target)
print(loss.item())  # 0.0467
```

### Problem 3: Manual Convolution
```python
def manual_conv2d(input_tensor, weight, bias):
    # Simple 2D convolution implementation
    batch, in_channels, height, width = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    out_height = height - kernel_h + 1
    out_width = width - kernel_w + 1
    
    output = torch.zeros(batch, out_channels, out_height, out_width)
    
    for b in range(batch):
        for oc in range(out_channels):
            for h in range(out_height):
                for w in range(out_width):
                    region = input_tensor[b, :, h:h+kernel_h, w:w+kernel_w]
                    output[b, oc, h, w] = torch.sum(region * weight[oc]) + bias[oc]
    
    return output

# Test
input_tensor = torch.randn(1, 1, 4, 4)
weight = torch.randn(1, 1, 2, 2)
bias = torch.zeros(1)
result = manual_conv2d(input_tensor, weight, bias)
print(result.shape)  # torch.Size([1, 1, 3, 3])
```

### Problem 4: Simple RNN
```python
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
    
    def forward(self, x, h):
        combined = torch.cat((x, h), 1)
        h_new = torch.tanh(self.i2h(combined))
        return h_new, h_new  # output, new_hidden

# Test
rnn = SimpleRNN(10, 5)
x = torch.randn(1, 10)
h = torch.randn(1, 5)
output, new_h = rnn(x, h)
print(output.shape)  # torch.Size([1, 5])
```

### Problem 5: Autograd Gradients
```python
# Compute gradient of x^2 at x=1
x = torch.tensor(1.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # tensor(2.)
```

## Advanced Interview Topics

### Complete Training Loop
```python
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Create dummy dataset
X = torch.randn(100, 10)
y = torch.randn(100, 1)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model, loss, optimizer
model = SimpleMLP(10, 5, 1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
```

### Custom Loss Functions
```python
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # Huber loss (less sensitive to outliers than MSE)
        diff = pred - target
        abs_diff = torch.abs(diff)
        quadratic = torch.min(abs_diff, torch.ones_like(abs_diff))
        linear = abs_diff - quadratic
        return torch.mean(quadratic * quadratic + linear)

# Test
loss_fn = CustomLoss()
pred = torch.randn(10, 1)
target = torch.randn(10, 1)
loss = loss_fn(pred, target)
print(loss.item())
```

### Model Saving and Loading
```python
# Save model
torch.save(model.state_dict(), 'model.pth')

# Load model
model = SimpleMLP(10, 5, 1)
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set to evaluation mode
```

### GPU Operations
```python
# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model and data to GPU
model = model.to(device)
X = X.to(device)
y = y.to(device)

# Forward pass on GPU
with torch.no_grad():
    outputs = model(X)
```

### Gradient Clipping
```python
# Clip gradients to prevent exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## Practice Tips
- Use `torch.no_grad()` for inference
- Move models to GPU: `model.cuda()`
- Batch operations for efficiency
- Understand tensor shapes and broadcasting
- Use `torchsummary` for model visualization
- Always set model to eval mode for inference
- Handle device placement consistently
