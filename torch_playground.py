import torch

# Create tensor 
x = torch.tensor([1, 2, 3]) 

# Indexing
print(x[0]) 

# Math operations
print(x + 2)

# Reshape 
x = x.view(3, 1)
print(x)

# One
torch.ones(3, 1)

# Broadcasting
print(x + torch.ones(3, 1)) 

# Concatenate 
y = torch.tensor([4, 5, 6]).view(3, 1)
print(torch.cat([x, y], dim=1))

# Linear algebra
print(x.matmul(y.t())) 

# Reduce
print(x.sum())

# GPU tensor
if torch.cuda.is_available():
  x = x.to('cuda') 

# Gradient
floating_tensor = torch.tensor([1, 2, 3], dtype=torch.float, requires_grad=True)
floating_tensor.requires_grad_()
floating_tensor.backward(torch.randn(3))
print(floating_tensor.grad)

# 3x3 tensor x filled with random values
y = torch.rand(3, 3)
print(y)