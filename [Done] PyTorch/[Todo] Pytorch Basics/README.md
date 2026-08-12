# Pytorch Basics

This is a pretty big topic but in this I will be only convering 4 main basic topics of pytorch that we may use ahead, for other topics we must refer to the official [pytorch](https://docs.pytorch.org/docs/2.13/index.html) documentaion or this resource by [Stanford CS230 Deep Learning](https://cs230.stanford.edu/blog/pytorch/)

## 1. Reshaping using Pytorch

The reshape method in pytorch is mainly used to chnage the dimnesions of a tensor without altering the actual data. It returns the exact same elemnts arranged in a different dimention

```python
import torch

x = torch.arange(12)  # Creates a 1D tensor: [0, 1, 2, ..., 11]

# Both approaches yield the exact same result
y1 = torch.reshape(x, (3, 4))
y2 = x.reshape(3, 4)

```

## 2. Average

Mainly used to calulcate the mean either column wise or wise, the input must be a floating point tensor. It calculates along specific dimnetion using the "dim" argument

```python
matrix = torch.tensor([[1.0, 2.0], 
                       [3.0, 4.0]])

# Mean of each column (down dimension 0)
print(torch.mean(matrix, dim=0)) 
# Output: tensor([2., 3.])

# Mean of each row (across dimension 1)
print(torch.mean(matrix, dim=1)) 
# Output: tensor([1.5000, 3.5000])

```

## 3. Concatenete

This is used to merge or join tensors side by side or top to bottom. We use `torch.cat` for this and just pass the tensors and the `dim` we want to join them on.

```python
t1 = torch.tensor([[1, 2]])
t2 = torch.tensor([[3, 4]])

# Join side by side (across dimension 1)
print(torch.cat((t1, t2), dim=1))
# Output: tensor([[1, 2, 3, 4]])

```

## 4. Loss

Used to calculate the loss. `torch.nn.functional.mse_loss` is a very common one that calculates the mean squared error between your models predictions and the actual targets.

```python
import torch.nn.functional as F

preds = torch.tensor([1.5, 2.5])
targets = torch.tensor([1.0, 2.0])

# Calculate how far off the predictions are
loss = F.mse_loss(preds, targets)
print(loss)
# Output: tensor(0.2500)

```