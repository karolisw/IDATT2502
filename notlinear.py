# predicts circumferance of a baby's head using age

import torch
import matplotlib.pyplot as plt
import pandas as pd # for reading csv

# Read in the columns
day_column = pd.read_csv("day_head_circumference.csv", usecols=[0])
circumference_column = pd.read_csv("day_head_circumference.csv", usecols=[1])

# Putting length -and weight values into their own tensors
x_train = torch.Tensor(day_column.values).reshape(-1, 1)
y_train = torch.Tensor(circumference_column.values).reshape(-1, 1)

class RegressionModel:

    def __init__(self):
    
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor --> f (x) = 20σ(xW + b) + 31

    def f(self, x):
        return 20 * torch.sigmoid( x @ self.W + self.b) + 31

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)
        # return torch.mean(torch.square(self.f(x) - y))  # Can also use torch.nn.functional.mse_loss(self.f(x), y) to possibly increase numberical stability


model = RegressionModel()

# Optimize: adjust W and b to minimize loss using "A Method for Stochastic Optimization"
optimizer = torch.optim.Adam([model.W, model.b], 0.01) 

for epoch in range(4000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    # similar to:
    # model.W -= model.W.grad * 0.01
    # model.b -= model.b.grad * 0.01

    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
x = torch.arange(torch.min(x_train), torch.max(x_train), 0.1).reshape(-1,1)
print(x.shape)
print(model.f(x).detach().shape)
# x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])  # x = [[1], [6]]]
plt.plot(x, model.f(x).detach(), label='$\\hat y = f(x) = xW+b$')
plt.legend()
plt.show()