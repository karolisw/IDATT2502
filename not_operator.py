import torch
import matplotlib.pyplot as plt
import pandas as pd # for reading csv


not_values = pd.read_csv("training/not_training.csv", usecols=[0]) # contains 0 and 1
output_values = pd.read_csv("training/not_training.csv", usecols=[1]) # contains 1 and 0

# Putting length -and weight values into their own tensors
x_train = torch.Tensor(not_values.values).reshape(-1, 1) 
y_train = torch.Tensor(output_values.values).reshape(-1, 1)

class NotOperatorModel:

    def __init__(self):
    
        # Model variables that enable calculation of gradients
        self.W = torch.tensor([[0.0]], requires_grad=True) 
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return torch.sigmoid(self.logits(x))

    def logits(self, x):
        return x @ self.W + self.b

    # Uses cross entropy to find the loss due to classification
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y) 
        # return torch.nn.functional.mse_loss(self.f(x), y)

    # for use with test dataset
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())

model = NotOperatorModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
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
x = torch.linspace(torch.min(x_train), torch.max(x_train), 100).reshape(100, 1)
plt.plot(x, model.f(x).detach(), label='$\hat y = f(x) = = σ(xW + b)$')
plt.legend()
plt.show()