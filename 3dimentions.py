import matplotlib.pyplot as plt
import pandas as pd 
import torch

day_column = pd.read_csv("day_length_weight.csv", usecols=[0])
length_column = pd.read_csv("day_length_weight.csv", usecols=[1])
weight_column = pd.read_csv("day_length_weight.csv", usecols=[2])

x_train = torch.Tensor(length_column.values).reshape(-1, 2) 
y_train = torch.Tensor(weight_column.values).reshape(-1, 2) 
z_train = torch.Tensor(day_column.values).reshape(-1, 2) 

class LinearRegressionModel:

    def __init__(self):
    
        # Weights can now hold both weight and length
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x): # TODO x is the long one
        # If x and W have the same dimentions, x must be altered
        if(x.size() == self.W.size()): # ex:  (2,1) x (2,1) must be turned into --> (1,2) x (2,1)
            print("\nx and w had the same size\n")
            # as_list = list(self.W.size()) 
            as_list = list(x.size()) 
            # nr_columns = as_list[-1]
            nr_rows = as_list[0]
            self.W = torch.reshape(self.W, (-1, nr_rows))
            print("after reshaping x has the size: ", x.size())
            print("after reshaping w has the size: ", self.W.size())

        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)
        # return torch.mean(torch.square(self.f(x) - y))  # Can also use torch.nn.functional.mse_loss(self.f(x), y) to possibly increase numberical stability


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.Adam([model.W, model.b], 0.01) 

for epoch in range(500):
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
x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])  # x = [[1], [6]]]
print("x size is: ", x.size())
plt.plot(x, model.f(x).detach(), label='$\\hat y = f(x) = xW+b$')
plt.legend()