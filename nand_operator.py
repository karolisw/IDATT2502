import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd 

and_not_values = pd.read_csv("training/nand_training.csv", usecols=[0,1]) 
output_values = pd.read_csv("training/nand_training.csv", usecols=[2]) 

# Putting and operator and not operator in the same tensor
x_train = torch.Tensor(and_not_values.values).reshape(-1, 2) # unspecified amount of rows and 1 column
y_train = torch.Tensor(output_values.values).reshape(-1, 1)

# Correct answer: w1 = -2, w2 = 2, b = -3
class NandOperatorModel:

    def __init__(self):
    
        self.W1 = torch.tensor([[0.0, 0.0], [0.0, 0.0]]).reshape(2,-1).requires_grad_() 
        self.W2 = torch.tensor([[0.0], [0.0]]).reshape(2,-1).requires_grad_() 
        self.b1 = torch.tensor([[0.0, 0.0]], requires_grad=True) 
        self.b2 = torch.tensor([[0.0]], requires_grad=True) 

    # Predictor that handles two layers f1 and f2
    def f(self, x):
        return self.f2(self.f1(x))
    
    # First layer
    def f1(self, x):
        print(x.shape)
        return torch.sigmoid(x @ self.W1 + self.b1) # number of rows in x == number of columns in y

    # Second layer --> h is the output of f1(x)
    def f2(self, h): 
        return torch.sigmoid(h @ self.W2 + self.b2)

    # Uses cross entropy with softmax to find the loss due classes being dependent
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy(self.f(x), y) 

    # for use with test dataset
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())

model = NandOperatorModel()

# Optimize: adjust W and b to minimize loss
optimizer = torch.optim.Adam([model.W1, model.b1, model.W2, model.b2], 0.01) 

for epoch in range(4000):
    model.loss(x_train, y_train).backward()  
    optimizer.step()  

    optimizer.zero_grad()  

print("W1 = %s, b1 = %s, W2 = %s, b2 = %s, loss = %s" % (model.W1, model.b1, model.W2, model.b2, model.loss(x_train, y_train)))

X, Y, Z = x_train[:, 0:1], x_train[:, 1:2], y_train

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plt.xlabel('x')
plt.ylabel('y')

ax.scatter3D(X, Y, Z, c=Z, cmap="plasma")

xs = torch.linspace(0, 1, steps=100)
ys = torch.linspace(0, 1, steps=100)
x, y = torch.meshgrid(xs, ys, indexing='xy')
z = model.f(torch.cat(tuple(torch.dstack([x, y]))))
asd = z.reshape(100, 100)
ax.plot_surface(x.numpy(), y.numpy(), asd.detach().numpy(), alpha=0.5)

plt.show()