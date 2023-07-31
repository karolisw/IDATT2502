# Based on the nn file 
import torch
import torch.nn as nn
import torchvision

# Making the code device-agnostic
device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.FashionMNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 1, 28, 28).float() # torch.functional.nn.conv2d argument must include channels (1)
y_train = torch.zeros((mnist_train.targets.shape[0], 10)) # Create output tensor 
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.FashionMNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 1, 28, 28).float()  # torch.functional.nn.conv2d argument must include channels (1)
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

# Normalization of inputs
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Putting x_train and y_train onto the GPU
x_train = x_train.to(device)
y_train = y_train.to(device)

# Putting x_test and y_test onto the GPU
x_test = x_test.to(device)
y_test = y_test.to(device)

# Divide training data into batches to speed up optimization
batches = 600
x_train_batches = torch.split(x_train, batches)
y_train_batches = torch.split(y_train, batches)


class ConvolutionalNeuralNetworkModel(nn.Module):

    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()

        # Model layers (includes initialized model variables):
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2) # Param: (in_channel, out_channel, kernel_size, padding)
        self.relu1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)   
        self.relu2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2) 
        self.dense1 = nn.Linear(64 * 7 * 7, 1 * 1024) 
        self.dense2 = nn.Linear(1 * 1024, 10)
        self.dropout = nn.Dropout(0.25)


    def logits(self, x):
        x = self.conv1(x)
        x = self.relu1(x) # TODO try to use false as input
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dense1(x.reshape(-1, 64 * 7 * 7))
        x = self.dropout(x)
        return self.dense2(x.reshape(-1, 1 * 1024))

    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).to(torch.float))
        


model = ConvolutionalNeuralNetworkModel()

# Putting the model onto the GPU
model = model.to(device)

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.Adam(model.parameters(), 0.001)
counter = 0
for epoch in range(10000):
    counter += 1
    for batch in range(len(x_train_batches)):
        if(model.accuracy(x_test, y_test > 0.93)):
            print("accuracy = %s" % model.accuracy(x_test, y_test))
            print("Stopped at epoch number: ", counter)
            break

        model.loss(x_train_batches[batch], y_train_batches[batch]).backward()  # Compute loss gradients
        optimizer.step()  # Perform optimization by adjusting W and b,
        optimizer.zero_grad()  # Clear gradients for next step

    #print("accuracy = %s" % model.accuracy(x_test, y_test))
