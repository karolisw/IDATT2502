import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.colors as colors

num_classes = 10
batch_size = 100
learning_rate = 0.001

# Load observations from the mnist dataset into the training 
# Train: If True, creates dataset from train-images-idx3-ubyte, otherwise from t10k-images-idx3-ubyte 
# Download: If True, downloads the dataset from the internet and puts it in root directory. 
#           If dataset is already downloaded, it is not downloaded again.
mnist_data = torchvision.datasets.MNIST('./data', train=True, download=True)

# 28 x 28 = 784 
x_train = mnist_data.data.reshape(-1, 784).float()

# Create output tensor 
# Zeroes: Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size.
# Targets: A list showing what number each image from the dataset is supposed to represent!
#          Because these are the answers, we can use them to calculate our loss
#          The size is as many rows as there are photos, and 10 rows    
y_train = torch.zeros((mnist_data.targets.shape[0], 10)) 
y_train[torch.arange(mnist_data.targets.shape[0]), mnist_data.targets] = 1 # Populating the output

# Downloading the test dataset
mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 784).float() # Reshape input
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output


class NumbersModel:

    def __init__(self):
        # Correct size for matrfiz multiplication
        self.W = torch.zeros(784, 10).requires_grad_()
        self.b = torch.tensor([[0.0]], requires_grad=True) 

    # Softmax: for multi-class classification to normalize the scores for the given classes.
    def f(self, x):
        return torch.nn.functional.softmax(self.f1(x), dim=1) # TODO try to change dim to 0
        # return torch.nn.LogSoftmax(x)
        # return torch.nn.Softmax(self.f1(x)) # With logits == without softmax
    
    def f1(self, x):
        return x @ self.W + self.b # number of rows in x == number of columns in y

    # Uses cross entropy with softmax to find the loss due classes being dependent
    def loss(self, images, labels): 
        cel = torch.nn.CrossEntropyLoss()
        return cel(self.f(images), labels)
        # return torch.nn.CrossEntropyLoss((self.f(x)), y) 

    # for use with test dataset
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())

model = NumbersModel()

# Optimize: adjust W and b to minimize loss
optimizer = torch.optim.Adam([model.W, model.b], 0.01) 

print ("y size: ", y_train.shape)
print ("x size: ", x_train.shape)
print ("W size: ", model.W.shape)

# Training
for epoch in range(200):
    model.loss(x_train, y_train).backward()  
    optimizer.step()
    optimizer.zero_grad()  

    fig = plt.figure("MNIST")
    cmap = colors.LinearSegmentedColormap.from_list('test', [(1,0,0), (0,0,0), (0,0,1)], 10)

    if (model.accuracy(x_test, y_test) > 0.9):
        for i in range(10):     
            fig.add_subplot(2,5,i+1)
            image = model.W[:,i].reshape(28,28).detach()
            plt.imshow(image, cmap=cmap)
            plt.axis('off')
            plt.title(i) 
            plt.imsave('image_results\\' + str(i) + '.png', model.W[:,i].reshape(28,28).detach(), cmap = cmap)
    
# Printing the final accuracy 
print("\nAccuracy: ", model.accuracy(x_test, y_test)) # log/save  # TODO move into training + create plot + save plot

plt.show()
