import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.colors as colors

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 784 # 28x28
hidden_size = 500 
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# Import MNIST training -and testing dataset
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transforms.ToTensor())


# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# nn.Module = Base class for all neural network modules. Models should subclass this class

class MnistClassificationModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(MnistClassificationModel, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU() #replaces all the negative elements in the input tensor with 0 (zero), and all the non-negative elements are left unchanged.
        self.l2 = nn.Linear(hidden_size, num_classes)  
    
    # The forward function computes output Tensors from input Tensors. 
    # The backward function receives the gradient of the output Tensors with respect to some scalar value, 
    # and computes the gradient of the input Tensors with respect to that same scalar value.
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out

model = MnistClassificationModel(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        print("the labels look like this: ", labels)
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')


# Test the model -> no need to compute gradients 
list_of_images = []
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        list_of_images = outputs

        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    # Accuracy calculation
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')


fig = plt.figure("MNIST")
cmap = colors.LinearSegmentedColormap.from_list('test', [(1,0,0), (0,0,0), (0,0,1)], 10)

for i in range(10):     
    fig.add_subplot(2,5,i+1)
    # image = model.W[:,i].reshape(28,28).detach()
    image = list_of_images[i]
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.title(i)
    plt.imsave('Exercise2\image_results' + str(i) + '.png', image.reshape(28,28).detach(), cmap = cmap)

plt.show()

