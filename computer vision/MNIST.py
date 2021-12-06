import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from time import time

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,)),])
trainset = datasets.MNIST('C:/Users/andrew/Desktop/mnist_train', download=True, train=True, transform=transform)
valset = datasets.MNIST('C:/Users/andrew/Desktop/mnist_test', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.01
FC1_DIMS = 1024
FC2_DIMS = 512
DEVICE = torch.device("cpu")
INPUT_SIZE = 784
OUTPUT_SIZE = 10

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = INPUT_SIZE
        self.action_space = OUTPUT_SIZE

        self.fc1 = nn.Linear(self.input_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, self.action_space)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.NLLLoss()
        self.to(DEVICE)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        m = nn.LogSoftmax(dim = 1)
        x = m(x)

        return x

network = Network()

for i in range(EPOCHS):
    running_loss = 0

    for images, labels in trainloader:
        network.optimizer.zero_grad()

        images = images.view(images.shape[0], -1)
        outputs = network.forward(images)

        loss = network.loss(outputs, labels)
        loss.backward()
        network.optimizer.step()

        running_loss += loss.item()

    print("Epoch {} - Training loss: {}".format(i, running_loss/len(trainloader)))

correct = 0
rendered_images = []
rendered_labels = []

for images, labels in valset:
    rendered_images.append(images.squeeze())

    network.optimizer.zero_grad()
    images = images.view(images.shape[0], -1)
    outputs = network.forward(images).detach()
    
    ps = torch.exp(outputs)
    probab = list(ps.numpy()[0])
    prediction = probab.index(max(probab))
    rendered_labels.append(prediction)

    if prediction == labels:
        correct += 1

print("model accuracy {}%".format((correct/10000) * 100))

fig = plt.figure()
for i in range(9):
  plt.subplot(3, 3, i+1)
  plt.tight_layout()
  plt.imshow(rendered_images[i], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(rendered_labels[i]))
  plt.xticks([])
  plt.yticks([])
plt.show()
