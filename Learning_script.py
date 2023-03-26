from LetNet5 import LetNet5
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys


def padding(input_tensor):
    immage_padded = torch.zeros(1,32,32)
    for i in range(2, 30):
        for j in range (2, 30):
            immage_padded[0][i][j] = input_tensor[0][i-2][j-2]
    
    return immage_padded


def loss_calculation(y_tilde, y):
    N = len(y_tilde)
    dl_dy = torch.zeros(1,N)
    for i in range(N):
        dl_dy[i] = (2/N)*(y_tilde[i]-y)
    return dl_dy
    

batch_size = 100
n_epochs = 10
learning_rate = 0.01

# MNIST dataset 
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


CNN = LetNet5(learning_rate)

for epoch in range(n_epochs):
    for i, (immage, label) in enumerate(train_loader):
        for b in range(batch_size):

            immage_padded_0 = padding(immage[b])
            output = CNN.forward(immage_padded_0)

            output.requires_grad = True
            y_softmax = torch.softmax(output, dim=0)
            y_softmax.requires_grad = True
            
            real_label = torch.zeros(1,10)
            real_label[label[b].item()] = 1.0
            dL_dy = loss_calculation(y_softmax,real_label)

            y_softmax.backward(dL_dy)
            CNN.backward(output.grad)

            y_softmax.grad.zero_()
            output.grad.zero_()

            























