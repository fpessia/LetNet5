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
    dl_dy = torch.zeros(N)
    for i in range(N):
        dl_dy[i] = (2/N)*(y_tilde[i]-y[i])
    return dl_dy

def loss (y_tilde,y):
    return ((y_tilde - y)**2).mean()
    

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
            print(b)
            immage_padded_0 = padding(immage[b])
            outpu = CNN.forward(immage_padded_0)

            output= outpu.clone().detach().requires_grad_ (True)
            y_softmax = torch.softmax(output, dim=0)
          
            
            real_label = torch.zeros(10)
            real_label[label[b].item()] = 1.0
            dL_dy = loss_calculation(y_softmax,real_label)

            
            y_softmax.backward(dL_dy)

    
            CNN.backward(output.grad)

            #y_softmax.grad.zero_()
            output.grad.zero_()
    if epoch % 1 == 0:
        l = loss(y_softmax,real_label)
        print (f'Epoch [{epoch+1}/{n_epochs}],  Loss: {l.item():.4f}')

#Now I calculate model accuracy:
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        for b in range(batch_size):
            immage_padded_0 = padding(images[b])
            o = CNN.forward(immage_padded_0)
            y_softmax = torch.softmax(o)

            # max returns (value ,index)
            _,predicted = torch.max(y_softmax, 1)
            predicted += 1
            n_samples += 1
            n_correct += (predicted == labels[b]).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')

            























