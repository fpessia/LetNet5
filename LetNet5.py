import torch
from Conv_layer import Conv_layer
from Activation_layer import Activation_layer
from Avg_Pooling_layer import Avg_Pooling_layer
from Fully_connected_layer import Fully_connected_layer
import sys

def reshape(l_to_reshape):
    output = torch.zeros(120)
    for i in range(120):
        output[i] = l_to_reshape[i][0][0]
    return output


class LetNet5(): 
    def __init__(self,learning_rate): #although layers are parametically described, LetNet5 structure is fixed
        self.C1 = Conv_layer(32,1,6,5,learning_rate)
        self.Act_C1 = Activation_layer(6,28,28)
        self.Avg_polling_1 = Avg_Pooling_layer(6,28,2)
        self.C2 = Conv_layer(14,6,16,5,learning_rate)
        self.Act_C2 = Activation_layer(16,10,10)
        self.Avg_pooling2 = Avg_Pooling_layer(16,10,2)
        self.C3 = Conv_layer(5,16,120,5,learning_rate)
        self.Act_C3 = Activation_layer(120,1,1)
        self.Fully1 = Fully_connected_layer(120,84,learning_rate)
        self.Act_Fully1 = Activation_layer(84,1,1)
        self.Fully2 = Fully_connected_layer(84,10,learning_rate)

    def forward(self,x):
        l1 = self.C1.forward(x)
        l2 = self.Act_C1.forward(l1)
        l3 = self.Avg_polling_1.forward(l2)
        l4 = self.C2.forward(l3)
        l5 = self.Act_C2.forward(l4)
        l6 = self.Avg_pooling2.forward(l5)
        l7 = self.C3.forward(l6)
        l_to_reshape = self.Act_C3.forward(l7)
        l8 = reshape(l_to_reshape)
        l9 = self.Fully1.forward(l8)
        l10 = self.Act_Fully1.forward(l9)
        l11 = self.Fully2.forward(l10)
        return l11
    
    def backward(self,loss):
        b0 = self.Fully2.backward(loss)
        b1 = self.Act_Fully1.backward(b0)
        b2 = self.Fully1.backward(b1)
        b3 = self.Act_C3.backward(b2)
        b4 = self.C3.backward(b3)
        b5 = self.Avg_pooling2.backward(b4)
        b6 = self.Act_C2.backward(b5)
        b7 = self.C2.backward(b6)
        b8 = self.Avg_polling_1.backward(b7)
        b9 = self.Act_C1.backward(b8)
        b10 = self.C1.backward(b9)





    