from Conv_layer import Conv_layer
from Activation_layer import Activation_layer
from Avg_Pooling_layer import Avg_Pooling_layer
from Fully_connected_layer import Fully_connected_layer


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
        l1 = self.C1(x)
        l2 = self.Act_C1(l1)
        l3 = self.Avg_polling_1(l2)
        l4 = self.C2(l3)
        l5 = self.Act_C2(l4)
        l6 = self.Avg_pooling2(l5)
        l7 = self.C3(l6)
        l8 = self.Act_C3(l7)
        l9 = self.Fully1(l8)
        l10 = self.Act_Fully1(l9)
        l11 = self.Fully2(l10)
        return l11
    
    def backward(self,loss):
        


    




    