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
        

       
