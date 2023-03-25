import torch

class Fully_connected_layer():
    def __init__(self,input_size,output_size,learning_rate):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.w = torch.randn(1, self.input_size)
        self.b = torch.randn(1,self.output_size)
        self.last_input = torch.zeros(1,input_size)

    def forward(self, x):
        y = torch.zeros(1, self.output_size)
        for n in range(self.output_size): #for each neuron
            for w_i in range(self.input_size):
                y[n] += self.w[w_i] * x[w_i]

            y[n] += self.b[n]
        

