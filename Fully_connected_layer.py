import torch
from pathos.multiprocessing import ProcessingPool as Pool

class Fully_connected_layer():
    def __init__(self,input_size,output_size,learning_rate):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.w = torch.randn(self.output_size,self.input_size)
        self.b = torch.randn(1,self.output_size)
        self.last_input = torch.zeros(input_size)
        self.y = torch.zeros(self.output_size)
        self.map = Pool().map
        self.dw = torch.zeros(self.output_size,self.input_size)
        self.db = torch.zeros(1,self.output_size)
        self.dx = torch.zeros(self.input_size)
        self.dy = torch.zeros(self.output_size)
        

    def forward(self, x):
        # I'm multiprocessing on the number of neurons
        self.last_input = x
        
        neuron_list = self.map(self.neuron_forward, range(self.output_size))
        for n in range(self.output_size):
            self.y[n] = neuron_list[n]
    
        return self.y
    
    def backward(self, dy):

        with torch.no_grad(): #dy is coming from softmax in outmost layer might cause gradient over calculation
            self.dy = dy
            #clear last gradient
            self.dw = torch.zeros(self.output_size,self.input_size)
            self.db = torch.zeros(1,self.output_size)
            self.dx = torch.zeros(self.input_size)
            #I calculate first dw using multiprocess
            self.map(self.dw_backward, range(self.output_size))
    
            #then db
            for n in range(self.output_size):
                self.db[0][n] = dy[n]

            #and to conclude dx multiprocessing over input size
            self.map(self.dx_backward, range(self.input_size))
            
                
        
            #Now I update weigth and biases multiprocessing over neurons
            self.map(self.updating_weigths_bias, range(self.output_size))

        return self.dx
    

    def neuron_forward(self, n):
        neuron = 0
        for w_i in range(self.input_size):
            neuron += self.w[n][w_i] * self.last_input[w_i]
        neuron += self.b[0][n]

        return neuron



    def dw_backward(self,n):
         for i in range(self.input_size):
            self.dw[n][i] = self.dy[n] * self.last_input[i]
    def dx_backward(self, i):
        for n in range(self.output_size):
            self.dx[i] += self.dy[n] * self.w[n][i]

    def updating_weigths_bias(self, n):
        self.b[0][n] -= self.learning_rate * self.db[0][n]
        for i in range(self.input_size):
            self.w[n][i] -= self.learning_rate * self.dw[n][i]


        











