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

    def forward(self, x):
        # I'm multiprocessing on the number of neurons
        self.last_input = x
        self.map(self.neuron_forward, range(self.output_size))
    
        return self.y
    
    def backward(self, dy):

        with torch.no_grad(): #dy is coming from softmax in outmost layer might cause gradient over calculation
            dw = torch.zeros(self.output_size,self.input_size)
            db = torch.zeros(1,self.output_size)
            dx = torch.zeros(self.input_size)
        #I calculate first dw
            for n in range(self.output_size):
                for i in range(self.input_size):
                    dw[n][i] = dy[n] * self.last_input[i]
        
        #then db
            for n in range(self.output_size):
                db[0][n] = dy[n]

        #and to conclude dx
            for i in range(self.input_size):
                for n in range(self.output_size):
                    dx[i] += dy[n] * self.w[n][i]
        
        #Now I update weigth and biases

            for n in range(self.output_size):
                self.b[0][n] -= self.learning_rate * db[0][n]
                for i in range(self.input_size):
                    self.w[n][i] -= self.learning_rate * dw[n][i]

        return dx
    

    def neuron_forward(self, n):
        for w_i in range(self.input_size):
            self.y[n] += self.w[n][w_i] * self.last_input[w_i]

        self.y[n] += self.b[0][n]

        
        











