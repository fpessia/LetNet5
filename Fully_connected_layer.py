import torch

class Fully_connected_layer():
    def __init__(self,input_size,output_size,learning_rate):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.w = torch.randn(self.output_size,self.input_size)
        self.b = torch.randn(1,self.output_size)
        self.last_input = torch.zeros(1,input_size)

    def forward(self, x):
        y = torch.zeros(1, self.output_size)
        for n in range(self.output_size): #for each neuron
            for w_i in range(self.input_size):
                y[n] += self.w[n][w_i] * x[w_i]

            y[n] += self.b[1,n]
        self.last_input = x
        return y
    
    def backward(self, dy):

        dw = torch.zeros(self.output_size,self.input_size)
        db = torch.zeros(1,self.output_size)
        dx = torch.zeros(self.input_size)
        #I calculate first dw
        for n in range(self.output_size):
            for i in range(self.input_size):
                dw[n][i] = dy[n] * self.last_input[i]
        
        #then db
        for n in range(self.output_size):
            db[1][n] = dy[n]

        #and to conclude dx
        for i in range(self.input_size):
            for n in range(self.output_size):
                dx[i] += dy[n] * self.w[n][i]
        
        #Now I update weigth and biases

        for n in range(self.output_size):
            self.b[n] -= self.learning_rate * db[n]
            for i in range(self.input_size):
                self.w[n][i] -= self.learning_rate * dw[n][i]

        return dx
        
        











