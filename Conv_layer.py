import torch
from pathos.multiprocessing import ProcessingPool as Pool

class Conv_layer():
    def __init__(self,input_size,n_channels,number_of_filters,filter_size, learning_rate):
        self.input_size = input_size
        self.n_channels = n_channels
        self.number_of_filters = number_of_filters
        self.filter_size = filter_size
        self.learning_rate = learning_rate
        self.w = torch.randn(number_of_filters,n_channels,filter_size,filter_size)
        self.b = torch.randn(1,number_of_filters)
        self.output_size = (input_size- filter_size) + 1
        self.last_input = torch.empty(n_channels,input_size,input_size)
        self.map = Pool().map
        self.y = torch.zeros(self.number_of_filters,self.output_size,self.output_size)
        self.dy = torch.zeros(self.filter_size, self.output_size, self.output_size)
        self.db = torch.zeros(1,self.number_of_filters)
        self.dw = torch.zeros(self.number_of_filters,self.n_channels,self.filter_size,self.filter_size)
        self.dx = torch.zeros(self.n_channels,self.input_size,self.input_size)


    def forward(self,x):
        # I'm going to use multiprocess over the number of filters in order to calculate the convolution faster
        self.last_input = x
        self.map(self.figure_forward, range(self.number_of_filters))    
        return self.y
    
    def backward(self,dy):
        # As a fisrt step I calculate db, dw, dx
        #then I update w & b

        #clear old gradient & save new dy
        self.db = torch.zeros(1,self.number_of_filters)
        self.dw = torch.zeros(self.number_of_filters,self.n_channels,self.filter_size,self.filter_size)
        self.dx = torch.zeros(self.n_channels,self.input_size,self.input_size)


        #caulculate db multiprocessing over number of filters
        self.map(self.db_backward, range(self.number_of_filters))
        
        
        #Now i proced to find dw still multiprocessing over filters
        self.map(self.dw_backward, range(self.number_of_filters))
       
         
       #and finally to find dx this time multiprocessing over n_channels
        self.map(self.dx_backward, range(self.n_channels))
        
        #Now I update w & b according to learinig rate multiprocessing over filters
        self.map(self.updating_weigths_and_bias, range(self.number_of_filters))
        
        return self.dx

    def figure_forward(self,f):
        for i in range(self.output_size):
                    for j in range(self.output_size):
                        for c in range(self.n_channels):
                            for k in range(self.filter_size):
                                for l in range(self.filter_size):
                                    self.y[f][i][j] += self.w[f][c][k][l] * self.last_input[c][i+k][j+l]
                        self.y[f][i][j] += self.b[0][f]

    def db_backward(self, f):
       for i in range(self.output_size):
            for j in range(self.output_size):
                self.db[0,f] += self.dy[f][i][j] 

    def dw_backward(self, f):
        for c in range(self.n_channels):
            for k in range(self.filter_size):
                for l in range(self.filter_size):
                    for i in range(self.output_size):
                        for j in range(self.output_size):
                            #for c_n in range(self.n_channels):
                            self.dw[f][c][k][l] += self.dy[f][i][j] * self.last_input[c][i+k][j+l]


    def dx_backward(self, c):
        for k in range(self.input_size):
            for l in range(self.input_size):
                for f in range(self.number_of_filters):
                    for i in range(self.output_size):
                        for j in range(self.output_size):
                            u = k - i
                            v = l - j
                            if u >= 0 and v >= 0 and u < self.filter_size and v < self.filter_size:
                                self.dx[c][k][l] = self.dy[f][i][j] *self.w[f][c][u][v]

    def updating_weigths_and_bias(self, f):
        self.b[0][f] -= self.learning_rate * self.db[0][f]
        for c in range(self.n_channels):
            for i in range(self.filter_size):
                 for j in range(self.filter_size):
                    self.w[f][c][i][j] -= self.learning_rate * self.dw[f][c][i][j]
