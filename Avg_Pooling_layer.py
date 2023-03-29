import torch
from pathos.multiprocessing import ProcessingPool as Pool

class Avg_Pooling_layer():
    def __init__(self,n_channel,input_size,pooling_size):
        self.n_channel = n_channel
        self.input_size = input_size
        self.pooling_size = pooling_size
        mod = self.input_size % self.pooling_size
        if mod != 0:
            print("Entered unsiutable pooling size")
        else :
            self.output_size = int(self.input_size/ self.pooling_size)
            self.y = torch.zeros(self.n_channel,self.output_size,self.output_size)
            self.last_input = torch.zeros(self.n_channel, self.input_size, self.input_size)
            self.map = Pool().map
            
        
    def forward(self,x):
        # I'm going to use multithreading on the different channels
        self.last_input = x
        self.map(self.channel_forward, range(self.n_channel))
        return self.y
    
    def backward(self,dy):
        dx = torch.zeros(self.n_channel,self.input_size,self.input_size)
        for c in range(self.n_channel):
            for k in range(self.output_size):
                for l in range(self.output_size):
                    for i in range(self.pooling_size):
                        for j in range(self.pooling_size):
                            dx[c][k*self.pooling_size + i][l *self.pooling_size + j] = dy[c][k][l] / (self.pooling_size * self.pooling_size)
        return dx
    
    def channel_forward(self,c):
        for k in range(self.output_size):
            for l in range(self.output_size):
                for i in range(self.pooling_size):
                    for j in range(self.pooling_size):
                        self.y[c][k][l] += self.last_input[c][k*self.pooling_size + i][ l * self.pooling_size + j]
                        

                self.y[c][k][l] = self.y[c][k][l] / (self.pooling_size * self.pooling_size)


