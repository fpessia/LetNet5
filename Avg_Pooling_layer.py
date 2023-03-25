import torch


class Avg_Pooling_layer():
    def __init__(self,n_channel,input_size,pooling_size):
        self.n_channel = n_channel
        self.input_size = input_size
        self.pooling_size = pooling_size
        mod = self.input_size % self.pooling_size
        if mod != 0:
            print("Entered unsiutable pooling size")
        else :
            self.output_size = self.input_size/ self.pooling_size
            
        
    def forward(self,x):
        y = torch.zeros(self.n_channel,self.output_size,self.output_size)
        for c in range(self.n_channel):
            for k in range(self.output_size):
                for l in range(self.output_size):
                    for i in range(self.pooling_size):
                        for j in range(self.pooling_size):
                            y[c][k][l] += x[c][k*self.pooling_size + i][ l * self.pooling_size + j]
                        

                    y[c][k][l] = y[c][k][l] / (self.pooling_size * self.pooling_size)
        return y
    
    def backward(self,dy):
        dx = torch.zeros(self.n_channel,self.input_size,self.input_size)
        for c in range(self.n_channel):
            for k in range(self.output_size):
                for l in range(self.output_size):
                    for i in range(self.pooling_size):
                        for j in range(self.pooling_size):
                            dx[c][k*self.pooling_size + i][l *self.pooling_size + j] = dy[c][k][l] / (self.pooling_size * self.pooling_size)
        return dx
    



