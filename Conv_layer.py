import torch

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

    def forward(self,x):
        y = torch.zeros(self.number_of_filters,self.output_size,self.output_size)
        for f in range(self.number_of_filters):
            for i in range(self.output_size):
                    for j in range(self.output_size):
                        for c in range(self.n_channels):
                            for k in range(self.filter_size):
                                for l in range(self.filter_size):
                                    y[f][i][j] += self.w[f][c][k][l] * x[c][i+k][j+l]
                        y[f][i][j] += self.b[0][f]
        self.last_input = x
        return y
    
    def backward(self,dy):
        # As a fisrt step I calculate db, dw, dx
        #then I update w & b
        db = torch.zeros(1,self.number_of_filters)
        dw = torch.zeros(self.number_of_filters,self.n_channels,self.filter_size,self.filter_size)
        dx = torch.zeros(self.n_channels,self.input_size,self.input_size)


        for f in range(self.number_of_filters):
            for i in range(self.output_size):
                for j in range(self.output_size):
                    db[0,f] += dy[f][i][j]
       
        #Now i proced to find dw
        for f in range(self.number_of_filters):
            for c in range(self.n_channels):
                for k in range(self.filter_size):
                    for l in range(self.filter_size):
                        for i in range(self.output_size):
                            for j in range(self.output_size):
                                #for c_n in range(self.n_channels):
                                dw[f][c][k][l] += dy[f][i][j] * self.last_input[c][i+k][j+l]
        
       
       #and finally to find dx
        for c in range(self.n_channels):
            for k in range(self.input_size):
                for l in range(self.input_size):
                    for f in range(self.number_of_filters):
                        for i in range(self.output_size):
                            for j in range(self.output_size):
                                u = k - i
                                v = l - j
                                if u >= 0 and v >= 0 and u < self.filter_size and v < self.filter_size:
                                    dx[c][k][l] = dy[f][i][j] *self.w[f][c][u][v]
                               

                                
                                


        

       
        #Now I update w & b according to learinig rate
        for f in range(self.number_of_filters):
            self.b[0][f] -= self.learning_rate * db[0][f]
        
        for f in range(self.number_of_filters):
            for c in range(self.n_channels):
                for i in range(self.filter_size):
                    for j in range(self.filter_size):
                        self.w[f][c][i][j] -= self.learning_rate * dw[f][c][i][j]
    
        
        return dx

