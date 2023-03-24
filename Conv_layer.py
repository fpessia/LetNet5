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

    def forward(x):
        for f in range(self.number_of_filters):
            for i in range(self.output_size):
                    for j in range(self.output_size):
                        y[f][i][j] = 0
                        for c in range(self.n_channels):
                            for k in range(self.filter_size):
                                for l in range(self.output_size):
                                    y[f][i][j] += self.w[f][c][k][l] * x[c][i+k][j+l]
                        y[f][i][j] += self.b[1][f]
        return y
    
    def back_prop(dy):
        # As a fisrt step I calculate db, dw, dx
        #then I update w & b
        for f in range(number_of_filters):
            db[f]= 0
            for i in range(output_size):
                for j in range(output_size):
                    db[f] += dy[f][i][j]
        #Now i proced to find dw

        for f in range(number_of_filters):
            for c in range(n_channels):
                for k in range(output_size):
                    for l in range(output_size):
                        dw[f][c][k][l] = 0 
                        for i in range(output_size):
                            for j in range(output_size):
                                for c_n in range(n_channels):
                                    dw[f][c][k][l] += 
        
    

    



