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
        self.dy = torch.zeros(self.number_of_filters, self.output_size, self.output_size)
        self.db = torch.zeros(1,self.number_of_filters)
        self.dw = torch.zeros(self.number_of_filters,self.n_channels,self.filter_size,self.filter_size)
        self.dx = torch.zeros(self.n_channels,self.input_size,self.input_size)


    def forward(self,x):
        # I'm going to use multiprocess over the number of filters in order to calculate the convolution faster
        self.last_input = x  
        figure_list = self.map(self.figure_forward, range(self.number_of_filters))  
        for f in range(self.number_of_filters):
            self.y[f] = figure_list[f]  
        return self.y
    
    def backward(self,dy):
        # As a fisrt step I calculate db, dw, dx
        #then I update w & b

        self.dy = dy

        #caulculate db multiprocessing over number of filters
        bias_grad_list = self.map(self.db_backward, range(self.number_of_filters))
        for f in range(self.number_of_filters):
            self.db[0,f] = bias_grad_list[f]
        
        
        #Now i proced to find dw still multiprocessing over filters
        w_grad_list = self.map(self.dw_backward, range(self.number_of_filters))

        for f in range(self.number_of_filters):
            self.dw[f] = w_grad_list[f]
       

       #and finally to find dx this time multiprocessing over n_channels
        input_grad_list = self.map(self.dx_backward, range(self.n_channels))

        for c in range(self.n_channels):
            self.dx[c] = input_grad_list[c]
        
        #Now I update w & b according to learinig rate multiprocessing over filters
        updated_w_list = self.map(self.updating_weigths_and_bias, range(self.number_of_filters))

        for f in range(self.number_of_filters):
            self.b[0][f] -= self.learning_rate * self.db[0][f]
            self.w[f] = updated_w_list[f]
        
        return self.dx

    def figure_forward(self,f):
        convoluted_figure = torch.zeros(self.output_size, self.output_size)
        for i in range(self.output_size):
                    for j in range(self.output_size):
                        for c in range(self.n_channels):
                            for k in range(self.filter_size):
                                for l in range(self.filter_size):
                                    convoluted_figure[i][j] += self.w[f][c][k][l] * self.last_input[c][i+k][j+l]
                        convoluted_figure[i][j] += self.b[0][f]
        return convoluted_figure

    def db_backward(self, f):
        bias_grad = 0
        for i in range(self.output_size):
            for j in range(self.output_size):
                bias_grad += self.dy[f][i][j]
        return bias_grad 

    def dw_backward(self, f):
        w_grad = torch.zeros(self.n_channels, self.filter_size, self.filter_size)
        for c in range(self.n_channels):
            for k in range(self.filter_size):
                for l in range(self.filter_size):
                    for i in range(self.output_size):
                        for j in range(self.output_size):
                            #for c_n in range(self.n_channels):
                            w_grad[c][k][l] += self.dy[f][i][j] * self.last_input[c][i+k][j+l]
        return w_grad


    def dx_backward(self, c):
        input_grad = torch.zeros(self.input_size, self.input_size)
        for k in range(self.input_size):
            for l in range(self.input_size):
                for f in range(self.number_of_filters):
                    for i in range(self.output_size):
                        for j in range(self.output_size):
                            u = k - i
                            v = l - j
                            if u >= 0 and v >= 0 and u < self.filter_size and v < self.filter_size:
                                input_grad[k][l] = self.dy[f][i][j] *self.w[f][c][u][v]
        return input_grad

    def updating_weigths_and_bias(self, f):
        updated_w = self.w[f]
        for c in range(self.n_channels):
            for i in range(self.filter_size):
                 for j in range(self.filter_size):
                    updated_w[c][i][j] -= self.learning_rate * self.dw[f][c][i][j]
        return updated_w
    
    def W_and_bias_write(self):
        file = open("C:/Users/fpess/OneDrive/Desktop/Magistrale/TESI/Pytorch/LetNet5/W_and_biases_4k_immages.txt", mode="a")
        file.write("\n \n")
        for f in range(self.number_of_filters):
            for c in range(self.n_channels):
                for i in range(self.filter_size):
                    for j in range(self. filter_size):
                        file.write(str(self.w[f][c][i][j].item()) + "\t")
                file.write("\n")

        file.write("\n \n")
        for f in range(self.number_of_filters):
            file.write(str(self.b[0][f].item())+ "\t")
        file.write("\n")
        file.close()
    
    def W_and_bias_read(self, file):
        line = file.readline()
        line = file.readline() #reading \n
        for f in range(self.number_of_filters):
            for c in range(self.n_channels):
                line = file.readline()
                float_list = line.split("\t")
                for i in range(self.filter_size):
                    for j in range(self.filter_size):
                        self.w[f][c][i][j] = float(float_list[i  * self.filter_size + j])
        line = file.readline()
        line = file.readline() #reading \n
        
        line = file.readline()
        float_list = line.split("\t")
        for f in range(self.number_of_filters):
            self.b[0][f] = float(float_list[f])
            
   

