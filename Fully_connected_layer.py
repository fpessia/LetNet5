import torch
from pathos.multiprocessing import ProcessingPool as Pool
from math import sqrt

class Fully_connected_layer():
    def __init__(self,input_size,output_size,learning_rate):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.w = torch.randn(self.output_size,self.input_size)* sqrt(2/(self.input_size + self.output_size))
        self.b = torch.zeros(1,self.output_size)
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
           
            #I calculate first dw using multiprocess
            w_grad_list = self.map(self.dw_backward, range(self.output_size))

            for n in range(self.output_size):
                self.dw[n] = w_grad_list[n]
    
            #then db
            for n in range(self.output_size):
                self.db[0][n] = dy[n]

            #and to conclude dx multiprocessing over input size
            input_grad_list = self.map(self.dx_backward, range(self.input_size))

            for i in range(self.input_size):
                self.dx[i] = input_grad_list[i]
            
                
        
            #Now I update weigth and biases multiprocessing over neurons
            updated_w_list = self.map(self.updating_weigths, range(self.output_size))
            
            for n in range(self.output_size):
                self.w[n] = updated_w_list[n]
                self.b[0][n] -= self.learning_rate * self.db[0][n]

        
        return self.dx
    

    def neuron_forward(self, n):
        neuron = 0
        for w_i in range(self.input_size):
            neuron += self.w[n][w_i] * self.last_input[w_i]
        neuron += self.b[0][n]

        return neuron



    def dw_backward(self,n):
        w_grad = torch.zeros(self.input_size)
        for i in range(self.input_size):
            w_grad[i] = self.dy[n] * self.last_input[i]
        return w_grad
    
    def dx_backward(self, i):
        input_grad = 0
        for n in range(self.output_size):
            input_grad += self.dy[n] * self.w[n][i]
        return input_grad

    def updating_weigths(self, n):
        updated_w = self.w[n]
        for i in range(self.input_size):
            updated_w[i] -= self.learning_rate * self.dw[n][i]
        return updated_w
    
    def W_and_biases_write(self):
        file = open("C:/Users/fpess/OneDrive/Desktop/Magistrale/TESI/Pytorch/LetNet5/W_and_biases_4k_immages.txt", mode="a")
        file.write("\n")
        for  n in range(self.output_size):
            for i in range(self.input_size):
                file.write(str(self.w[n][i].item())+ "\t")
        file.write("\n \n")
        for n in range(self.output_size):
            file.write(str(self.b[0][n].item()) + "\t")
        file.write("\n")
        file.close()

    def W_and_biases_read(self, file):
        
        line = file.readline()
        line = file.readline()
        float_list = line.split("\t")
        for n in range(self.output_size):
            for i in range(self.input_size):
                self.w[n][i] = float(float_list[n * self.input_size + i])
        
        line = file.readline()#reading \n
        line = file.readline()
        float_list = line.split("\t")
        for n in range(self.output_size):
            self.b[0][n] = float(float_list[n])
        











