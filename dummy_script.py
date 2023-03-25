import torch
import numpy as np
import sys


#x = torch.tensor([[2.0,2.0],[2.0,2.0],[2.0,2.0]],requires_grad=True)
#print(x1.size())


#x = torch.randn(3, requires_grad = True)
x = torch.zeros(2,2, requires_grad = True)
print(x)
x = torch.tensor([[0.9, -0.67],[0.7, 0.4]], requires_grad = True)
print(x)
y = torch.tanh(x)
print(y)
dy = torch.tensor([[0.01, 0.0001],[ 0.1, 1.0]])
y.backward(dy)
print(x.grad)


sys.exit()



        #  for f in range(self.number_of_filters):   
      #      for i in range(self.input_size+2): 
       #         for j in range(self.input_size+2):
       #             for k in range(self.filter_size): 
      #                 for l in range(self.filter_size):
        #                    for c in range(self.n_channels): 
          #                      dx[c,i,j] += dy_0[f, i+k, j+l] * w_transposed[f, c, k, l]
        


 # I need zero padding of dy & manually transpose w

 #
  #      dy_0 = torch.zeros(self.number_of_filters,self.output_size+2, self.output_size+2)
   #     for f in range(self.number_of_filters):
   #         for w in  range(self.output_size):
    #            for h in range(self.output_size):
     #               dy_0[f][w+1][h+1] = dy[f][w][h] 
     
        #transposing w
      #  w_transposed = torch.zeros(self.number_of_filters,self.n_channels,self.filter_size,self.filter_size)
       # for i in range(self.filter_size):
        #    for j in range(self.filter_size):
         #       w_transposed[:,:,i,j] = self.w[:,:,self.filter_size-i-1,self.filter_size-j-1]