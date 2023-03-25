import torch
import numpy as np


x1 = [[2,2],[2,2],[2,2]]
#print(x1.size())

x = np.ones_like(x1)
x = np.pad(x,((0,),(1,),(1,)),'constant')

print(x)
for i in range(4):
    for j in range(4):
        print(x[i][j])


        #  for f in range(self.number_of_filters):   
      #      for i in range(self.input_size+2): 
       #         for j in range(self.input_size+2):
       #             for k in range(self.filter_size): 
      #                 for l in range(self.filter_size):
        #                    for c in range(self.n_channels): 
          #                      dx[c,i,j] += dy_0[f, i+k, j+l] * w_transposed[f, c, k, l]
        


 # I need zero padding of dy & manually transpose w
        dy_0 = torch.zeros(self.number_of_filters,self.output_size+2, self.output_size+2)
        for f in range(self.number_of_filters):
            for w in  range(self.output_size):
                for h in range(self.output_size):
                    dy_0[f][w+1][h+1] = dy[f][w][h] 
     
        #transposing w
        w_transposed = torch.zeros(self.number_of_filters,self.n_channels,self.filter_size,self.filter_size)
        for i in range(self.filter_size):
            for j in range(self.filter_size):
                w_transposed[:,:,i,j] = self.w[:,:,self.filter_size-i-1,self.filter_size-j-1]