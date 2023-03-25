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
        for f in range(self.number_of_filters):
            for i in range(self.output_size):
                    for j in range(self.output_size):
                        y[f][i][j] = 0
                        for c in range(self.n_channels):
                            for k in range(self.filter_size):
                                for l in range(self.filter_size):
                                    y[f][i][j] += self.w[f][c][k][l] * x[c][i+k][j+l]
                        y[f][i][j] += self.b[1][f]
        self.last_input = x
        return y
    
    def back_prop(self,dy):
        # As a fisrt step I calculate db, dw, dx
        #then I update w & b
        db = torch.empty(1,self.number_of_filters)
        dw = torch.empty(self.number_of_filters,self.n_channels,self.filter_size,self.filter_size)
        dx = torch.zeros(self.n_channels,self.input_size+2,self.input_size+2) #already zero padded


        for f in range(self.number_of_filters):
            db[f]= 0
            for i in range(self.output_size):
                for j in range(self.output_size):
                    db[1,f] += dy[f][i][j]
       
        #Now i proced to find dw
        for f in range(self.number_of_filters):
            for c in range(self.n_channels):
                for k in range(self.filter_size):
                    for l in range(self.filter_size):
                        dw[f][c][k][l] = 0 
                        for i in range(self.output_size):
                            for j in range(self.output_size):
                                #for c_n in range(self.n_channels):
                                dw[f][c][k][l] += dy[f][i][j] * self.last_input[c][i+k][j+l]
        
        #Now I procede to find dx = dy_0 * w'
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

            
        for f in range(self.number_of_filters):   
            for i in range(self.input_size+2): 
                for j in range(self.input_size+2):
                    for k in range(self.filter_size): 
                        for l in range(self.filter_size):
                            for c in range(self.n_channels): 
                                dx[c,i,j] += dy_0[f, i+k, j+l] * w_transposed[f, c, k, l]
        
    
        #Now I update w & b according to learinig rate
        for f in range(self.number_of_filters):
            self.b[1][f] -= self.learning_rate * db[1][f]
        
        for f in range(self.number_of_filters):
            for c in range(self.n_channels):
                for i in range(self.filter_size):
                    for j in range(self.filter_size):
                        self.w[f][c][i][j] -= self.learning_rate * dw[f][c][i][j]
    
        
        return dx[:,1:-1,1:-1]

