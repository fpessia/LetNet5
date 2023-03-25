import torch


class Activation_layer():
    def __init__(self,dim_x,dim_y,dim_z):
        self.x = torch.randn(dim_x,dim_y,dim_z, requires_grad = True)
        self.y = torch.randn(dim_x,dim_y,dim_z, requires_grad = True)
    
    def forward(self,x):
        self.x = x
        self.y = torch.tanh(x)
        return self.y

    def backward(self,dl_dy):
        self.y.backward(dl_dy)
        return self.x.grad

        