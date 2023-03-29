import torch


class Activation_layer():
    def __init__(self,dim_x,dim_y,dim_z):
        self.x = torch.randn(dim_x,dim_y,dim_z, requires_grad = True)
        self.y = torch.randn(dim_x,dim_y,dim_z, requires_grad = True)

    
    def forward(self,x):
         
        self.x = x.clone().detach().requires_grad_(True)
        self.y = torch.tanh(self.x)
        out = self.y.clone().detach().requires_grad_(False)
        return out

    def backward(self,dl_dy):
        self.y.backward(dl_dy)
        output = self.x.grad
        self.x.grad.zero_()
       # self.y.grad.zero_()
        return output

        