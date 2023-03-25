import torch


x = torch.empty(2,2)
print(x.size())

x[1][1] = 1
for i in range(2):
    for j in range(2):
        print(x[i][j])
