import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from tqdm import tqdm
from matplotlib import cm, colors, colormaps
import torch
import torch.nn as nn


# defining pytorch neural network model
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 20), # 2 because there are 2 inputs x and t. 
            nn.Tanh(),
            nn.Linear(20, 30), # 2 because there are 2 inputs x and t. 
            nn.Tanh(),
            nn.Linear(30, 30), # 2 because there are 2 inputs x and t. 
            nn.Tanh(),
            nn.Linear(30, 20), # 2 because there are 2 inputs x and t. 
            nn.Tanh(),
            nn.Linear(20, 20), # 2 because there are 2 inputs x and t. 
            nn.Tanh(),
            nn.Linear(20, 1)            # 64 nodes from the 2nd layer mapping to the output node 
        )
        
        
    def forward(self, x):
        out = self.net(x)
        return out
    
    
class Net:        
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self._h = 0.5
        self._t = 1/10
        x = torch.arange(-5, 5.25, self._h)
        y = torch.arange(-5, 5.25, self._h)
        t = torch.arange(0, 1.0, self._t)
        
        
        self.X = torch.stack(torch.meshgrid(x, y, t)).reshape(3, -1).T
        ic = torch.stack(torch.meshgrid(x, y, t[0])).reshape(3, -1).T
        y_ic = []
        for i in range(0, len(ic)):
            res = ic[i][0]**2 + ic[i][1]**2
            
            y_ic.append(np.sin(res)*np.sin(res))
        y_ic = torch.tensor(y_ic)
        
        
        self.X_train = torch.cat([ic])
        self.y_train = torch.cat([y_ic])
        
        self.y_train = self.y_train.unsqueeze(1)

        self.X = self.X.to(device) # sending both x and t grid to device
        self.y_train = self.y_train.to(device)
        self.X_train = self.X_train.to(device)
        self.X.requires_grad = True
        
        self.adam = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        self.optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_change=1.0*np.finfo(float).eps,
            tolerance_grad=1e-7,
            line_search_fn='strong_wolfe'
            )
        
        
        self.criterion = torch.nn.MSELoss()
        self.iter = 1
        
    def loss_func(self):
        self.adam.zero_grad()
        self.optimizer.zero_grad()
        y_pred = self.model(self.X_train)
        
        #print(f'y_pred - {y_pred}')
        #print(f'y_train - {self.y_train}')
        loss_data = self.criterion(y_pred, self.y_train)
        
        
        u = self.model(self.X)
        du_dxyt = torch.autograd.grad(
            u,
            self.X,
            grad_outputs=torch.ones_like(u), #gradients wrt each output
            create_graph=True, #graph of the derivative will be constructed allowing to compute higher order derivate products
            retain_graph=True #graph used to compute the gradient will be freed if true
            )[0]
        
        du_dx = du_dxyt[:, 0]
        du_dy = du_dxyt[:, 1]
        du_dt = du_dxyt[:, 2]
        
        du_dxxyytt = torch.autograd.grad(
            du_dxyt,
            self.X,
            grad_outputs=torch.ones_like(du_dxyt), #gradients wrt each output
            create_graph=True, #graph of the derivative will be constructed allowing to compute higher order derivate products
            retain_graph=True #graph used to compute the gradient will be freed if true
            )[0]
        
        du_dxx = du_dxxyytt[:, 0]
        du_dyy = du_dxxyytt[:, 1]
        du_dtt = du_dxxyytt[:, 2]
        v = 1
        loss_pde = self.criterion(du_dt, v**2*(du_dxx + du_dyy))
        
        loss = loss_data + loss_pde
        #loss = loss_pde
        loss.backward()
        
        if self.iter % 100 == 0:
            print(self.iter, loss.item())
            
        self.iter += 1
        return loss
        
    
    def train(self):
        self.model.train()
        for i in range(10000):
            self.adam.step(self.loss_func)
            
        #self.optimizer.step(self.loss_func)
    
    def eval_(self):
        self.model.eval()
    
    
    
if torch.cuda.is_available():
    print(f'GPU is available')
else:
    print(f'GPU not available. Using CPU!!')        
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = NN().to(device)

net = Net(model, device)
#Training the model
net.train()

net.model.eval()

model = net.model
model.eval()





with torch.no_grad():
    x_vals = torch.linspace(-5, 5, 100)
    y_vals = torch.linspace(-5, 5, 100)
    X, Y = torch.meshgrid(x_vals, y_vals)
    t_val = torch.ones_like(X) * 0.0          # specify the time t [0, 1]
    
    input_data = torch.stack([X.flatten(), Y.flatten(), t_val.flatten()], dim=1)
    solution = model(input_data).reshape(X.shape, Y.shape) 
    
    plt.imshow(solution, cmap='jet')
    plt.title("0.0")
    plt.xlabel("X")
    plt.ylabel("y")
    cmap = colormaps['jet']
    norm = colors.Normalize(0, 1)
    cax = plt.axes((0.85, 0.1, 0.075, 0.8))
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=cax)
    plt.show()
    

with torch.no_grad():
    x_vals = torch.linspace(-5, 5, 100)
    y_vals = torch.linspace(-5, 5, 100)
    X, Y = torch.meshgrid(x_vals, y_vals)
    t_val = torch.ones_like(X) * 0.25          # specify the time t [0, 1]
    
    input_data = torch.stack([X.flatten(), Y.flatten(), t_val.flatten()], dim=1)
    solution = model(input_data).reshape(X.shape, Y.shape)
    
    plt.imshow(solution, cmap='jet')
    plt.title("0.25")
    plt.xlabel("X")
    plt.ylabel("y")
    cmap = colormaps['jet']
    norm = colors.Normalize(0, 1)
    cax = plt.axes((0.85, 0.1, 0.075, 0.8))
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=cax)
    plt.show()
    
with torch.no_grad():
    x_vals = torch.linspace(-5, 5, 100)
    y_vals = torch.linspace(-5, 5, 100)
    X, Y = torch.meshgrid(x_vals, y_vals)
    t_val = torch.ones_like(X) * 0.5          # specify the time t [0, 1]
    
    input_data = torch.stack([X.flatten(), Y.flatten(), t_val.flatten()], dim=1)
    solution = model(input_data).reshape(X.shape, Y.shape)
    
    plt.imshow(solution, cmap='jet')
    plt.title("0.5")
    plt.xlabel("X")
    plt.ylabel("y")
    cmap = colormaps['jet']
    norm = colors.Normalize(0, 1)
    cax = plt.axes((0.85, 0.1, 0.075, 0.8))
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=cax)
    plt.show()
    
with torch.no_grad():
    x_vals = torch.linspace(-5, 5, 100)
    y_vals = torch.linspace(-5, 5, 100)
    X, Y = torch.meshgrid(x_vals, y_vals)
    t_val = torch.ones_like(X) * 0.75          # specify the time t [0, 1]
    
    input_data = torch.stack([X.flatten(), Y.flatten(), t_val.flatten()], dim=1)
    solution = model(input_data).reshape(X.shape, Y.shape)
    
    plt.imshow(solution, cmap='jet')
    plt.title("0.75")
    plt.xlabel("X")
    plt.ylabel("y")
    cmap = colormaps['jet']
    norm = colors.Normalize(0, 1)
    cax = plt.axes((0.85, 0.1, 0.075, 0.8))
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=cax)
    plt.show()
    

with torch.no_grad():
    x_vals = torch.linspace(-5, 5, 100)
    y_vals = torch.linspace(-5, 5, 100)
    X, Y = torch.meshgrid(x_vals, y_vals)
    t_val = torch.ones_like(X) * 1          # specify the time t [0, 1]
    
    input_data = torch.stack([X.flatten(), Y.flatten(), t_val.flatten()], dim=1)
    solution = model(input_data).reshape(X.shape, Y.shape)
    
    plt.imshow(solution, cmap='jet')
    plt.title("1.0")
    plt.xlabel("X")
    plt.ylabel("y")
    cmap = colormaps['jet']
    norm = colors.Normalize(0, 1)
    cax = plt.axes((0.85, 0.1, 0.075, 0.8))
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=cax)
    plt.show()
    
    
    
  
    
plt.show()
    
    

    