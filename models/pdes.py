import torch
from torch import nn


class ReactionDiffusionPDE(nn.Module):
    def __init__(self, latent_dim, diffusion_type = 'uniform',use_grads = True,reg_weight = 0) -> None:
        super().__init__()
        sobelx = (torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]])).float()
        sobely = sobelx.T

        grad = torch.stack([sobelx,sobely])[:,None].cuda()
        laplace = (torch.tensor([[0,1,0],[1,-4,1],[0,1,0]])).float()
        self.kernel = torch.cat([laplace[None,None].cuda(), grad])
        self.lmu = nn.Parameter(torch.rand(1), requires_grad = True)
        if diffusion_type=='uniform':
            self.ldiff = nn.Parameter(torch.rand(1), requires_grad=True)
        else:
            self.ldiff = nn.Parameter(torch.rand(1,latent_dim,1,1), requires_grad=True)

        self.reaction = nn.Sequential(nn.Conv2d(3*latent_dim, latent_dim, 1,bias = None), nn.Tanh())
        self.reaction[0].weight.data[:,latent_dim:] = 0
        self.relu  = nn.ReLU()
        self.reg_weight = reg_weight

        if use_grads:
            self.grad_weight = 1.0
        else:
            self.grad_weight = 0.0


    def build_params(self):
        pass
    
    def step(self, x):
        n,c,i,j = x.shape
        state = torch.nn.functional.conv2d(x.reshape(-1,1,i,j), self.kernel,padding = (1,1))
        mu = torch.sigmoid(self.lmu)
        diffusion = torch.exp(-3 + self.ldiff)*state[:,0].reshape(n,c,i,j)
        grads = state[:,1:].reshape(n,2*c,i,j)
        react = self.reaction(torch.cat([x, self.grad_weight*grads], dim = 1))
        return self.relu(x + mu* diffusion + (1-mu)* react)

    
    def reg(self):
        return self.reg_weight* torch.abs(self.reaction[0].weight).mean()
    


class ReactionDiffusionPDE3D(nn.Module):
    def __init__(self, latent_dim, diffusion_type = 'uniform',use_grads = True,reg_weight = 0) -> None:
        super().__init__()

        # 3D Sobel filter kernels for X, Y, and Z directions
        sobel_x = torch.tensor([[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                            [[2, 0, -2], [4, 0, -4], [2, 0, -2]],
                            [[1, 0, -1], [2, 0, -2], [1, 0, -1]]]).float()

        sobel_y = torch.tensor([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                            [[2, 4, 2], [0, 0, 0], [-2, -4, -2]],
                            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]).float()

        sobel_z = torch.tensor([[[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                            [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]]).float()

        # Normalize the kernels (optional)
        sobel_x /= 16.0
        sobel_y /= 16.0
        sobel_z /= 16.0



        # 3D Laplace filter kernel
        laplace_3d = torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                                [[0, 0, 0], [0, 1, 0], [0, 0, 0]]],

                            [[[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                                [[1, -6, 1], [-6, 26, -6], [1, -6, 1]],
                                [[0, 1, 0], [1, -6, 1], [0, 1, 0]]],

                            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                                [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]]).float()

        # Normalize the Laplace filter kernel (optional)
        laplace = laplace_3d[0] / 6.0
        
        self.kernel = torch.stack([laplace,sobel_z,sobel_x,sobel_y])[:,None].float().cuda()

        
        self.lmu = nn.Parameter(torch.rand(1), requires_grad = True)
        if diffusion_type=='uniform':
            self.ldiff = nn.Parameter(torch.rand(1), requires_grad=True)
        else:
            self.ldiff = nn.Parameter(torch.rand(1,latent_dim,1,1,1), requires_grad=True)

        self.reaction = nn.Sequential(nn.Conv3d(4*latent_dim, latent_dim, 1,bias = None), nn.Tanh())
        self.reaction[0].weight.data[:,latent_dim:] = 0
        self.relu  = nn.ReLU()
        self.reg_weight = reg_weight

        if use_grads:
            self.grad_weight = 1.0
        else:
            self.grad_weight = 0.0


    def build_params(self):
        pass
    
    def step(self, x):
        n,c,k,i,j = x.shape
        state = torch.nn.functional.conv3d(x.reshape(-1,1,k,i,j), self.kernel,padding = (1,1,1))
        mu = torch.sigmoid(self.lmu)
        diffusion = torch.exp(-3 + self.ldiff)*state[:,0].reshape(n,c,k,i,j)
        grads = state[:,1:].reshape(n,3*c,k,i,j)
        react = self.reaction(torch.cat([x, self.grad_weight*grads], dim = 1))
        return self.relu(x + mu* diffusion + (1-mu)* react)

    
    def reg(self):
        return self.reg_weight* torch.abs(self.reaction[0].weight).mean()
    
