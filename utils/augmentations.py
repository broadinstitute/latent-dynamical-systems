# utils.py

import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from torch.nn.functional import affine_grid, grid_sample

def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.max(kernel)

@torch.no_grad()
def get_gginv(s = 0.2):
    A = torch.randn(2,2).cuda()
    _,_,rot = torch.svd(A)
    b = (torch.randn(2)/10).cuda()
    
    scale = torch.exp(-s + 2*s*torch.rand(1,device = 'cuda'))
    
    inv_rot = rot.T/scale
    inv_b = -inv_rot@b
    return ((scale*rot,b),(inv_rot, inv_b))

def masker(size):
    ker = np.ones((3,size,size,size,size))

    ## gaussian kernel to simulate injury?
    for k in [1,2]:
        for i in range(2,43):
            for j in range(2,43):
                if k==2:
                    ker[k,i,j,(i-2):(i+3),(j-2):(j+3)] = 1-gkern(5, sig = 1.5)
                if k==1:
                    ker[k,i,j,(i-1):(i+2),(j-1):(j+2)] = 1-gkern(3, sig = 1.5)

    masker = torch.from_numpy(ker).float().cuda()

    return masker


