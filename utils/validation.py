
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.stats import pearsonr
import os

from utils.config import get_args
from utils.data_utils import load_data, preprocess_data, preprocess_coords, set_timepoints, set_num_steps
from utils.model_utils import initialize_models, initialize_rotations, set_scale

def get_evals(mixer, pde, tp, T0,sel,idxs, pos, zero_pos, size, TIMESCALE, RECON_DIM, args,time_offset = 0, register_val = True):

    if args.val_reg_type == 'orthogonal':
        from models.registration import OrthoRegistration as Registration
    elif args.val_reg_type =='affine':
        from models.registration import AffineRegistration as Registration
    elif args.val_reg_type =='conformal':
        from models.registration import ConformalRegistration as Registration

    batch_size = 512
    ref_x = torch.tensor([[-1,0],[0,1]]).float()

    if register_val:
        rot_inits = [torch.eye(2), torch.eye(2), torch.eye(2), torch.eye(2), torch.eye(2), torch.eye(2)]

    mixer.eval()
    pde.eval()
    
    co, ge = tp[sel]

    if register_val:
        rot = Registration(weight = rot_inits[sel].cuda().clone().detach(),bias = torch.zeros(2,dtype = torch.float,device = 'cuda'),device = 'cuda')
        val_opt = torch.optim.Adam(list(rot.parameters()), lr = 0.0005)

    with torch.no_grad():    
        temp = T0.detach().clone()
        for _ in range(sel*TIMESCALE + time_offset):
            temp = pde.step(temp)
    
    t = temp.detach().clone()
    ge_diff_im = np.zeros((size,size))
    ge_loss_reg_batches = []
    zero_loss_reg_batches = []
    for _ in tqdm(range(500)): 
        dec_eval = mixer.decoder
        mixer.build_params()

        if register_val:
            weight = rot.weight()
            bias = rot.bias()

            t2 = rotate(weight,bias,t)[0].permute(1,2,0)
        else:
            t2 = t[0].permute(1,2,0)

        samp = np.random.choice(idxs[sel], batch_size)

        zer, zer_inds,ratio = zero_pos[sel]
        zer_samp = np.random.choice(zer_inds, int(ratio*batch_size))

        ge_loss_reg, ge_diff_im = compute_batched_ge_loss_eval(pos[sel], t2, dec_eval, ge, RECON_DIM, samp, ge_diff_im)
        zero_loss_reg = compute_batched_zer_loss_eval(zero_pos[sel], t2, dec_eval, RECON_DIM, zer_samp)

        ge_loss_reg_batches.append(ge_loss_reg)
        zero_loss_reg_batches.append(zero_loss_reg)

        loss = np.concatenate(ge_loss_reg).mean() + np.concatenate(zero_loss_reg).mean()
        
        if register_val:
            val_opt.zero_grad()
            (loss + 1e-8*torch.square(bias).mean()).backward()
            val_opt.step()

    with torch.no_grad():

        outline_im = t2.mean(axis =-1).detach().cpu().numpy()
        unreg_im = t[0].mean(axis = 0).detach().cpu().numpy()

    mixer.train()
    pde.train()

    glr = np.concatenate(ge_loss_reg_batches)
    zlr = np.concatenate(ge_loss_reg_batches)

    return glr, zlr, ge_diff_im, outline_im,unreg_im


def rotate(W,b, image, align_corners = False):
    theta = torch.cat([W,b[:,None]], dim = -1)[None]
    affine_grid = torch.nn.functional.affine_grid(theta, size = image.shape, align_corners=align_corners) ## TODO: double check this
    return torch.nn.functional.grid_sample(image, affine_grid, align_corners=align_corners)

def compute_batched_ge_loss_eval(pos_sel, t2, dec_eval, ge, RECON_DIM, samp, ge_diff_im=None):
    coord = pos_sel[samp]
    i,j = (coord[:,0],coord[:,1])

    ge_diff = ge_loss(t2[(i,j)], RECON_DIM, dec_eval, ge[samp])
    ge_loss_reg = (ge_diff.detach().cpu().numpy()) # shape is num_pos x num_genes

    if ge_diff_im is not None:
        ge_diff_im[(i,j)] += ge_loss_reg.mean(axis = 1) # mean over genes, shape is num_pos x 1
    
    if ge_diff_im is not None:
        return ge_loss_reg, ge_diff_im
    else:
        return ge_loss_reg
    

def ge_loss(t2_ge, RECON_DIM, dec_eval, ge):
    t2_ge_recon = t2_ge[...,:RECON_DIM]
    ge_diff = (torch.square(dec_eval(t2_ge_recon) - ge))
    return ge_diff 

def compute_batched_zer_loss_eval(zero_pos_sel, t2, dec_eval, RECON_DIM, zer_samp):

    zer, zer_inds,ratio = zero_pos_sel
    zer_coord = zer[zer_samp]
    zer_i,zer_j = (zer_coord[:,0], zer_coord[:,1])

    zer_diff = zer_loss(t2[(zer_i,zer_j)], RECON_DIM, dec_eval)
    zero_loss_reg = (zer_diff.detach().cpu().numpy())

    return zero_loss_reg

def zer_loss(t2_zer, RECON_DIM, dec_eval):
    t2_zer_recon = t2_zer[...,:RECON_DIM]
    zer_diff = torch.square(dec_eval(t2_zer_recon))
    return zer_diff



