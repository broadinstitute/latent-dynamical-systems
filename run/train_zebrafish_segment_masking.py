import argparse
import os
import numpy as np
import wandb
from scipy.sparse import csr_matrix
from scanpy import pp
import torch
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from torch.nn import Dropout

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import get_args
from utils.data_utils import load_data, preprocess_data, preprocess_coords, preprocess_coords_zero_masked, set_timepoints, set_num_steps
from utils.model_utils import initialize_models, initialize_rotations, set_scale
from utils.augmentations import get_gginv, masker
from utils.validation import rotate, get_evals, ge_loss, zer_loss

args = get_args()

## set arguments
VAL_TIME_OFFSETS = range(-args.val_time_offsets, args.val_time_offsets+1)

if args.real_or_debug == 'debug':
    PROJECT = 'ZebrafishPDE'

if args.real_or_debug =='real':
    PROJECT = 'ZebrafishPDEReal'

ZERO_SCALE = args.zero_scale

LEARN_START = True
if args.donot_learn_start:
    LEARN_START = False
    print("Not learning start!")

FIT_GEX0 = True
if args.donot_fit_gex0:
    FIT_GEX0 = False
    print('Only fitting shape from T0')

EQUIV = False
if args.equiv:
    EQUIV = True

INJURY = False
if args.injury:
    INJURY = True

DROPOUT = False
if args.dropout:
    DROPOUT = True

REGISTER = True
if args.donot_register:
    REGISTER = False

REGISTER_VAL = True
if args.donot_register_val:
    REGISTER_VAL = False

if args.filter == "True":
    FILTER = True
elif args.filter == "False":
    FILTER = False

if args.preprocess_gex == "True":
    PREPROCESS_GEX = True
elif args.preprocess_gex == "False":
    PREPROCESS_GEX = False

LEARN_BASELINE_FACTORS = False
if args.learn_baseline_factors:
    LEARN_BASELINE_FACTORS = True

SCALE = args.scale
OUT_DIR = args.out_dir
OUT_PREFIX = args.out_prefix
DIFFUSION_TYPE = args.diffusion_type
try:
    os.mkdir(OUT_DIR)
except:
    print('creation of out dir failed')
    if args.overwrite ==False:
        quit()

kwargs = {}

LATENT_DIM  = args.latent_dim 
HIDDEN_DIM = args.hidden_dim
assert LATENT_DIM > HIDDEN_DIM
RECON_DIM = LATENT_DIM - HIDDEN_DIM
TIMESCALE = args.timescale

CHOSEN_TIMEPOINTS, VALIDATION_TIMEPOINTS, ALL_TIMEPOINTS = set_timepoints(args)

LOG_LOSS_EVERY = 50
LOG_IMS_EVERY = 1000
SAVE_EVERY = 2000
VAL_EVERY = args.val_every
MAX_ITS = args.max_its

BATCH_SIZE = 512

LOSS_WEIGHTS = {k: 1.0 for k in CHOSEN_TIMEPOINTS}
if FIT_GEX0 == False:
    LOSS_WEIGHTS[0] = 0.0

SCALE = args.scale

## Load Data
x = load_data('/home/skambha6/chenlab/lads/data/zebrafish/spatial_fivetime_slice_stereoseq.h5ad')

print("args:")
print(args) 


mask_segments_mode = args.mask_mode
mask_segments_tp = args.mask_tp

tp, tp_masked,gex = preprocess_data(x, args, filter=FILTER, preprocess_gex=PREPROCESS_GEX, mask_segments=True, mask_segments_tp=mask_segments_tp,mask_segments_mode=mask_segments_mode)

size, yshifts, xshifts = set_scale(SCALE)
zero_pos, pos, nonzero_ims = preprocess_coords_zero_masked(tp, size=size, yshifts=yshifts, xshifts=xshifts, mask=True, tp_masked=tp_masked, mask_segments_tp=mask_segments_tp, mask_mode=mask_segments_mode)

idxs = {i : np.arange(len(tp_masked[i][0])) for i in tp_masked}

## Initialize models

init_ind = CHOSEN_TIMEPOINTS[0]
print(f"init_ind: {init_ind}")
init  = torch.rand(size,size,LATENT_DIM,device = 'cuda',dtype = torch.float)
init[(zero_pos[init_ind][0][:,0], zero_pos[init_ind][0][:,1])] = 0
start = init.permute(2,0,1).detach().clone().requires_grad_(True)

print("start shape:")
print(start.shape)

kwargs['reg_weight'] = float(args.mixer_reg)
pde, mixer = initialize_models(args, gex.shape, kwargs)


nonzero = torch.from_numpy(nonzero_ims[init_ind]).float().cuda()
where_nonzero = np.where(nonzero_ims[init_ind])
num_nonzero = len(where_nonzero[0])

num_timepoints = len(CHOSEN_TIMEPOINTS) + len(VALIDATION_TIMEPOINTS)

rots = initialize_rotations(5) # 5 total timepoints


if REGISTER:
    rot_opt = torch.optim.Adam(list(rots.parameters()), lr = args.lr)

if LEARN_START:
    pde_opt = torch.optim.Adam(list(pde.parameters()) + [start], lr = args.lr) 
else:
    pde_opt = torch.optim.Adam(list(pde.parameters()), lr = args.lr)

mixer_opt = False
if len(list(mixer.parameters())) > 0:
    mixer_opt = True
    opt = torch.optim.Adam(list(mixer.parameters()),lr = args.lr)

wandb.login()

run = wandb.init(project = PROJECT, notes=OUT_PREFIX, config = args)
wandb.watch(mixer, log = 'parameters',log_freq = 500)


# Train Latent PDE

wandb.config.update({'loss_weight_{}'.format(k):v for k,v in LOSS_WEIGHTS.items()})

## define evaluation
def evaluate_model(mixer,pde,T0,its, VALIDATION_TIMEPOINTS, VAL_TIME_OFFSETS, register_val = REGISTER_VAL):
    ge_loss_reg = {}
    zero_loss_reg = {}
    ge_diff_ims = {}
    outline_ims = {}
    unreg_ims = {}
    print("==== BEGIN VALIDATION ====")
    for sel in VALIDATION_TIMEPOINTS:
        current = 10000000

        for time_offset in VAL_TIME_OFFSETS:
            glr, zlr, gdi, oi,ui =  get_evals(mixer, pde, tp, T0,sel,idxs, pos, zero_pos, size, TIMESCALE, RECON_DIM, args,time_offset = 0)
            m = np.median(glr)
            if m<current:
                ge_loss_reg[sel] = glr 
                zero_loss_reg[sel]=zlr
                ge_diff_ims[sel] = gdi 
                outline_ims[sel] = oi
                unreg_ims[sel] = ui
                current = m.copy()

        plt.Figure(figsize = (18*3,3*3))
        plt.subplot(1,6,1)
        plt.imshow(unreg_ims[sel])
        plt.axis('off')
        plt.subplot(1,6, 2)
        plt.imshow(nonzero_ims[sel])
        plt.axis('off')
        plt.subplot(1,6,3)
        plt.imshow(outline_ims[sel])
        plt.axis('off')
        plt.subplot(1,6, 4)
        plt.imshow(nonzero_ims[sel],alpha = 0.5, cmap = 'Greys')
        plt.imshow(outline_ims[sel],alpha = 0.5,vmin = 0, vmax = 0.05)
        plt.axis('off')
        plt.subplot(1,6,5)
        vmax = np.percentile(ge_diff_ims[sel], 99.9)
        if vmax<1e-6:
            vmax = 1e-6
        plt.imshow(ge_diff_ims[sel], cmap = 'bwr',vmin =0, vmax = vmax)
        plt.axis('off')
        wandb.log({'val/alignments/_{}'.format(sel): plt}, step =its)
        plt.close()

    ## logging histogram? takes a while
    wandb.log({'val/ge_median_{}'.format(k): np.median(v) for k,v in ge_loss_reg.items()},step = its)
    wandb.log({'val/zero_median_{}'.format(k): np.median(v) for k,v in zero_loss_reg.items()}, step = its)
    # wandb.log({'val/ge_reg_{}'.format(k): wandb.Histogram(v) for k,v in ge_loss_reg.items()}, step = its)
    # wandb.log({'val/zero_reg_{}'.format(k): wandb.Histogram(v) for k,v in zero_loss_reg.items()}, step = its)
    
    print("==== END VALIDATION ====")



num_steps = set_num_steps(CHOSEN_TIMEPOINTS, args)

print("tp")
for k, v in tp.items():
    print(k)

if LEARN_BASELINE_FACTORS:
    print("Running NMF to learn baseline factors") 
    baseline_factor_ims = {}
    from sklearn.decomposition import NMF

    num_baseline_factors = 10
    nmf = NMF(n_components=num_baseline_factors, init='nndsvd', max_iter=2000)

    gex_chosen_tps = []
    for k in CHOSEN_TIMEPOINTS:
        gex_chosen_tps.append(tp[k][1].detach().cpu())
    gex_chosen_tps = torch.cat(gex_chosen_tps, dim = 0)

    print(gex_chosen_tps.shape)
    nmf.fit(gex_chosen_tps)

    H = nmf.components_

    for k in CHOSEN_TIMEPOINTS:
        print(f"k: {k}") 
        print(tp[k][1].shape)
        gex_factors = nmf.transform(tp[k][1].detach().cpu())
        baseline_factor_im = np.zeros((size,size, num_baseline_factors))
        baseline_factor_im[pos[k][:,0], pos[k][:,1]] = gex_factors 
        baseline_factor_ims[k] = baseline_factor_im

for its in tqdm(range(MAX_ITS)):
    imgs = []
    sum_imgs = []
    ims = []
    loss = 0 
    bias_reg = 0 # since timepoints are roughly centered, expect bias to be small
    theta_reg = 0 # since timepoints are roughly aligned, expect rotations to be small
    losses = {}

    mixer.build_params()
    pde.build_params()

    g,ginv = get_gginv()
    rot,b = g
    inv_rot,inv_b = ginv
    dec_eval = mixer.decoder

    for sel_ind, sel in enumerate(CHOSEN_TIMEPOINTS):
        rot = rots[sel].weight
        b = rots[sel].bias

        co, ge = tp[sel]
        samp = np.random.choice(idxs[sel], BATCH_SIZE)
        coord = pos[sel][samp]
        i,j = (coord[:,0],coord[:,1])

        zer, zer_inds,ratio = zero_pos[sel]
        zer_samp = np.random.choice(zer_inds, int(ratio*BATCH_SIZE))
        zer_coord = zer[zer_samp]
        zer_i,zer_j = (zer_coord[:,0], zer_coord[:,1])


        if sel_ind == 0:
            initial = nonzero[None,None]*(torch.square(start) + 0.001*torch.rand_like(start))

            t2 = initial[0].permute(1,2,0)

            if INJURY:
                mask_pix = np.random.randint(num_nonzero)
                k = np.random.randint(3)
                mask = masker[k,where_nonzero[0][mask_pix], where_nonzero[1][mask_pix]]
                temp = mask*initial 
            else:
                temp = initial

            if EQUIV:
                temp = rotate(rot,b,temp)
        elif sel_ind>0:
            temp = initial

            if DROPOUT:
                ## apply with probability 0.5
                if np.random.rand() < 0.5:
                    temp = Dropout(p=0.2)(temp)

            for _ in range(num_steps[sel]):
                temp = pde.step(temp)
            if EQUIV:
                temp = rotate(inv_rot,inv_b,temp)

            if REGISTER: ##TODO: make register conditions and rotation conditions separate
                t2 = rotate(rots[sel].weight,rots[sel].bias,temp)[0].permute(1,2,0)
            else:
                t2 = temp[0].permute(1,2,0) 

            if REGISTER:
                bias_reg += torch.square(rots[sel].bias).sum()
                theta_reg += torch.square(rots[sel].weight).sum()

        if LEARN_BASELINE_FACTORS:
            baseline_factor_im = baseline_factor_ims[sel]
            gex_residual = ge[samp] - torch.tensor(np.matmul(baseline_factor_im[i,j], H)).cuda()
            gex_residual[gex_residual < 0] = 0 ## set negative residuals to 0
            ge_diff = ge_loss(t2[(i,j)], RECON_DIM, dec_eval, gex_residual)
        else:
            ge_diff = ge_loss(t2[(i,j)], RECON_DIM, dec_eval, ge[samp])

        zer_diff = zer_loss(t2[zer_i, zer_j], RECON_DIM, dec_eval)

        l = ge_diff.mean() + ZERO_SCALE*zer_diff.mean()
        loss+=LOSS_WEIGHTS[sel]*l
        losses[sel] = l.detach().cpu().numpy()
        ims.append(t2.detach().cpu().numpy())

    if mixer_opt:
        opt.zero_grad()
    pde_opt.zero_grad()
    if REGISTER:
        rot_opt.zero_grad() 

    if mixer_opt:
        mixer_reg = mixer.reg()
    else:
        mixer_reg = 0

    pde_reg = pde.reg()
    (loss + theta_reg + bias_reg + mixer_reg + pde_reg).backward() 

    
    if np.random.rand()<0.9 and REGISTER:
        rot_opt.step()
    pde_opt.step()
    if mixer_opt:
        opt.step()

    if (its+1)%LOG_LOSS_EVERY==0:
        wandb.log({'loss/{}'.format(sel):losses[sel] for sel in losses}, step = its )
        wandb.log({'mixer_reg':mixer_reg, 'pde_reg':pde_reg}, step = its)
    
    if (its+1)%LOG_IMS_EVERY==0:
        
        for k,se in enumerate(CHOSEN_TIMEPOINTS):
            plt.Figure(figsize = (10*LATENT_DIM,10))
            for j in range(LATENT_DIM):
                plt.subplot(1,LATENT_DIM,j+1)
                plt.imshow(ims[k][:,:,j],cmap = 'Reds')
                plt.axis('off')
            wandb.log({'latents_{}'.format(se): plt}, step =its)
            plt.close()

        if LEARN_BASELINE_FACTORS:
            for k,se in enumerate(CHOSEN_TIMEPOINTS):
                plt.Figure(figsize = (10*num_baseline_factors,10))
                for j in range(num_baseline_factors):
                    plt.subplot(1,num_baseline_factors,j+1)
                    plt.imshow(baseline_factor_ims[se][:,:,j],cmap = 'Reds')
                    plt.axis('off')
                wandb.log({'baseline_factor_{}'.format(se): plt}, step =its)
                plt.close()
        
        for k,se in enumerate(CHOSEN_TIMEPOINTS):
            plt.Figure(figsize = (10,10))
            plt.imshow(ims[k].sum(axis =-1),alpha  = 0.8)
            plt.axis('off')
            plt.imshow(nonzero_ims[se], alpha = 0.5, cmap = 'Greys')
            plt.axis('off')
            wandb.log({'outline_{}'.format(se):plt},step = its)
            plt.close()

    if (its+1)%VAL_EVERY==0:
        evaluate_model(mixer, pde, initial.detach(),its, VALIDATION_TIMEPOINTS, VAL_TIME_OFFSETS)

    if (its+1)%SAVE_EVERY==0:
        if mixer_opt:
            state_dict = { 'pde':pde.state_dict(),'mixer':mixer.state_dict(),
                        'pde_opt':pde_opt.state_dict(), 'mixer_opt':opt.state_dict(),
                        'start':start, 'rots': [r.state_dict() for r in rots],
                        'run_id':run.id}
        else:
            state_dict = { 'pde':pde.state_dict(),'mixer':mixer.state_dict(),
                'pde_opt':pde_opt.state_dict(), 'start':start, 'rots': [r.state_dict() for r in rots],
                'run_id':run.id}
        torch.save(state_dict, os.path.join(OUT_DIR, OUT_PREFIX + '_ckpt_iter_{}'.format(its)))
        print('saved checkpoint after {}'.format(its+1))

