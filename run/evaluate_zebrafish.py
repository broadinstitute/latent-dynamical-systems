import argparse
import os
import numpy as np
import wandb
from scipy.sparse import csr_matrix
from scanpy import pp
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import random
from scipy.stats import truncnorm
from matplotlib import gridspec

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import get_args
from utils.data_utils import load_data, preprocess_data, preprocess_coords, set_timepoints, set_num_steps
from utils.model_utils import initialize_models, initialize_rotations, set_scale
from utils.augmentations import get_gginv, masker
from utils.validation import rotate, get_evals, compute_batched_ge_loss_eval, compute_batched_zer_loss_eval, ge_loss, zer_loss

import pickle
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
# USE_GENES = 'all_genes.npy'
if args.use_genes == 'seq':
    USE_GENES = 'genes_with_seq.npy'

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



LATENT_DIM  =args.latent_dim 
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

BATCH_SIZE = 128
RANDOM_BATCHING = False

LOSS_WEIGHTS = {k: 1.0 for k in ALL_TIMEPOINTS} ## TODO: validation timepoint inegration
if FIT_GEX0 == False:
    LOSS_WEIGHTS[ALL_TIMEPOINTS[0]] = 0.0

SCALE = args.scale

size, yshifts, xshifts = set_scale(SCALE)


RESULTS_DIR = "/home/skambha6/chenlab/lads/code/run/zebrafish_evaluation"

checkpoint_iteration = int(args.ckpt_iter) - 1 ## -1 to correct for 0 indexing 


## Load Data
x = load_data('/home/skambha6/chenlab/lads/data/zebrafish/spatial_fivetime_slice_stereoseq.h5ad')

tp, gex = preprocess_data(x, args, filter=FILTER, preprocess_gex=PREPROCESS_GEX)

zero_pos, pos, nonzero_ims = preprocess_coords(tp, size=size, yshifts=yshifts, xshifts=xshifts)

idxs = {i : np.arange(len(tp[i][0])) for i in tp}



## Evaluate PDE
wandb.login()


fig1,ax1 = plt.subplots(nrows=3, ncols=5, figsize=(20,10))
plt.tight_layout()
fig1.subplots_adjust(hspace = 1)

fig2,ax2 = plt.subplots(nrows=args.num_reps+1, ncols=5, figsize=(20,10))
plt.tight_layout()
fig2.subplots_adjust(hspace = 1)

print("Registration set to: {}".format(REGISTER))

for rep in range(args.num_reps):

    init_ind = CHOSEN_TIMEPOINTS[0]
    nonzero = torch.from_numpy(nonzero_ims[init_ind]).float().cuda()
    where_nonzero = np.where(nonzero_ims[init_ind])
    
    try:
        num_nonzero = len(where_nonzero[init_ind])
    except:
        num_nonzero = None 

    # initialize models
    state_dict = torch.load(os.path.join(OUT_DIR, OUT_PREFIX + f'_rep{rep+1}_ckpt_iter_{checkpoint_iteration}'))

    run_id = state_dict['run_id']

    # Initialize rotations properly
    rots = initialize_rotations(5)  # 5 total timepoints
    for sel, rot_state in enumerate(state_dict['rots']):
        rots[sel].load_state_dict(rot_state)

    ## create optimizer for val_rots
    rots_opt = torch.optim.Adam(list(rots.parameters()),lr = 0.01)

    # Initialize initial state once
    start = state_dict['start'].cuda()
    initial = nonzero[None,None]*(torch.square(start) + 0.001*torch.rand_like(start))

    kwargs['reg_weight'] = float(args.mixer_reg)
    pde, mixer = initialize_models(args, gex.shape, kwargs)

    pde.load_state_dict(state_dict['pde'])
    mixer.load_state_dict(state_dict['mixer'])

    pde.eval()  
    mixer.eval()

    num_steps = set_num_steps(ALL_TIMEPOINTS, args)
    
    
    ## iterate through for num_steps and compute loss against all timepoints at each step
    ge_loss_reg_tps = {}
    zero_loss_reg_tps = {}

    for sel in ALL_TIMEPOINTS:
        ge_loss_reg_tps[sel] = []
        zero_loss_reg_tps[sel] = []

    ge_diff_im_tps = []
    outline_im_tps = []
    unreg_im_tps = []

    list_of_coord_across_its = []
    list_of_dec_smg_across_its = []


    time_offset = 5
    T0 = initial.detach()
    x_dim=y_dim=size

    ## add in val_opt registration

    if REGISTER_VAL and len(VALIDATION_TIMEPOINTS) > 0:

        def sample_truncated_normal(a, b):
            mean = int((a + b) / 2)

            # Create the truncated normal distribution
            trunc_normal_dist = truncnorm(a, b, loc=mean, scale=1)
            
            # Sample a value
            sample = trunc_normal_dist.rvs()
        
            # Ensure the sampled value is an integer
            return int(np.round(sample))
        
        ## Optimizing validation rotation
        for its in tqdm(range(500)):
            temp = T0.clone()
            bias_reg = 0
            loss = 0

            mixer.build_params()
            pde.build_params()

            dec_eval = mixer.decoder

            for val_ind, sel in enumerate(VALIDATION_TIMEPOINTS):
                ## randomly pick timestep between previous and next timepoint to iterate towards and apply rotation, sampling following a normal distribution centered at middle of two timepoints
                if sel > 2:
                    num_steps_val = sample_truncated_normal(num_steps[sel-1]+1,num_steps[sel+1]-1) ## assuming there is a timestep before and after VALIDATION TIMEPOINT
                else:
                    num_steps_val = sample_truncated_normal(1,num_steps[sel+1]-1)
                for _ in range(num_steps_val):
                    temp = pde.step(temp)
                rot_temp = rotate(rots[val_ind].weight,rots[val_ind].bias,temp)[0].permute(1,2,0)
                co, ge = tp[sel]
                samp = np.random.choice(idxs[sel], BATCH_SIZE)
                coord = pos[sel][samp]
                i,j = (coord[:,0],coord[:,1])

                zer, zer_inds,ratio = zero_pos[sel]
                zer_samp = np.random.choice(zer_inds, int(ratio*BATCH_SIZE))
                zer_coord = zer[zer_samp]
                zer_i,zer_j = (zer_coord[:,0], zer_coord[:,1])
                ge_diff = ge_loss(rot_temp[(i,j)], RECON_DIM, dec_eval, ge[samp])
                zer_diff = zer_loss(rot_temp[zer_i, zer_j], RECON_DIM, dec_eval)

                loss += ge_diff.mean() + ZERO_SCALE*zer_diff.mean()
                bias_reg += 10*torch.square(rots[val_ind].bias).sum()

            rots_opt.zero_grad()
            (loss + bias_reg).backward()
            rots_opt.step()

        # write val rots into rots for subsequent evaluation
        for i, sel in enumerate(VALIDATION_TIMEPOINTS):
            rots[sel].weight = rots[sel].weight
            rots[sel].bias = rots[sel].bias

        print("rotations:")
        for sel in ALL_TIMEPOINTS:
            print(str(sel) + ": " + str(rots[sel].weight) + ", " + str(rots[sel].bias))

    with torch.no_grad():    
        temp = T0.detach().clone()
        
        for it in range(num_steps[ALL_TIMEPOINTS[len(ALL_TIMEPOINTS)-1]] + time_offset): 

            # print(f"iteration: {it}")
            temp = pde.step(temp)
         
            t = temp.detach().clone()
            
            mixer.build_params()
            dec_eval = mixer.decoder

            t2 = t[0].permute(1,2,0)

            with torch.no_grad():
                for sel_ind, sel in enumerate(ALL_TIMEPOINTS):
                    ge_loss_reg_batches = []
                    zero_loss_reg_batches = []

                    _, ge = tp[sel]
                    ge_diff_im = np.zeros((x_dim, y_dim))

                    zer, zer_inds,ratio = zero_pos[sel]
                    
                    ## TODO: this rotate is messing things up with all timepoints 
                    if REGISTER: #and sel in VALIDATION_TIMEPOINTS: 
                        t2 = rotate(rots[sel].weight,rots[sel].bias,t)[0].permute(1,2,0)
                    elif REGISTER_VAL and sel in VALIDATION_TIMEPOINTS:
                        t2 = rotate(rots[sel].weight,rots[sel].bias,t)[0].permute(1,2,0)
                    else:
                        t2 = t[0].permute(1,2,0)

                    if RANDOM_BATCHING:
                        for _ in range(100):
                            samp = np.random.choice(idxs[sel], BATCH_SIZE)
                            ge_loss_reg, ge_diff_im = compute_batched_ge_loss_eval(pos[sel], t2, dec_eval, ge, RECON_DIM, samp, ge_diff_im)
                            ge_diff_im += ge_diff_im
                            ge_loss_reg_batches.append(ge_loss_reg)

                            zer_samp = np.random.choice(zer_inds, int(ratio*BATCH_SIZE))
                            zero_loss_reg = compute_batched_zer_loss_eval(zero_pos[sel], t2, dec_eval, RECON_DIM, zer_samp)
                            zero_loss_reg_batches.append(zero_loss_reg)

                    else:
                        #iterate through all positions; do non-zero and zero positions separately
                        for i in range(0, idxs[sel].shape[0], BATCH_SIZE):
                            if i + BATCH_SIZE < idxs[sel].shape[0]:
                                samp = idxs[sel][i:i+BATCH_SIZE]
                            else:
                                samp = idxs[sel][i:]
                            ge_loss_reg, ge_diff_im = compute_batched_ge_loss_eval(pos[sel], t2, dec_eval, ge, RECON_DIM, samp, ge_diff_im)
                            ge_diff_im += ge_diff_im
                            ge_loss_reg_batches.append(ge_loss_reg)

                        for i in range(0, zer.shape[0], BATCH_SIZE):
                            if i + BATCH_SIZE < zer.shape[0]:
                                zer_samp = zer[i:i+BATCH_SIZE]
                            else:
                                zer_samp = zer[i:]
                            zer_i,zer_j = (zer_samp[:,0], zer_samp[:,1])
                            zero_loss_reg = zer_loss(t2[zer_i, zer_j], RECON_DIM, dec_eval).detach().cpu().numpy()
                            zero_loss_reg_batches.append(zero_loss_reg)
                    
                    outline_im = t2.mean(axis = -1).detach().cpu().numpy()
                    unreg_im = t[0].mean(axis = 0).detach().cpu().numpy()

                    ge_loss_reg_batches = np.concatenate(ge_loss_reg_batches)
                    zero_loss_reg_batches = np.concatenate(zero_loss_reg_batches)

                    # Combine losses with weights before taking mean, like in training
                    if RANDOM_BATCHING:
                        total_loss = np.array(ge_loss_reg_batches).mean() + ZERO_SCALE * np.array(zero_loss_reg_batches).mean()
                    else:
                        total_loss = np.array(ge_loss_reg_batches).mean() + np.array(zero_loss_reg_batches).mean()

                    total_loss = total_loss * LOSS_WEIGHTS[sel]

                    ge_loss_reg_tps[sel].append(ge_loss_reg_batches.mean())
                    zero_loss_reg_tps[sel].append(zero_loss_reg_batches.mean())
                    ge_diff_im_tps.append(ge_diff_im)
                    outline_im_tps.append(outline_im)
                    unreg_im_tps.append(unreg_im)                


    # plot mean loss across its 
    for sel_ind, sel in enumerate(ALL_TIMEPOINTS):
        num_steps[ALL_TIMEPOINTS[0]] = 0
        ax1[0,sel_ind].plot(ge_loss_reg_tps[sel])
        ax1[0,sel_ind].set_xlabel('Iteration')
        ax1[0,0].set_ylabel('Mean Loss')
        ax1[0,sel_ind].set_title(f'GEX mean Loss for tp {sel}')
        ax1[0,sel_ind].axvline(x=num_steps[sel],color='r')
        if sel in VALIDATION_TIMEPOINTS:
            ax1[0,sel_ind].set_title(f'GEX mean Loss for tp {sel}', color='red')
            ax1[0,sel_ind].axvline(x=num_steps[sel-1],linestyle='--',color='r')
            ax1[0,sel_ind].axvline(x=num_steps[sel+1],linestyle='--',color='r')
        # ax[sel].set_ylim(0.1,0.6)
        
    for sel_ind, sel in enumerate(ALL_TIMEPOINTS):
        ax1[1,sel_ind].plot(zero_loss_reg_tps[sel])
        ax1[1,sel_ind].set_xlabel('Iteration')
        ax1[1,0].set_ylabel('Loss')
        ax1[1,sel_ind].set_title(f'Zero Loss for tp {sel}')
        ax1[1,sel_ind].axvline(x=num_steps[sel],color='r')
        if sel in VALIDATION_TIMEPOINTS:
            ax1[1,sel_ind].set_title(f'Zero Loss for tp {sel}', color='red')
            ax1[1,sel_ind].axvline(x=num_steps[sel-1],linestyle='--',color='r')
            ax1[1,sel_ind].axvline(x=num_steps[sel+1],linestyle='--',color='r')

    for sel_ind, sel in enumerate(ALL_TIMEPOINTS):
        ax1[2,sel_ind].plot(np.array(ge_loss_reg_tps[sel]) + np.array(zero_loss_reg_tps[sel]))
        ax1[2,sel_ind].set_xlabel('Iteration')
        ax1[2,0].set_ylabel('Loss')
        ax1[2,sel_ind].set_title(f'Total Loss for tp {sel}')
        ax1[2,sel_ind].axvline(x=num_steps[sel],color='r')
        if sel in VALIDATION_TIMEPOINTS:
            ax1[2,sel_ind].set_title(f'Total Loss for tp {sel}', color='red')
            ax1[2,sel_ind].axvline(x=num_steps[sel-1],linestyle='--',color='r')
            ax1[2,sel_ind].axvline(x=num_steps[sel+1],linestyle='--',color='r')

    print(f"completed rep {rep+1}") 


    validation_sel_min_total_loss_iter = {}   
    validation_sel_min_gex_loss = {}
    validation_sel_min_zero_loss = {}
    validation_sel_min_total_loss = {}     



    ## calculate minimum mean, median, and sum of interpolation loss for each validation timepoint
    for sel in VALIDATION_TIMEPOINTS:

        ## find the iteration where the min total loss (gex + zero) is achieved, and then print all of the losses at that iteration
        total_loss = np.array(ge_loss_reg_tps[sel][num_steps[sel-1]:num_steps[sel+1]]) + np.array(zero_loss_reg_tps[sel][num_steps[sel-1]:num_steps[sel+1]])
        min_total_loss_iter = np.argmin(total_loss) + num_steps[sel-1]
        validation_sel_min_total_loss_iter[sel] = min_total_loss_iter

        validation_sel_min_total_loss[sel] = np.min(total_loss) 

        validation_sel_min_gex_loss[sel] = ge_loss_reg_tps[sel][min_total_loss_iter]
        validation_sel_min_zero_loss[sel] = zero_loss_reg_tps[sel][min_total_loss_iter]

        print(np.min(total_loss))

    run = wandb.init(project=PROJECT, notes=OUT_PREFIX, resume="must", id = run_id)

    # run.log({'minimum_gex_loss_mean': min_gex_loss_mean, 'minimum_zero_loss_mean': min_zero_loss_mean, 'minimum_gex_loss_median': min_gex_loss_median, 'minimum_zero_loss_median': min_zero_loss_median, 'minimum_gex_loss_sum': min_gex_loss_sum, 'minimum_zero_loss_sum': min_zero_loss_sum}, step=args.ckpt_iter)
    
    for sel in VALIDATION_TIMEPOINTS:
        run.log({f'minimum_gex_loss_mean_tp_{sel}': validation_sel_min_gex_loss[sel]})
        run.log({f'minimum_zero_loss_mean_tp_{sel}': validation_sel_min_zero_loss[sel]})
        run.log({f'minimum_total_loss_mean_tp_{sel}': validation_sel_min_total_loss[sel]})
        
    fig1.savefig(os.path.join(os.path.join(RESULTS_DIR,'eval'),'eval_' + OUT_PREFIX + f'_ckpt_iter_{args.ckpt_iter-1}' + '.pdf'), transparent=True, bbox_inches='tight', dpi=100)

    torch.cuda.empty_cache()

    zero_pos, pos, nonzero_ims = preprocess_coords(tp, size=size, yshifts=yshifts, xshifts=xshifts)
    for i in range(1,5):
        ax2[0,i].imshow(nonzero_ims[i])

    nonzero = torch.from_numpy(nonzero_ims[init_ind]).float().cuda()
    where_nonzero = np.where(nonzero_ims[init_ind])

    try:
        num_nonzero = len(where_nonzero[init_ind])
    except:
        num_nonzero = None 

    # initialize models
    state_dict = torch.load(os.path.join(OUT_DIR, OUT_PREFIX + f'_rep{rep+1}_ckpt_iter_{int(args.ckpt_iter)-1}'))

    run_id = state_dict['run_id']

    # Load models
    start = state_dict['start'].cuda()
    initial = nonzero[None,None]*(torch.square(start) + 0.001*torch.rand_like(start))

    list_of_latent_tps = []
    list_of_latent_its = []
    INJURY=False
    EQUIV=False
    for sel_ind, sel in enumerate(ALL_TIMEPOINTS):

        co, ge = tp[sel]
        samp = np.random.choice(idxs[sel], BATCH_SIZE)
        coord = pos[sel][samp]
        i,j = (coord[:,0],coord[:,1])

        zer, zer_inds,ratio = zero_pos[sel]
        zer_samp = np.random.choice(zer_inds, int(ratio*BATCH_SIZE))
        zer_coord = zer[zer_samp]
        zer_i,zer_j = (zer_coord[:,0], zer_coord[:,1])

        if sel_ind ==0:
            initial = nonzero[None,None]*(torch.square(start) + 0.001*torch.rand_like(start))

            t2 = initial[0].clone().detach().permute(1,2,0)

            if INJURY:
                mask_pix = np.random.randint(num_nonzero)
                k = np.random.randint(3)
                mask = masker[k,where_nonzero[0][mask_pix], where_nonzero[1][mask_pix]]
                temp = mask*initial
            else:
                temp = initial

            # if EQUIV:
            #     temp = rotate(rot,b,temp)
        elif sel_ind>0:
            if sel in VALIDATION_TIMEPOINTS:
                temp = initial.clone().detach()
                for _ in range(validation_sel_min_total_loss_iter[sel]):
                    temp = pde.step(temp)
            else:
                temp = initial.clone().detach()
                for _ in range(num_steps[sel]):
                    temp = pde.step(temp)
            # if EQUIV:
            #     temp = rotate(inv_rot,inv_b,temp)
            # t2 = temp[0].permute(1,2,0)
            # t2 = rotate(rots[sel].weight,rots[sel].bias,temp)[0].permute(1,2,0)

        if REGISTER: 
            t2 = rotate(rots[sel].weight,rots[sel].bias,temp)[0].permute(1,2,0)
        elif REGISTER_VAL and sel in VALIDATION_TIMEPOINTS:
            t2 = rotate(rots[sel].weight,rots[sel].bias,temp)[0].permute(1,2,0)
        else:
            t2 = temp[0].permute(1,2,0)
        ax2[rep+1,sel_ind].imshow(t2.detach().cpu().sum(axis=-1))
        list_of_latent_tps.append(t2.detach().cpu())


    ## save list_of_latent_tps as pickle file
    with open(os.path.join(os.path.join(RESULTS_DIR, 'pkls'), OUT_PREFIX + f'_rep{rep+1}' + f'_ckpt_iter_{args.ckpt_iter-1}'  + '_list_of_latent_tps.pkl'), 'wb') as f:
        pickle.dump(list_of_latent_tps, f)


    fig2.savefig(os.path.join(os.path.join(RESULTS_DIR,'vis'),'vis_' + OUT_PREFIX + f'_rep{rep+1}' + f'_ckpt_iter_{args.ckpt_iter-1}'  + '.pdf'), transparent=True, bbox_inches='tight', dpi=100)

