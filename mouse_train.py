import argparse
import os

# Initialize argument parser
parser = argparse.ArgumentParser(description="Parse hyperparameters for training.")

# Add arguments

parser.add_argument("--LATENT_DIM", type=int, required=True, help="Latent dimension size.")
parser.add_argument("--OUT_DIR", type=str, required=True, help="Output directory path.")
parser.add_argument("--TIMESCALE", type=int, required=True, help="Timescale")
parser.add_argument("--MIX", type=str,required=True, help="Mixer format")

parser.add_argument("--MIX_PRIOR", type=str, help="Pickle file with pandas df of features, rownames should be genes")
parser.add_argument("--MIX_HIDDEN_DIM", type=int, default=200, help="Hidden dim for MLP if used")

parser.add_argument("--MAX_ITS", type=int, default=30000, help="Maximum iterations.")
parser.add_argument("--USE_GRADS", type=bool, default = False, help="Whether to use gradients (True/False).")
parser.add_argument("--USE_MIDDLE", type=bool, default = False, help="Whether to use intermediate sample 2")
parser.add_argument("--VAL_SIZE", type=int, default = 8, help="Radius of held out spatial regions")
parser.add_argument("--LEARN_START", type=bool, default=False, help="Learn initial condition or start from noise")

parser.add_argument("--BATCH_SIZE", type=int, default=512, help="Batch size for training.")

parser.add_argument("--DROPOUT", type = bool, default = False, help="use dropout Dropout rate.")
parser.add_argument("--DEVICE", type=int, default=0, help="Computation device (e.g., 'cpu' or 'cuda').")
parser.add_argument("--ZERO_SCALE", type=float, default=5.0, help="Scale factor for zero loss")
parser.add_argument("--VAL_EVERY", type=int, default=100, help="val every")

parser.add_argument("--SAVE_EVERY", type=int, default=5000, help="save every")
parser.add_argument("--OVERWRITE", type=bool, default = False, help="overwrite out directory if already exists")
parser.add_argument("--SEED", type=int, default = 1238882, help="torch Manual Seed = np random seed")




# Parse arguments
args = parser.parse_args()

# Load as variables
MAX_ITS = args.MAX_ITS
LATENT_DIM = args.LATENT_DIM
USE_GRADS = args.USE_GRADS
USE_MIDDLE = args.USE_MIDDLE
ZERO_SCALE = args.ZERO_SCALE
LEARN_START = args.LEARN_START
MIX = args.MIX
BATCH_SIZE = args.BATCH_SIZE
TIMESCALE = args.TIMESCALE
DROPOUT = args.DROPOUT
OUT_DIR = args.OUT_DIR
DEVICE = args.DEVICE
VAL_EVERY = args.VAL_EVERY
MIX_PRIOR = args.MIX_PRIOR
MIX_HIDDEN_DIM = args.MIX_HIDDEN_DIM
SAVE_EVERY = args.SAVE_EVERY
mix = MIX
VAL_SIZE = args.VAL_SIZE
SEED = args.SEED

OVERWRITE = args.OVERWRITE
if os.path.exists(OUT_DIR):
    if OVERWRITE:
        print(f"Overwriting existing directory: {OUT_DIR}")

    else:
        print(f"Output directory {OUT_DIR} already exists. Use --OVERWRITE True to overwrite.")
        exit
else:
    # Create the directory
    os.mkdir(OUT_DIR)

import numpy as np
import pandas as pd
from scanpy import read_h5ad
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import cdist


from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from scipy.cluster.hierarchy import fcluster

import wandb

#prepare the gene expression dataset tensors
x = read_h5ad('Mouse_embryo_all_stage.h5ad')
gene_info = pd.read_table('mm10.ncbiRefSeq.gtf',comment='#',header = None)
gene_info = gene_info[gene_info[2]=='transcript']
feats = pd.read_pickle('hg38_mm10_1k_features.pkl')
gene_info['symbol'] = [a.split(';')[0].split('"')[-2] for a in gene_info[8]]
gene_info['id'] = [a.split(';')[1].split('"')[-2].split('.')[0] for a in gene_info[8]]
feat_ids = pd.Series(['_'.join(a[:2]) for a in feats.index.str.split('_')],index = feats.index)
common = set(feat_ids.values).intersection(gene_info['id'])
mm10_feats = feats.loc[feat_ids[feat_ids.isin(common)].index.values]
gene_dict = gene_info[gene_info['id'].isin(common)][['id','symbol']].drop_duplicates().set_index('id')['symbol'].to_dict()
mm10_feats['gene_symbol'] = feat_ids.loc[mm10_feats.index].map(gene_dict)
mm10_avs = mm10_feats.groupby('gene_symbol').mean()
sel_feat = sorted(set(mm10_avs.index).intersection(x.var.index))
gex = x[:,sel_feat].to_df()
coords = pd.DataFrame(np.array([[int(i) for i in a.split('-')[0].split('_')] for a in x.obs.index]),index = x.obs.index)
gene_features = mm10_avs.loc[sel_feat].values

if MIX in ['prior','prior_shuf','prior_att']:
    np.random.seed(SEED)
    prior_features = pd.read_pickle(MIX_PRIOR).loc[sel_feat].values
    if 'shuf' in MIX:
        shuf = np.arange(len(prior_features))
        prior_features = prior_features[shuf]


run=wandb.init(config=args)
torch.cuda.set_device(DEVICE)
tp = {}
for k,v in x.obs.groupby(x.obs['timepoint'].cat.codes).groups.items():
    if k>=4:
        continue
    tp[k] = (coords.loc[v], torch.from_numpy(gex.loc[v].values).float())
    print(k)

for k,v in tp.items():
    tp[k] = (v[0],v[1].cuda())

zero_pos = {}
pos = {}
nonzero_ims = {}

yshifts = [0,0,0,0,1]
xshifts = [0,0,0,0,1]

size = 400    
sf = 1
for k in range(4):
    coo, _ = tp[k]
    img = np.zeros((size,size))
    pos[k] = ((size-1)//2 + coo.values//sf - coo.values.mean(axis = 0,keepdims = True)/sf).astype(int)
    pos[k][:,0]+= yshifts[k]
    pos[k][:,1]+=xshifts[k]
    img[(pos[k][:,0], pos[k][:,1])] += 1
    zer = np.array(np.where(img==0)).T
    m = np.mean(img)
    nonzero_ims[k] = img.copy()
    
    ratio = (1-m)/m
    zero_pos[k] = (zer, np.arange(len(zer)),(1-m)/m)


#remove held out regions for training/testing set
idxs = {i : np.arange(len(tp[i][0])) for i in tp}
np.random.seed(60)
val = pos[3][np.random.choice(np.arange(pos[3].shape[0]), 12)]


dist_to_val = cdist(pos[3],val)

train_pos = pos[3][np.where(dist_to_val.min(axis=1)>VAL_SIZE)]
train_ind = np.where(dist_to_val.min(axis=1)>VAL_SIZE)[0]
val_ind = {i:np.where(dist_to_val[:,i]<=VAL_SIZE)[0] for i in range(12)}
val_border = {i:np.where((dist_to_val[:,i]>VAL_SIZE) & (dist_to_val[:,i]<=(VAL_SIZE+1)))[0] for i in range(12)}
val_pos = {i:pos[3][np.where(dist_to_val[:,i]<=VAL_SIZE)] for i in range(12)}

new_pos = {k:v.copy() for k,v in pos.items()}
new_pos[3] = train_pos
new_idxs = {k:v.copy() for k,v in idxs.items()}
new_idxs[3] = train_ind

#auxiliar functions to rotate img tensor and generate rotation matries
from torch import nn
from torch.nn.utils.parametrizations import orthogonal

def rotate(W,b, image):
    theta = torch.cat([W,b[:,None]], dim = -1)[None]
    affine_grid = torch.nn.functional.affine_grid(theta, size = image.shape)
    return torch.nn.functional.grid_sample(image, affine_grid)

def rotmat(degs):
    th = 2*np.pi*degs/360
    return torch.tensor([[np.cos(th), -np.sin(th) ],[np.sin(th), np.cos(th) ] ]).float()
ref_x = torch.tensor([[-1,0],[0,1]]).float()


ref_x = torch.tensor([[-1,0],[0,1]]).float()

rot0_inits = [torch.eye(2), torch.eye(2), 0.88*rotmat(17)@ref_x, torch.eye(2), torch.eye(2)]
rot4_inits = [torch.eye(2), torch.eye(2), 1.2*(ref_x@rotmat(90)), torch.eye(2),torch.eye(2)]



mod0 = nn.Linear(2,2).cuda()
mod4 = nn.Linear(2,2).cuda()
sel = 2
mod0.weight.data = rot0_inits[sel].cuda().clone().detach().requires_grad_(True)
mod0.bias.data = torch.zeros(2,dtype = torch.float,device = 'cuda',requires_grad = True)
mod4.weight.data = rot4_inits[sel].cuda().clone().detach().requires_grad_(True)
mod4.bias.data = torch.zeros(2,dtype = torch.float,device = 'cuda',requires_grad = True)



im4 = torch.from_numpy(nonzero_ims[3]).float().cuda()[None,None]
im0 = torch.from_numpy(nonzero_ims[1]).float().cuda()[None,None]
post_0 = rotate(mod0.weight,mod0.bias,im0)[0,0].detach().cpu().numpy()
post_4 = rotate(mod4.weight,mod4.bias,im4)[0,0].detach().cpu().numpy()

#helper to compute correlations
@torch.no_grad()
def vec_corr(a,b,axis=0):
    na = a-a.mean(axis=axis,keepdim=True)
    nb = b-b.mean(axis=axis,keepdim=True)
    na/=(1e-5+torch.norm(na, dim = axis, keepdim=True))
    nb/=(1e-5+torch.norm(nb, dim = axis, keepdim=True))
    return (na*nb).sum(axis = axis)
    
print('loaded data, made splits')
from models.mixers import NMFMixer, ReLUMLPMixer, SeqPredMLPMixer,SeqAttentionMLPMixer
from models.pdes import ReactionDiffusionPDE

torch.manual_seed(SEED)
print('seeded {}'.format(SEED))

def decode(t2ij, mixer):
    return mixer.decoder(t2ij)

def build_init(nonzero,start):
    return nonzero[None,None]*(start**2 ) # note in orig there is square!

out_dim = gex.shape[1]


#initialize the noise and initial conditions
if DROPOUT:
    noise = torch.nn.Dropout(0.2)
else:
    noise = torch.nn.Identity()
if LEARN_START: 
    LOSS_WEIGHTS = {k: 1.0 for k in [1,2,3]}
else:
    LOSS_WEIGHTS = {k: 1.0 for k in [1,2,3]}
    LOSS_WEIGHTS[1]=0


#initialize mixer
if mix =='relu':
    mixer = ReLUMLPMixer(LATENT_DIM,out_dim,hidden_dim = MIX_HIDDEN_DIM).cuda()
elif mix =='seq':
    mixer = SeqPredMLPMixer(LATENT_DIM, out_dim,gene_features = gene_features,syntax_weight=0.01,hidden_dim = MIX_HIDDEN_DIM).cuda()
elif 'prior_att' in mix:
    mixer =SeqAttentionMLPMixer(LATENT_DIM, out_dim,gene_features = gene_features,syntax_weight=0.01,hidden_dim = MIX_HIDDEN_DIM).cuda()
elif mix =='prior_shuf':
    # np.random.seed(SEED)
    # shuf = np.arange(len(gene_features))
    # np.random.shuffle(shuf) 
    # commenting this out since we have reimplemented it when reading in the prior features
    mixer = SeqPredMLPMixer(LATENT_DIM, out_dim,gene_features = prior_features,syntax_weight=0.01,hidden_dim = MIX_HIDDEN_DIM).cuda()
elif mix =='prior':
    mixer = SeqPredMLPMixer(LATENT_DIM, out_dim,gene_features = prior_features,syntax_weight=0.01,hidden_dim = MIX_HIDDEN_DIM).cuda()
elif mix =='nmf':
    mixer = NMFMixer(LATENT_DIM,out_dim).cuda()
else:
    print('quitting, valid mixer not specified')
    exit


#initialize registrations
rots = []


rot0_inits = [torch.eye(2),torch.eye(2), 0.78*rotmat(15)@ref_x, 0.62*rotmat(90), 0.5*ref_x@rotmat(150)]


for sel in [0,1,2,3,4]:
    mod = nn.Linear(2,2).cuda()
    mod.weight.data = rot0_inits[sel].cuda().clone().detach().requires_grad_(True)
    mod.bias.data = torch.zeros(2,dtype = torch.float,device = 'cuda',requires_grad = True)
    im = torch.from_numpy(nonzero_ims[1]).float().cuda()[None,None]
    post = rotate(mod.weight,mod.bias,im)[0,0].detach().cpu().numpy()
    rots.append(mod)
    rots =nn.ModuleList(rots).cuda()



#initialize initial conditions
init  = torch.rand(size,size,LATENT_DIM,device = 'cuda',dtype = torch.float)
init[(zero_pos[1][0][:,0], zero_pos[1][0][:,1])] = 0
start = init.permute(2,0,1).detach().clone().requires_grad_(True)
nonzero = torch.from_numpy(nonzero_ims[1]).float().cuda()
where_nonzero = np.where(nonzero_ims[1])
num_nonzero = len(where_nonzero[1])


#initialize PDE
pde = ReactionDiffusionPDE(latent_dim = LATENT_DIM, use_grads = USE_GRADS, diffusion_type = 'variable').cuda()

torch.nn.init.normal_(pde.reaction[0].weight, 0.01)
torch.nn.init.zeros_(pde.ldiff)

pde_opt = torch.optim.Adam(list(pde.parameters()) +[start]  ,lr = 0.001) 
opt = torch.optim.Adam(list(mixer.parameters()),lr = 0.001)
rot_opt = torch.optim.Adam(list(rots.parameters()) ,lr = 0.001) #do not learn scale

#initialize loss collection dicts
t2s = {}
ge_losses = {}
zero_losses = {}
enc_losses = {}
val_loss = {}
val_corr = {}
print('initialized everything')

if USE_MIDDLE:
    sel_samps = [1,2,3]
else:
    sel_samps = [1,3]
#training loop
np.random.seed(SEED)
for its in tqdm(range(MAX_ITS)):
    imgs = []
    sum_imgs = []
    ims = []
    loss = 0
    bias_reg = 0


    mixer.build_params()
    pde.build_params()
    sf = 1
    for sel in sel_samps:
        #sample coordinates with gex
        co, ge = tp[sel]
        samp = np.random.choice(new_idxs[sel], BATCH_SIZE)
        coord = pos[sel][samp]
        i,j = (coord[:,0],coord[:,1])

        #sample zero coordinates 
        zer, zer_inds,ratio = zero_pos[sel]
        zer_samp = np.random.choice(zer_inds, int(ratio*BATCH_SIZE))
        zer_coord = zer[zer_samp]
        zer_i,zer_j = (zer_coord[:,0], zer_coord[:,1])


        if sel ==1:
            #set up the initial condition
            initial = build_init(nonzero,start)

            #collect the latent image
            t2 = initial[0].permute(1,2,0)
            if its%VAL_EVERY==0:
                temp = initial # this is not noising when validating
            else:
                temp = noise(initial)
            


        if sel in [2,3]:
            #step the dynamics
            for _ in range(TIMESCALE):
                temp = pde.step(temp)
            
            # affine transformation
            t2 = rotate(rots[sel].weight,rots[sel].bias,temp)[0].permute(1,2,0)

            #regularize the bias in the affine transformation
            bias_reg += 10*torch.square(rots[sel].bias).sum()


        #compute loss for sampled gex points
        ge_loss = torch.square((sf*decode(t2[(i,j)],mixer) - ge[samp])).mean() 

        #compute loss for zero points
        zero_loss = ZERO_SCALE*(torch.square(decode(t2[(zer_i,zer_j)],mixer) ).mean() + torch.square(decode(t2[(zer_i,zer_j)],mixer) ).mean())
        
        #weight each sample's loss by the loss weights
        loss+=LOSS_WEIGHTS[sel]*(ge_loss + zero_loss)
        
        ge_losses[sel]= ge_loss.item()
        zero_losses[sel] = zero_loss.item()
        
        ims.append(t2.detach().cpu().numpy())
        t2s[sel] = t2.detach().cpu().numpy()
    
    
    # step the 
    opt.zero_grad()
    pde_opt.zero_grad()
    rot_opt.zero_grad() 

    mixer_reg = mixer.reg()
    pde_reg = pde.reg()
    (loss+  mixer_reg + pde_reg).backward()
    
    
    pde_opt.step()
    opt.step()
    
    if np.random.rand()<0.9:
        #sometimes don't step the registrations 
        rot_opt.step()

    #validation
    with torch.no_grad():
        if (its%VAL_EVERY) ==0:
            #t2 is always going to be at the *last* time point here
            val_losses = {}
            val_corrs = {}
            for k in val_pos:
                vpi, vpj = (val_pos[k][:,0],val_pos[k][:,1])
                vpsamp = val_ind[k]
                val_losses[k] = torch.square((sf*decode(t2[(vpi,vpj)],mixer) - ge[vpsamp])).mean()
                val_corrs[k] = vec_corr(sf*decode(t2[(vpi,vpj)],mixer), ge[vpsamp])

            val_loss_log = {'val_loss/{}'.format(k):v for k,v in val_losses.items()}
            val_corr_log = {'val_cors{}/{}'.format(thresh,k):(val_corrs[k]>thresh).sum() for k in val_corrs for thresh in [0.1,0.2,0.3,0.4,0.5]}
            wandb.log(val_loss_log,step = its)
            wandb.log(val_corr_log, step = its)
            wandb.log({'val_cors_hist/{}'.format(k): wandb.Histogram(v.cpu().numpy()) for k,v in val_corrs.items()},step = its)
            wandb.log({'train_gex_tp/{}'.format(k):v for k,v in ge_losses.items()},step = its)
            wandb.log({'train_zero_tp/{}'.format(k):v for k,v in ge_losses.items()},step = its)


    if ((1+its)%SAVE_EVERY)==0:
        torch.save({'mix':mixer.state_dict(), 'pde': pde.state_dict(), 'rots':rots.state_dict(),'start':start}, os.path.join(OUT_DIR, 'checkpoint_{}'.format(its)))