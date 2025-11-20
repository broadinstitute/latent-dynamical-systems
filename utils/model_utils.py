# model_utils.py

import torch
from torch import nn
from torch.nn.utils.parametrizations import orthogonal
from models.mixers import (NMFMixer, NMFDecomposedMixer, MLPMixer, SeqAttentionMLPMixer, 
                           SeqKernelNMFMixer, SeqKernelMLPMixer, 
                           SeqPredNMFMixer, SeqPredMLPMixer, 
                           GeneAttentionMLPMixer)
from models.pdes import ReactionDiffusionPDE

def get_mixer(args, kwargs, gex_shape):
    recon_dim = args.latent_dim - args.hidden_dim
    if args.mix == 'nmf':
        return NMFMixer(latent_dim=recon_dim, out_dim=gex_shape[1], **kwargs).cuda()
    elif args.mix == 'nmf_decomposed':
        return NMFDecomposedMixer(recon_dim, args.tf_dim, args.target_dim, **kwargs).cuda()
    elif args.mix == 'mlp':
        return MLPMixer(latent_dim=recon_dim, hidden_dim = recon_dim, out_dim=gex_shape[1], nonlinearity = 'celu', dropout_prob=args.mixer_dropout_prob, **kwargs).cuda()
    elif args.mix == 'mlp_complex':
        return MLPMixer(latent_dim=recon_dim, out_dim=gex_shape[1], nonlinearity = 'celu', **kwargs).cuda()
    elif args.mix == 'seq_att_mlp':
        return SeqAttentionMLPMixer(latent_dim=recon_dim, out_dim=gex_shape[1], **kwargs).cuda()
    elif args.mix == 'seq_nmf':
        return SeqKernelNMFMixer(latent_dim=recon_dim, out_dim=gex_shape[1], **kwargs).cuda()
    elif args.mix == 'seq_mlp':
        return SeqKernelMLPMixer(latent_dim=recon_dim, out_dim=gex_shape[1], **kwargs).cuda()
    elif args.mix == 'seq_pred_nmf':
        return SeqPredNMFMixer(latent_dim=recon_dim, out_dim=gex_shape[1], **kwargs).cuda()
    elif args.mix == 'seq_pred_mlp':
        return SeqPredMLPMixer(latent_dim=recon_dim, out_dim=gex_shape[1], **kwargs).cuda()
    elif args.mix == 'tf_att_mlp':
        return GeneAttentionMLPMixer(latent_dim=recon_dim, out_dim=gex_shape[1], **kwargs).cuda()
    else:
        raise ValueError("Invalid mixer option")

def get_pde(args):
    if args.donot_use_grads:
        use_grads = False
    else:
        use_grads = True
    if args.pde == 'ReactionDiffusion':
        return ReactionDiffusionPDE(latent_dim=args.latent_dim, use_grads=use_grads, diffusion_type=args.diffusion_type, reg_weight=args.pde_reg_weight).cuda()
    # elif args.pde == 'Convolutional':
    #     return ConvolutionalPDE(latent_dim=args.latent_dim, use_grads=use_grads, diffusion_type=args.diffusion_type, reg_weight=args.pde_reg_weight).cuda()
    # elif args.pde == 'Reaction':
    #     return ReactionPDE(latent_dim=args.latent_dim, use_grads=use_grads, reg_weight=args.pde_reg_weight).cuda()
    # elif args.pde == 'ReactionDiffusion_InteractionRegularized':
    #     return ReactionDiffusionPDE(latent_dim=args.latent_dim, use_grads=use_grads, diffusion_type=args.diffusion_type, reg_weight=args.pde_reg_weight).cuda()
    # elif args.pde == 'ReactionDiffusionBilinear':
    #     return ReactionDiffusionBilinearPDE(latent_dim=args.latent_dim, use_grads=use_grads, diffusion_type=args.diffusion_type, reg_weight=args.pde_reg_weight).cuda()
    # elif args.pde == 'ReactionDiffusionTwoLayer':
    #     return ReactionDiffusion_TwoLayer_PDE(latent_dim=args.latent_dim, use_grads=use_grads, diffusion_type=args.diffusion_type, reg_weight=args.pde_reg_weight).cuda()
    # elif args.pde == 'ReactionDiffusionBilinearTwoLayer':
    #     return ReactionDiffusionBilinear_TwoLayer_PDE(latent_dim=args.latent_dim, use_grads=use_grads, diffusion_type=args.diffusion_type, reg_weight=args.pde_reg_weight).cuda()
    # elif args.pde == 'ReactionDiffusionFourLayer':
    #     return ReactionDiffusion_FourLayer_PDE(latent_dim=args.latent_dim, use_grads=use_grads, diffusion_type=args.diffusion_type, reg_weight=args.pde_reg_weight).cuda()

def initialize_models(args, gex_shape, kwargs):
    pde = get_pde(args)
    mixer = get_mixer(args, kwargs, gex_shape)
    return pde, mixer

def initialize_rotations(num_timepoints):
    rot0_inits = [torch.eye(2) for _ in range(num_timepoints)]
    
    rots = []
    for sel in range(num_timepoints):
        mod0 = nn.Linear(2, 2).cuda()
        mod0.weight.data = rot0_inits[sel].cuda().clone().detach().requires_grad_(True)
        mod0.bias.data = torch.zeros(2, dtype=torch.float, device='cuda', requires_grad=True)
        rots.append(mod0)
    return nn.ModuleList(rots).cuda()

def set_scale(scale):

    if scale=='medium':
        yshifts = [0,0,0,0,14]
        xshifts = [10,10,10,10,0]
        size = 100
    elif scale=='small':
        yshifts = [-2,0,-1,-2,4]
        xshifts = [8,6,5,4,0]
        size = 50
    elif scale=='big':
        xshifts = [15,15,15,15,0]
        yshifts = [0,3,0,0,21]
        size = 150
    elif scale=='true':
        yshifts = [0,0,0,0,280]
        xshifts = [200,200,200,200,0]
        size = 2000
    else:
        raise ValueError("Scale not recognized")

    return size, yshifts, xshifts