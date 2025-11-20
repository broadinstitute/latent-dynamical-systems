# config.py

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--out_prefix', type=str, required=True)
    parser.add_argument('--scale', type = str, default = 'small')
    parser.add_argument('--timescale', type = int, default = 5)
    parser.add_argument('--timescale_type', type = str, default = 'uniform')
    parser.add_argument('--latent_dim', type = int, default = 20)
    parser.add_argument('--hidden_dim', type = int, default = 0)
    parser.add_argument('--tf_dim', type = int, default = None)
    parser.add_argument('--target_dim', type = int, default = None)
    parser.add_argument('--equiv', type = bool)
    parser.add_argument('--injury', type = bool)
    parser.add_argument('--donot_register', type = bool)
    parser.add_argument('--donot_register_val', type = bool)
    parser.add_argument('--donot_use_grads', type = bool)
    parser.add_argument('--pde', type = str, default = 'ReactionDiffusion')
    parser.add_argument('--mix', type = str, default = 'nmf')
    parser.add_argument('--overwrite', type = bool)
    parser.add_argument('--chosen_tps', type=str, default=None, help='Sequence of ints or string')
    parser.add_argument('--val_tps', type=str, default=None, help='Sequence of ints or string')
    parser.add_argument('--val_samples', type = int, default = 0)
    parser.add_argument('--val_every', type = int, default = 1000)
    parser.add_argument('--max_its', type = int, default = 10000)
    parser.add_argument('--syn_weight', type = float, default = 1e-4)
    parser.add_argument('--tag', type = str)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--diffusion_type', type = str, default = 'uniform')
    parser.add_argument('--real_or_debug', type = str, default = 'real')
    parser.add_argument('--donot_learn_start', type = bool)
    parser.add_argument('--donot_fit_gex0', type = bool)
    parser.add_argument('--dropout', type = bool)
    parser.add_argument('--zero_scale', type = float, default = 5.0)
    parser.add_argument('--val_reg_type', type = str, default = 'orthogonal')
    parser.add_argument('--reg_scale', type = float, default = 0.01)
    parser.add_argument('--mixer_reg', type = float, default = 1e-5) ## only used for nmf mixers
    parser.add_argument('--initialize_mixer', type = bool, default = False) # only used for nmf mixers 
    parser.add_argument('--mixer_dropout_prob', type = float, default = 0.0) #3 only used for mlp mixers 
    parser.add_argument('--pde_reg_weight', type = float, default = 0)
    parser.add_argument('--val_time_offsets', type = int, default = 0)
    parser.add_argument('--use_genes', type = str, default = 'all')
    parser.add_argument('--preprocess_gex', type = str)
    parser.add_argument('--normalize', type = str)
    parser.add_argument('--filter', type = str)
    parser.add_argument('--fg_bc_min_pct_cells_by_counts', type = int, default = 1)
    parser.add_argument('--fg_bc_high_var', type = int, default = None)
    parser.add_argument('--learn_baseline_factors', type = bool)
    parser.add_argument('--ckpt_iter', type = int, default = 9999) ## only for evaluation, should be <= max its in train
    parser.add_argument('--num_reps', type = int, default = 1) ## only for evaluation
    parser.add_argument('--mask_mode', type = str, default = 'y>x') ## only for train_zebrafish_interpolating_segments.py
    parser.add_argument('--inverse_mask_mode', type = str, default = 'y<x') ## only for evaluate_zebrafish_interpolating_segments.py
    parser.add_argument('--mask_tp', type = int, default = 3) ## only for train_zebrafish_interpolating_segments.py, evaluate_zebrafish_interpolating_segments.py
   

    args = parser.parse_args()
    return args
