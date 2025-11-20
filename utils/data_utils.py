# data_utils.py

import numpy as np
from scanpy import read_h5ad, pp
import torch
import anndata as ad

def load_data(filepath):
    x = read_h5ad(open(filepath, 'rb'))
    return x

import numpy as np
from scanpy import read_h5ad, pp
import torch

def preprocess_data(x, args, filter, preprocess_gex, mask_regions=False, mask_segments=False, mask_segments_tp=None, mask_segments_mode=None):

    ## rewrite raw counts into .X
    x.X = x.layers['counts']

    if args.preprocess_gex:
        print("preprocessing")
        x_norm = pp.normalize_total(x, copy=True)
        x = x_norm

    if preprocess_gex:
        print("log transforming")
        x_log = pp.log1p(x, copy=True)
        x = x_log

    if args.use_genes == 'all':
        if filter:
            print("filtering")
            fg_bc_high_var = args.fg_bc_high_var
            fg_bc_min_pct_cells_by_counts = args.fg_bc_min_pct_cells_by_counts
            
            ## filter genes that appear in too few cells
            if fg_bc_min_pct_cells_by_counts is not None:
                ## add metrics
                x.var['mt'] = x.var_names.str.startswith('mt-')  # annotate the group of mitochondrial genes as 'mt'
                pp.calculate_qc_metrics(x, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
                
                x.var["pct_cells_by_counts"] = x.var["n_cells_by_counts"]/x.X.shape[0] * 100

                x = x[:, x.var["pct_cells_by_counts"] > fg_bc_min_pct_cells_by_counts]

            ## filter to high var genes
            if fg_bc_high_var is not None:
                if not preprocess_gex:
                    x_norm = pp.normalize_total(x, copy=True)
                    x_log = pp.log1p(x_norm, copy=True)
                    pp.highly_variable_genes(x_log, n_top_genes=fg_bc_high_var)
                    x = x[:,x_log.var['highly_variable']]
                else:
                    pp.highly_variable_genes(x, n_top_genes=fg_bc_high_var)
                    x = x[:,x.var['highly_variable']]
    elif args.use_genes == 'tf_target':
        tf_target_keep_genes = np.unique(np.load('tf_target_keep_genes.npy', allow_pickle=True))
        x = x[:, tf_target_keep_genes]
    elif args.use_genes == "ssh_genes_v2":
        print("loading ssh genes")
        genes = np.load('/home/skambha6/chenlab/stnca/code/zebrafish_data_preprocessing/ssh_genes_v2.npy', allow_pickle=True)
        x = x[:, genes]  
    elif args.use_genes == "morphogenesis":
        print("loading morphogenesis genes")
        genes = np.load('/home/skambha6/chenlab/stnca/stnca_data/zebrafish/morphogenesis_genes.npy', allow_pickle=True)
        x = x[:, genes]
    else:
        print("Use_genes argument not recognized")
        
    ## filter out, ribosomal and mitochondrial genes
    ribo_genes = x.var_names.str.startswith(('rps', 'rpl', 'RPL', 'RPS'))
    mito_genes = x.var_names.str.startswith('mt-', 'MT-')
    remove = np.add(mito_genes, ribo_genes)
    keep = np.invert(remove)

    x = x[:,keep]


    if args.normalize == "quantile":
        print("quantile normalizing")
        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer()
        x.X = qt.fit_transform(x.X)
    elif args.normalize == "nonzero_median":
        print("nonzero median normalizing")
        # Convert to dense array if necessary
        if not isinstance(x.X, np.ndarray):
            x.X = x.X.toarray()

        # Calculate the nonzero median for each gene (set to 1 if no nonzero values to avoid divide by zero error)
        nonzero_medians = np.array([np.median(x.X[:, i][x.X[:, i] != 0]) if np.any(x.X[:, i] != 0) else 1 for i in range(x.shape[1])])

        # Normalize each gene by its nonzero median
        x.X = x.X / nonzero_medians



    print("num used genes:", x.shape[1])
    print("num cells across all tps:", x.shape[0])
    gex = x.to_df()
    coords = x.obs[['spatial_x', 'spatial_y']]
    print(x)
    tp = {}
    shift_refs = {}  # <-- store original min/max values per timepoint
    for k, v in x.obs.groupby(x.obs['time'].cat.codes).groups.items():
        if k >= 5:
            continue
        tp[k] = (coords.loc[v], torch.from_numpy(gex.loc[v].values).float())

    for k, v in tp.items():
        shift_refs[k] = (v[0]['spatial_x'].min(), v[0]['spatial_y'].max())
        v[0]['spatial_x'] = (v[0]['spatial_x'] - shift_refs[k][0])
        v[0]['spatial_y'] = (v[0]['spatial_y'] - shift_refs[k][1] + 2000)
        tp[k] = (v[0], v[1].cuda())

    ## masks a single timepoint (specified by mask_segments_tp)
    if mask_segments:
        if mask_segments_tp is None:
            print("mask_segments_tp not provided")
            return
        
        list_of_masked_segments_x_tp = []
        for k,v in x.obs.groupby(x.obs['time'].cat.codes).groups.items():
            # print(k,v)
            if k == mask_segments_tp:
                x_mask_segments_tp = x[v].copy()
                segment_tp_mask = make_segment_mask(x_mask_segments_tp, mask_segments_mode)
                x_masked_segments_tp = x_mask_segments_tp[segment_tp_mask]
                list_of_masked_segments_x_tp.append(x_masked_segments_tp)
            else:
                x_tp = x[v].copy()
                list_of_masked_segments_x_tp.append(x_tp)

        x_masked = ad.concat(list_of_masked_segments_x_tp)
    elif mask_regions:
        x_masked = x[x.obs['held_out'] == False]

    if mask_regions or mask_segments:
        gex_masked = x_masked.to_df()
        coords_masked = x_masked.obs[['spatial_x', 'spatial_y']]
        tp_masked = {}
        for k, v in x_masked.obs.groupby(x_masked.obs['time'].cat.codes).groups.items():
            if k >= 5:
                continue
            tp_masked[k] = (coords_masked.loc[v], torch.from_numpy(gex_masked.loc[v].values).float())

        for k, v in tp_masked.items():
            v[0]['spatial_x'] = (v[0]['spatial_x'] - shift_refs[k][0])
            v[0]['spatial_y'] = (v[0]['spatial_y'] - shift_refs[k][1] + 2000)
            tp_masked[k] = (v[0], v[1].cuda())

    if mask_regions or mask_segments:
        return tp, tp_masked, gex_masked
    else:
        return tp, gex 
    
def get_centroid(adata):
    x = adata.obs['spatial_x'].values
    y = adata.obs['spatial_y'].values
    return np.mean(x), np.mean(y)

def make_segment_mask(adata, mode):
    x = adata.obs['spatial_x'].values
    y = adata.obs['spatial_y'].values
    x0, y0 = get_centroid(adata)

    dx = x - x0
    dy = y - y0

    if mode == 'y>x':
        return dy > dx
    elif mode == 'y<x':
        return dy < dx
    elif mode == 'x>centroid':
        return x > x0
    elif mode == 'x<centroid':
        return x < x0
    elif mode == 'y>centroid':
        return y > y0
    elif mode == 'y<centroid':
        return y < y0
    else:
        raise ValueError(f"Unknown mode: {mode}")

def preprocess_coords(tp, size=50, yshifts=[0]*5, xshifts=[0]*5, mask=False, tp_masked=None):

    if not mask:
        zero_pos = {}
        pos = {}
        nonzero_ims = {}
        zero_ims = {}  # Initialize zero_ims for both branches
        sf = int(50 * 50 / size)
        for k in range(5):
            coo, _ = tp[k]
            img = np.zeros((size, size))
            # When size is large (e.g. 2000), sf becomes very small (e.g. 1.25)
            # This causes integer division to round down to 0, making all coordinates map to (0,0)
            # Use float division and round at the end instead
            pos[k] = np.round((size - 1) / 2 + coo.values / sf - coo.values.mean(axis=0, keepdims=True) / sf).astype(int)
            pos[k][:, 0] += yshifts[k]
            pos[k][:, 1] += xshifts[k]
            
            # Clip coordinates to valid image bounds
            pos[k][:, 0] = np.clip(pos[k][:, 0], 0, size-1)
            pos[k][:, 1] = np.clip(pos[k][:, 1], 0, size-1)
            
            img[(pos[k][:, 0], pos[k][:, 1])] += 1
            zer = np.array(np.where(img == 0)).T
            m = np.mean(img)
            nonzero_ims[k] = img.copy()
            zero_pos[k] = (zer, np.arange(len(zer)), (1 - m) / m)      
    elif mask:
        if tp_masked is None:
            print("tp_masked not provided")
            return
        zero_pos = {}
        pos = {}
        nonzero_ims = {}
        sf = int(50 * 50 / size)

        for k in range(5):
            coo_dummy, _ = tp[k]
            coo_masked, _ = tp_masked[k]

            mean_offset = coo_dummy.values.mean(axis=0, keepdims=True)
            
            # Align dummy (original)
            coords_dummy = np.round((size - 1) / 2 + (coo_dummy.values - mean_offset) / sf).astype(int)
            coords_dummy[:, 0] += yshifts[k]
            coords_dummy[:, 1] += xshifts[k]
            coords_dummy = np.clip(coords_dummy, 0, size - 1)
            
            img_dummy = np.zeros((size, size))
            img_dummy[coords_dummy[:, 0], coords_dummy[:, 1]] += 1
            
            zer = np.array(np.where(img_dummy == 0)).T
            m = np.mean(img_dummy)
            zero_pos[k] = (zer, np.arange(len(zer)), (1 - m) / m)
            
            # Align masked
            coords_masked = np.round((size - 1) / 2 + (coo_masked.values - mean_offset) / sf).astype(int)
            coords_masked[:, 0] += yshifts[k]
            coords_masked[:, 1] += xshifts[k]
            coords_masked = np.clip(coords_masked, 0, size - 1)

            img = np.zeros((size, size))
            img[coords_masked[:, 0], coords_masked[:, 1]] += 1
            nonzero_ims[k] = img.copy()
            pos[k] = coords_masked

    return zero_pos, pos, nonzero_ims

def preprocess_coords_zero_masked(tp, size=50, yshifts=[0]*5, xshifts=[0]*5, mask=False, tp_masked=None, mask_segments_tp=None, mask_mode=None):

    if not mask:
        zero_pos = {}
        pos = {}
        nonzero_ims = {}
        zero_ims = {}  # Initialize zero_ims for both branches
        sf = int(50 * 50 / size)
        for k in range(5):
            coo, _ = tp[k]
            img = np.zeros((size, size))
            # When size is large (e.g. 2000), sf becomes very small (e.g. 1.25)
            # This causes integer division to round down to 0, making all coordinates map to (0,0)
            # Use float division and round at the end instead
            pos[k] = np.round((size - 1) / 2 + coo.values / sf - coo.values.mean(axis=0, keepdims=True) / sf).astype(int)
            pos[k][:, 0] += yshifts[k]
            pos[k][:, 1] += xshifts[k]
            
            # Clip coordinates to valid image bounds
            pos[k][:, 0] = np.clip(pos[k][:, 0], 0, size-1)
            pos[k][:, 1] = np.clip(pos[k][:, 1], 0, size-1)
            
            img[(pos[k][:, 0], pos[k][:, 1])] += 1
            zer = np.array(np.where(img == 0)).T
            m = np.mean(img)
            nonzero_ims[k] = img.copy()
            zero_pos[k] = (zer, np.arange(len(zer)), (1 - m) / m)

    if mask:
        if tp_masked is None or mask_segments_tp is None or mask_mode is None:
            print("tp_masked, mask_segments_tp, or mask_mode not provided")
            return
        zero_pos = {}
        pos = {}
        nonzero_ims = {}
        sf = int(50 * 50 / size)

        for k in range(5):
            coo_dummy, _ = tp[k]
            coo_masked, _ = tp_masked[k]

            mean_offset = coo_dummy.values.mean(axis=0, keepdims=True)
            
            # Align dummy (original)
            coords_dummy = np.round((size - 1) / 2 + (coo_dummy.values - mean_offset) / sf).astype(int)
            coords_dummy[:, 0] += yshifts[k]
            coords_dummy[:, 1] += xshifts[k]
            coords_dummy = np.clip(coords_dummy, 0, size - 1)
            
            img_dummy = np.zeros((size, size))
            img_dummy[coords_dummy[:, 0], coords_dummy[:, 1]] += 1
            
            zer = np.array(np.where(img_dummy == 0)).T
            
            # Filter zero positions based on mask_mode
            if mask_mode is not None and mask_segments_tp == k:
                # Get centroid of the embryo (where cells are located)
                y0 = coords_dummy[:, 1].mean()
                x0 = coords_dummy[:, 0].mean()

                print("y0, x0", y0, x0)
                
                # Apply mask filter to zero positions
                if mask_mode == 'y>x':
                    # Keep only positions where dy > dx (above the diagonal)
                    mask_filter = (zer[:, 1] - y0) > (zer[:, 0] - x0)
                elif mask_mode == 'y<x':
                    # Keep only positions where dy < dx (below the diagonal)
                    mask_filter = (zer[:, 1] - y0) < (zer[:, 0] - x0)
                elif mask_mode == 'x<centroid':
                    mask_filter = zer[:, 0] < x0
                elif mask_mode == 'x>centroid':
                    mask_filter = zer[:, 0] > x0
                elif mask_mode == 'y<centroid':
                    mask_filter = zer[:, 1] < y0
                elif mask_mode == 'y>centroid':
                    mask_filter = zer[:, 1] > y0
                else:
                    mask_filter = np.ones(len(zer), dtype=bool)
                
                zer = zer[mask_filter]
            
            m = np.mean(img_dummy)
            zero_pos[k] = (zer, np.arange(len(zer)), (1 - m) / m)
            
            # Align masked
            coords_masked = np.round((size - 1) / 2 + (coo_masked.values - mean_offset) / sf).astype(int)
            coords_masked[:, 0] += yshifts[k]
            coords_masked[:, 1] += xshifts[k]
            coords_masked = np.clip(coords_masked, 0, size - 1)

            img = np.zeros((size, size))
            img[coords_masked[:, 0], coords_masked[:, 1]] += 1
            nonzero_ims[k] = img.copy()
            pos[k] = coords_masked

    return zero_pos, pos, nonzero_ims

def set_timepoints(args):

    # default behavior
    if args.chosen_tps is None:
        ## val_samples set to 0 by default so this if statement will always execute
        if args.val_samples == 0:
            CHOSEN_TIMEPOINTS = [0,1,2,3,4]
            VALIDATION_TIMEPOINTS = []
        elif args.val_samples == 1:
            CHOSEN_TIMEPOINTS = [0,1,2,4]
            VALIDATION_TIMEPOINTS = [3]
        elif args.val_samples == 2:
            CHOSEN_TIMEPOINTS = [0,3,4]
            VALIDATION_TIMEPOINTS = [1,2]
        elif args.val_samples == 3:
            CHOSEN_TIMEPOINTS = [0,4]
            VALIDATION_TIMEPOINTS = [1,2,3]

        ALL_TIMEPOINTS = [0,1,2,3,4]
    else:
        ## check if chosen timepoints are provided as a list or a string
        if isinstance(args.chosen_tps, str):
            CHOSEN_TIMEPOINTS = [int(t) for t in args.chosen_tps.split(',')]
        else:
            CHOSEN_TIMEPOINTS = args.chosen_tps

        print("chosen timepoints provided")
        print(CHOSEN_TIMEPOINTS)
        


        if args.val_tps is not None:
            if isinstance(args.val_tps, str):
                VALIDATION_TIMEPOINTS = [int(t) for t in args.val_tps.split(',')]
            else:
                VALIDATION_TIMEPOINTS = [args.val_tps]
            print("validation timepoints provided")
            print(VALIDATION_TIMEPOINTS)
        else:
            print("no validation timepoints provided, skipping interpolation")
            VALIDATION_TIMEPOINTS = []

        ALL_TIMEPOINTS = list(set(CHOSEN_TIMEPOINTS + VALIDATION_TIMEPOINTS))
        ## sort ALL_TIMEPOINTS in ascending order
        ALL_TIMEPOINTS = sorted(ALL_TIMEPOINTS)
       

    print(f"Chosen timepoints: {CHOSEN_TIMEPOINTS}")
    print(f"Validation timepoints: {VALIDATION_TIMEPOINTS}")
    print(f"All timepoints: {ALL_TIMEPOINTS}")

    return CHOSEN_TIMEPOINTS, VALIDATION_TIMEPOINTS, ALL_TIMEPOINTS

def set_num_steps(CHOSEN_TIMEPOINTS, args):

    if args.timescale_type == 'bio':
        tps = [5.25, 10, 12, 18, 24]

        rel_tps = []

        chosen_tp_0 = tps[CHOSEN_TIMEPOINTS[0]]
        for i in range(len(CHOSEN_TIMEPOINTS)):
            chosen_tp = tps[CHOSEN_TIMEPOINTS[i]]
            rel_tps.append(chosen_tp - chosen_tp_0)

        TIMESCALE = args.timescale

        num_steps = {CHOSEN_TIMEPOINTS[i] : TIMESCALE*int(rel_tps[i]) for i in range(1, len(CHOSEN_TIMEPOINTS))}
    elif args.timescale_type == 'uniform':
         TIMESCALE = args.timescale
         num_steps = {CHOSEN_TIMEPOINTS[i] : TIMESCALE*CHOSEN_TIMEPOINTS[i] for i in range(1, len(CHOSEN_TIMEPOINTS))}

    return num_steps
