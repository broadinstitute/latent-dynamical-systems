import torch
from torch import nn
from torch.nn.utils.parametrizations import orthogonal
import numpy as np
import argparse
import os
import numpy as np
import pickle
import wandb
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scanpy import pp
import matplotlib.gridspec as gridspec
import sys
from scipy.stats import pearsonr
import seaborn as sns
from IPython.display import clear_output, display
import anndata as ad


from models.mixers import (NMFMixer, MLPMixer, 
                           GeneAttentionMLPMixer)
from models.pdes import (ReactionDiffusionPDE)

from utils.data_utils import load_data, preprocess_coords
from utils.model_utils import set_scale

from utils.config import get_args
from utils.data_utils import load_data, preprocess_data, preprocess_coords, set_timepoints, make_segment_mask 
from utils.model_utils import initialize_models, initialize_rotations, set_scale
from utils.augmentations import get_gginv, masker
from utils.validation import rotate
from utils.validation import get_evals, compute_batched_ge_loss_eval, compute_batched_zer_loss_eval, ge_loss, zer_loss

# Import attribution methods
from captum.attr import InputXGradient, DeepLift
import gseapy as gp
import pandas as pd


def load_pretrained_model(ckpt_path, config, device='cuda'):
    """
    Load a pretrained PDE model from checkpoint.
    
    This function loads a complete pretrained model consisting of a mixer network and 
    a PDE (Partial Differential Equation) component from a saved checkpoint file.
    """
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Extract model configuration from checkpoint
    latent_dim = config['latent_dim']
    recon_dim = latent_dim  # No hidden dimension for this model
    
    # Create mixer model
    if config['mix'] == "mlp":
        mixer = MLPMixer(latent_dim=recon_dim, out_dim=config['out_dim']).to(device)
    elif config['mix'] == "mlp_complex":
        mixer = MLPComplexMixer(latent_dim=recon_dim, out_dim=config['out_dim']).to(device)
    elif config['mix'] == "nmf":
        mixer = NMFMixer(latent_dim=recon_dim, out_dim=config['out_dim']).to(device)
    else:
        raise ValueError(f"Unsupported mixer type: {config['mix']}")
    
    # Create PDE model with correct constructor
    pde = ReactionDiffusionPDE(
        latent_dim=latent_dim,
        diffusion_type=config['diffusion_type'],
        use_grads=config['use_grads'],
        reg_weight=config['pde_reg']
    ).to(device)
    
    # Load model weights
    mixer.load_state_dict(checkpoint['mixer'])
    pde.load_state_dict(checkpoint['pde'])
    
    mixer.build_params()  # Important: call build_params after loading
    mixer.eval()
    pde.eval()

    rots = checkpoint['rots']
    
    return mixer, pde, rots

def iterate_latent_representations(pde, chosen_timepoints, initial_state, num_steps, device='cuda'):   
    """Iterate latent representations using PDE grid evolution.
    
    This function evolves a latent state representation through time using a partial 
    differential equation (PDE) model. Starting from an initial state, it performs 
    iterative evolution to compute the latent representations at specified timepoints.
    
    Args:
        pde: The PDE model used for evolution. Must have a `step()` method that 
            advances the state by one time step.
        chosen_timepoints (list): List of timepoint indices to evolve to. The first 
            timepoint is treated as the initial state (T0).
        initial_state (torch.Tensor): The initial latent state tensor to start 
            evolution from. Will be automatically unsqueezed to add batch dimension.
        num_steps (dict): Dictionary mapping timepoint indices to the number of 
            evolution steps required to reach that timepoint from the initial state.
        device (str, optional): Device to perform computations on. Defaults to 'cuda'.
    
    Returns:
        dict: Dictionary mapping timepoint indices to their corresponding evolved 
            state tensors. Each state tensor has shape [1, latent_dim, height, width].
        
    Example:
        >>> pde = ReactionDiffusionPDE(latent_dim=20)
        >>> initial_state = torch.randn(20, 64, 64)
        >>> chosen_timepoints = [1, 2, 3, 4]
        >>> num_steps = {1: 10, 2: 20, 3: 30, 4: 70}
        >>> evolved_states = iterate_latent_representations(
        ...     pde, chosen_timepoints, initial_state, num_steps
        ... )
    """
    latent_representations = {}
    
    with torch.no_grad():
        # Start with initial state (T0)
        current_state = initial_state.unsqueeze(0).detach().clone()
        print(f"Starting PDE evolution with initial state shape: {current_state.shape}")
        
        # Evolve the PDE state for each timepoint
        evolved_states = {1: current_state.detach().clone()}  # Store T0

        for tp_idx in chosen_timepoints[1:]:
            print(f"Evolving to timepoint {tp_idx}...")
            # Evolve the state by timescale steps to get to the next timepoint

            current_state = initial_state.unsqueeze(0).detach().clone()
            for substep in range(num_steps[tp_idx]):
                current_state = pde.step(current_state)
                # Check for NaN or inf values
                if torch.isnan(current_state).any() or torch.isinf(current_state).any():
                    print(f"WARNING: NaN or Inf detected in PDE state at timepoint {tp_idx}, substep {substep}")
            evolved_states[tp_idx] = current_state.detach().clone()
            print(f"Completed evolution to timepoint {tp_idx}, state shape: {current_state.shape}")

    return evolved_states

def load_visualization_data(data_path, device='cuda'):
    """Load and preprocess data for visualization purposes.
    
    This function loads spatial transcriptomics data and preprocesses it. It performs gene filtering, normalization,
    coordinate preprocessing, and converts data to PyTorch tensors for downstream
    visualization and analysis.
    
    Args:
        data_path (str): Path to the data file to be loaded.
        device (str, optional): Device to place tensors on ('cuda' or 'cpu'). 
            Defaults to 'cuda'.
    
    Returns:
        tuple: A tuple containing the following elements:
            - coords_all_timepoints (list): List of coordinate tensors for each timepoint.
              Each tensor has shape [num_cells, 2] with spatial coordinates.
            - gex_tensor (torch.Tensor): Gene expression tensor of shape [num_cells, num_genes]
              containing normalized expression values.
            - pos (list): List of numpy arrays containing preprocessed spatial coordinates
              for each timepoint.
            - gex (pandas.DataFrame): Gene expression data as a DataFrame with cells as rows
              and genes as columns.
            - tp (dict): Dictionary mapping timepoint indices to tuples of (coordinates, expression)
              data.
            - size (tuple): Size parameters used for coordinate preprocessing.
            - zero_pos (numpy.ndarray): Zero position reference used in coordinate preprocessing.
    """
    # Load the data
    x = load_data(data_path)
    
    # Preprocess data using the same parameters as Figure1
    tp, gex = preprocess_data(
        x, 
        use_genes='all',
        preprocess_gex=True,
        filter=True,
        fg_bc_min_pct_cells_by_counts=1.0,
        normalize='mediannorm'
    )
    
    # Set scale and preprocess coordinates
    size, yshifts, xshifts = set_scale('big')
    zero_pos, pos, nonzero_ims = preprocess_coords(tp, size=size, yshifts=yshifts, xshifts=xshifts)
    
    # Convert coordinates for all timepoints to tensors
    coords_all_timepoints = []
    for tp_idx in range(len(pos)):
        coords_tensor = torch.tensor(pos[tp_idx], dtype=torch.float32).to(device)
        coords_all_timepoints.append(coords_tensor)
    
    gex_tensor = torch.tensor(gex.values, dtype=torch.float32).to(device)
    
    return coords_all_timepoints, gex_tensor, pos, gex, tp, size, zero_pos

def load_initial_state(ckpt_path, latent_dim, size, zero_pos, init_timepoint=0, device='cuda'):
    """
    Load the initial state (T0) from a checkpoint file for PDE evolution.
    
    This function loads a saved initial state from a checkpoint file, or creates a random
    initialization if no saved state is found. It also handles setting zero positions
    in the initial state based on provided zero position coordinates.
    
    Args:
        ckpt_path (str): Path to the checkpoint file containing the saved model state
        latent_dim (int): Number of latent dimensions for the state tensor
        size (int): Spatial size of the grid (height and width)
        zero_pos (dict): Dictionary mapping timepoint indices to tuples containing
            zero position coordinates as (zero_coords, other_data)
        init_timepoint (int, optional): Timepoint index to use for zero positions.
            Defaults to 0.
        device (str, optional): Device to load tensors on ('cuda' or 'cpu').
            Defaults to 'cuda'.
    """
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # The initial state should be stored in the checkpoint
    if 'start' in checkpoint:
        initial_state = checkpoint['start'].to(device)
        print(f"Loaded 'start' state from checkpoint: {initial_state.shape}")
    else:
        # Create a random initialization as fallback
        print("Warning: No 'start' state found in checkpoint, creating random initialization")
        initial_state = torch.rand(latent_dim, size, size, device=device, dtype=torch.float)
        
        # Set zero positions to zero
        if zero_pos and init_timepoint in zero_pos:
            zero_coords = zero_pos[init_timepoint][0]  # Get zero coordinates
            if len(zero_coords) > 0:
                # Ensure zero coordinates are within bounds
                valid_zero = (zero_coords[:, 0] >= 0) & (zero_coords[:, 0] < size) & (zero_coords[:, 1] >= 0) & (zero_coords[:, 1] < size)
                valid_zero_coords = zero_coords[valid_zero]
                if len(valid_zero_coords) > 0:
                    initial_state[:, valid_zero_coords[:, 0], valid_zero_coords[:, 1]] = 0
                    print(f"Set {len(valid_zero_coords)} positions to zero in initial state")
    
    return initial_state

def compute_latent_representations_grid(pde, coords_all_timepoints, chosen_timepoints, 
                                        initial_state, size, num_steps, rots, device='cuda'):
    """
    Compute latent representations for chosen timepoints using PDE grid evolution.
    
    This function evolves a PDE system starting from an initial state to generate latent
    representations at specific spatial coordinates for chosen timepoints. It performs
    grid-based evolution and extracts latent values at specified coordinates.
    
    Args:
        pde: PDE model object with step() method for evolving the state
        coords_all_timepoints (list): List of coordinate tensors for each timepoint
        chosen_timepoints (list): List of timepoint indices to compute representations for
        initial_state (torch.Tensor): Initial state tensor of shape (latent_dim, height, width)
        size (int): Grid size for the PDE evolution
        num_steps (dict): Dictionary mapping timepoint indices to number of evolution steps
        rots (dict): Dictionary containing rotation parameters for each timepoint with 'weight' and 'bias'
        device (str, optional): Device to run computations on. Defaults to 'cuda'.
    """
    latent_representations = {}
    latent_grid = {}
    figures = {}
    
    with torch.no_grad():
        # Start with initial state (T0)
        current_state = initial_state.unsqueeze(0).detach().clone()
        print(f"Starting PDE evolution with initial state shape: {current_state.shape}")
        
        # Evolve the PDE state for each timepoint
        evolved_states = {1: current_state.detach().clone()}  # Store T0
        
        try:
            for tp_idx in chosen_timepoints[1:]:
                print(f"Evolving to timepoint {tp_idx}...")
                # Evolve the state by timescale steps to get to the next timepoint

                current_state = initial_state.unsqueeze(0).detach().clone()

                print(num_steps)
                print(tp_idx)
                print(num_steps[tp_idx])
                for substep in range(num_steps[tp_idx]):
                    current_state = pde.step(current_state)
                    # Check for NaN or inf values
                    if torch.isnan(current_state).any() or torch.isinf(current_state).any():
                        print(f"WARNING: NaN or Inf detected in PDE state at timepoint {tp_idx}, substep {substep}")
                evolved_states[tp_idx] = current_state.detach().clone()
                print(f"Completed evolution to timepoint {tp_idx}, state shape: {current_state.shape}")
        except Exception as e:
            print(f"Error during PDE evolution: {e}")
            raise e
        
        # Create separate figures for each latent dimension showing timepoints
        figures['latent_dimensions'] = {}
        
        for latent_idx in range(20):  # Assuming 20 latent dimensions
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            for tp_col, tp_idx in enumerate(chosen_timepoints):
                ax = axes[tp_col]
                
                # Extract the specific latent dimension
                latent_map = evolved_states[tp_idx][0, latent_idx].cpu().numpy()
                
                im = ax.imshow(latent_map, cmap='viridis')
                ax.set_title(f'Latent {latent_idx}, TP {tp_idx}')
                ax.axis('off')
                
                # Add colorbar for each subplot
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            figures['latent_dimensions'][latent_idx] = fig

        figures['timepoint_plots'] = {}
        for tp_idx in chosen_timepoints:
            # compute evolved_grid_rotated
            evolved_grid_rotated = rotate(rots[tp_idx]['weight'],rots[tp_idx]['bias'],evolved_states[tp_idx])
            # create figure for evolved_grid_rotated
            fig = plt.figure()
            plt.imshow(evolved_grid_rotated[0].sum(axis=0).cpu().numpy())
            plt.colorbar()
            plt.title(f"Timepoint {tp_idx}")
            figures['timepoint_plots'][tp_idx] = fig
            latent_grid[tp_idx] = evolved_grid_rotated

        # Extract latent values at specific coordinates for chosen timepoints
        for tp_idx in chosen_timepoints:
            print(f"\nProcessing timepoint {tp_idx}...")
            if tp_idx < len(coords_all_timepoints) and tp_idx in evolved_states:
                coords = coords_all_timepoints[tp_idx]
                evolved_grid = evolved_states[tp_idx]  # Shape: (latent_dim, height, width)
                
                evolved_grid_rotated = rotate(rots[tp_idx]['weight'],rots[tp_idx]['bias'],evolved_grid)

                coords_np = coords.cpu().numpy()
                
                # Check if coordinates are in reasonable range
                if coords_np[:, 0].max() >= size or coords_np[:, 1].max() >= size:
                    print(f"  WARNING: Some coordinates exceed grid size {size}")
                if coords_np[:, 0].min() < 0 or coords_np[:, 1].min() < 0:
                    print(f"  WARNING: Some coordinates are negative")
                
                # Convert coordinates to grid indices
                x_coords = coords_np[:, 0]
                y_coords = coords_np[:, 1]
                
                # Ensure indices are within valid range [0, size-1]
                x_indices = np.clip(np.round(x_coords).astype(int), 0, size-1)
                y_indices = np.clip(np.round(y_coords).astype(int), 0, size-1)
                
                print(f"  Index ranges after clipping: x=[{x_indices.min()}, {x_indices.max()}], y=[{y_indices.min()}, {y_indices.max()}]")
                
                # Convert to torch tensors
                x_indices_torch = torch.from_numpy(x_indices).long().to(device)
                y_indices_torch = torch.from_numpy(y_indices).long().to(device)
                
                # Extract latent values
                try:
                    # Use advanced indexing with torch tensors
                    latent_values = evolved_grid_rotated[:, :, x_indices_torch, y_indices_torch].T  # Shape: (num_points, latent_dim)
                    
                    latent_representations[tp_idx] = {
                        'latent': latent_values.cpu().numpy(),
                        'coords': coords.cpu().numpy()
                    }
                except Exception as e:
                    print(f"  Error extracting latent values for timepoint {tp_idx}: {e}")
                    
            else:
                print(f"  Timepoint {tp_idx} not available in data or evolution")
    
    return latent_representations, latent_grid, figures


def compute_latent_gene_attribution(latent_representations, mixer, tp, config, attribution_method='DeepLift', baseline='zeros', device='cuda', batch_size=1000):
    """
    Compute attribution matrix from latent dimensions to genes using gradient-based attribution methods.
    
    This function computes how much each latent dimension contributes to each gene's expression
    by using gradient-based attribution methods like InputXGradient or DeepLift. The attribution
    scores help understand which latent dimensions are most important for predicting specific genes.
    
    Args:
        latent_representations (dict): Dictionary mapping timepoint indices to dictionaries
            containing 'latent' (numpy array of latent values) and 'coords' (spatial coordinates).
        mixer: Pretrained mixer model with a decoder component used for attribution analysis.
        tp: Timepoint data (currently not used but kept for compatibility).
        config (dict): Configuration dictionary containing:
            - 'out_dim' (int): Number of output genes
            - 'latent_dim' (int): Number of latent dimensions
        attribution_method (str, optional): Attribution method to use. Options are:
            - 'InputXGradient': Computes input times gradient attribution
            - 'DeepLift': Uses DeepLift attribution method
            Defaults to 'DeepLift'.
        baseline (str, optional): Baseline method for attribution computation. Options are:
            - 'zeros': Use all-zeros baseline
            - 'random': Use random baseline
            - 'mean': Use mean baseline (not implemented)
            Defaults to 'zeros'.
        device (str, optional): Device to run computations on ('cuda' or 'cpu'). 
            Defaults to 'cuda'.
        batch_size (int, optional): Batch size for processing genes (currently not used
            as genes are processed individually). Defaults to 1000.
    
    Returns:
        dict: Dictionary mapping timepoint indices to attribution matrices. Each attribution
            matrix has shape (latent_dim, n_genes) where entry [i, j] represents how much
            latent dimension i contributes to gene j's expression.
    """
    print(f"Computing latent-to-gene attribution matrix using {attribution_method}...")
    
    # Get the decoder (MLP)
    decoder = mixer.decoder
    

    if attribution_method == 'InputXGradient':
        # Create InputXGradient interpreter
        input_x_gradient = InputXGradient(decoder)
    elif attribution_method == 'DeepLift':
        # Create DeepLift interpreter
        deep_lift = DeepLift(decoder)
    else:
        raise ValueError(f"Invalid attribution method: {attribution_method}")
    
    # Initialize attribution matrix
    n_genes = config['out_dim']
    latent_dim = config['latent_dim']

    attribution_matrix_all_tps = {}

    for i, tp in enumerate(latent_representations):

        print("Attributing for tp", tp)

        latent_tp = torch.from_numpy(latent_representations[tp]['latent']).squeeze().to(device)

        attribution_matrix_tp = np.zeros((latent_dim, n_genes))

        if baseline == 'random':
            baseline= torch.randn(1, latent_dim, device=device, requires_grad=True)
        elif baseline == 'zeros':
            # use all zeros input
            baseline = torch.zeros(1, latent_dim, device=device, requires_grad=True)
        elif baseline == 'mean':
            raise NotImplementedError("Mean baseline computation not implemented")
        
        # Compute attribution for each gene in this batch
        for g in range(n_genes):
            if g % 1000 == 0:
                print(f"Completed attributions for {g} genes")
            try:
                # Compute InputXGradient for target gene
                if attribution_method == 'InputXGradient':
                    attribution = input_x_gradient.attribute(latent_tp, baselines=baseline, target=g)
                elif attribution_method == 'DeepLift':
                    attribution = deep_lift.attribute(latent_tp, baselines=baseline, target=g)
                else:
                    raise ValueError(f"Invalid attribution method: {attribution_method}")
                # Take the mean across the batch dimension and store
                attribution_matrix_tp[:, g] = attribution.mean(dim=0).detach().cpu().numpy()
            except Exception as e:
                print(f"Error computing attribution for gene {g}: {e}")
                # Set to zero if computation fails
                attribution_matrix_tp[:, g] = 0.0

        attribution_matrix_all_tps[i] = attribution_matrix_tp
    
    return attribution_matrix_all_tps

# Gene Set Enrichment Analysis (GSEA) for each latent dimension
def perform_gsea_for_latent(attribution_matrix, latent_idx, gene_names, organism='Fish', 
                           library_name='GO_Biological_Process_2018', top_n=1000):
    """
    Perform Gene Set Enrichment Analysis (GSEA) for a specific latent dimension.
    
    This function conducts GSEA analysis on attribution scores from a specific latent 
    dimension to identify enriched biological pathways and gene sets. It uses the 
    attribution scores as gene rankings and performs prerank GSEA against a specified 
    gene set library.
    
    Args:
        attribution_matrix (numpy.ndarray): Attribution matrix of shape (latent_dim, n_genes)
            containing attribution scores for each latent dimension and gene.
        latent_idx (int): Index of the latent dimension to analyze (0-based).
        gene_names (list or array-like): List or array of gene names corresponding to
            the genes in the attribution matrix.
        organism (str, optional): Organism name for GSEA gene set library. 
            Defaults to 'Fish' for zebrafish.
        library_name (str, optional): Name of the gene set library to use for enrichment
            analysis. Defaults to 'GO_Biological_Process_2018'.
        top_n (int, optional): Number of top-ranked genes to consider for analysis.
            Currently not used in implementation but reserved for future filtering.
            Defaults to 1000.
    
    Returns:
        pandas.DataFrame: GSEA results dataframe containing enrichment statistics
            including gene set names, normalized enrichment scores (NES), p-values,
            FDR q-values, and other GSEA metrics.
    """
    print(f"Performing GSEA for latent dimension {latent_idx}...")
    
    # Get attribution scores for this latent dimension
    latent_scores = attribution_matrix[latent_idx, :]
    
    # Create gene ranking dataframe
    gene_ranking = pd.DataFrame({
        'gene': gene_names,
        'score': latent_scores
    })
    
    # Sort by score (descending)
    gene_ranking = gene_ranking.sort_values('score', ascending=False)
    
    fish_mf = gp.get_library(name=library_name, organism=organism)
    
    # Perform GSEA
    pre_res = gp.prerank(
        rnk=gene_ranking,
        gene_sets=fish_mf,
        threads=4,
        min_size=5,
        max_size=1000,
        permutation_num=1000,
        outdir=None,
        format='png',
        seed=6,
        verbose=True
    )
    
    return pre_res.res2d

def perform_gsea_all_latents(attribution_matrix, gene_names, 
                                organism='Fish', library_name='GO_Biological_Process_2018', 
                                RESULTS_DIR = '', OUT_PREFIX = '', 
                                FDR_THRESHOLD = 0.25,
                                filter_positive_nes = True,
                                verbose = True,
                                top_n=1000):
    """
    Perform Gene Set Enrichment Analysis (GSEA) for all latent dimensions.
    
    This function performs GSEA analysis on each latent dimension by using the attribution
    scores from the attribution matrix. For each latent dimension, it creates a gene ranking
    based on attribution scores and performs enrichment analysis using the specified gene
    set library. Results are saved individually for each latent dimension.
    
    Args:
        attribution_matrix (numpy.ndarray): Attribution matrix of shape (latent_dim, n_genes)
            where each entry [i, j] represents how much latent dimension i contributes to 
            gene j's expression.
        gene_names (list): List of gene names corresponding to the columns in the attribution
            matrix. Length should match attribution_matrix.shape[1].
        organism (str, optional): Organism name for GSEA analysis. Used to select appropriate
            gene set libraries. Defaults to 'Fish'.
        library_name (str, optional): Name of the gene set library to use for enrichment
            analysis. Examples include 'GO_Biological_Process_2018', 'KEGG_2019', etc.
            Defaults to 'GO_Biological_Process_2018'.
        RESULTS_DIR (str, optional): Directory path where individual GSEA results files
            will be saved. If empty string, saves to current directory. Defaults to ''.
        OUT_PREFIX (str, optional): Prefix to add to output filenames. Used to distinguish
            different runs or experiments. Defaults to ''.
        top_n (int, optional): Number of top-ranking genes to consider for GSEA analysis.
            Higher values include more genes but may reduce signal. Defaults to 1000.
    
    Returns:
        dict: Dictionary mapping latent dimension indices to their corresponding GSEA
            results DataFrames. Keys are integers (latent dimension indices) and values
            are pandas DataFrames containing enrichment results, or None if analysis
            failed for that dimension.
    """
    print("Performing GSEA for all latent dimensions...")
    
    all_gsea_results = {}
    latent_dim = attribution_matrix.shape[0]
    
    for latent_idx in tqdm(range(latent_dim), desc="GSEA for latent dimensions"):
        try:
            gsea_results = perform_gsea_for_latent(
                attribution_matrix, latent_idx, gene_names, 
                organism, library_name, top_n
            )
            all_gsea_results[latent_idx] = gsea_results

            filtered_results = gsea_results[gsea_results['FDR q-val'] < FDR_THRESHOLD]
            if filter_positive_nes:
                filtered_results = filtered_results[filtered_results['NES'] > 0]
            filtered_results = filtered_results.sort_values('NES', ascending=False)

            if verbose:
                print(f"GSEA results for latent dimension {latent_idx}:")
                print(filtered_results)
            
            # Save results for this latent dimension
            filtered_results.to_csv(f'{RESULTS_DIR}/gsea_results_latent_{latent_idx}_{OUT_PREFIX}_filtered.csv')
            gsea_results.to_csv(f'{RESULTS_DIR}/gsea_results_latent_{latent_idx}_{OUT_PREFIX}_all.csv')
            
        except Exception as e:
            print(f"Error in GSEA for latent dimension {latent_idx}: {e}")
            all_gsea_results[latent_idx] = None
    
    return all_gsea_results