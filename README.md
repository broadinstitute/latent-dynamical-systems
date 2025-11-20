# Latent Dynamical Systems (LDS)

A computational framework for learning spatiotemporal dynamics from time-series spatial transcriptomics data using latent partial differential equations (PDEs) and spatial registration.

## Key Features

- **Latent PDE**: Fits reaction-diffusion PDE in a learned latent space to capture spatiotemporal dynamics
- **Decoder**: Learns decoder mapping latent variables to gene expression
- **Spatial Registration**: Registers latent trajectory to each sample via a learnable affine transformation

## Installation

### Requirements

```bash
# Core dependencies
torch>=1.9.0
numpy
scipy
pandas
matplotlib
tqdm
scanpy
```

### Setup

```bash
git clone https://github.com/broadinstitute/latent-dynamical-systems
cd latent-dynamical-systems
pip install -r requirements.txt
```

## Quick Start

Preprocessed data for the mouse dataset can be downloaded from: https://storage.googleapis.com/lds_recomb_data/Mouse_embryo_all_stage.h5ad. Additional metadata files for running LDS on the mouse dataset are also available: https://storage.googleapis.com/lds_recomb_data/hg38_mm10_1k_features.pkl, https://storage.googleapis.com/lds_recomb_data/mm10.ncbiRefSeq.gtf. Preprocesed data for the zebrafish dataset can be downloaded from https://storage.googleapis.com/lds_recomb_data/zebrafish_spatial_fivetime_slice_stereoseq.h5ad. Additional information on these datasets is available from the corresponding original publications for mouse [Chen et al. Spatiotemporal transcriptomic atlas of mouse organogenesis using DNA nanoball-patterned arrays. (2022)] and zebrafish [Liu et al. Spatiotemporal mapping of gene expression landscapes and developmental trajectories during zebrafish embryogenesis. (2022)].

The scripts expect the data to be downloaded to the root directory of the repository. 

### Training a Model

#### Using Python Scripts

Train on zebrafish data:

```bash
mkdir output
python run/train_zebrafish.py \
    --out_dir ./output \
    --out_prefix zebrafish_exp1 \
    --latent_dim 20 \
    --max_its 10000 \
    --pde ReactionDiffusion \
    --mix mlp
```

Train on mouse data:

```bash
mkdir output
python run/mouse_train.py \
    --out_dir ./output \
    --out_prefix mouse_exp1 \
    --latent_dim 20 \
    --max_its 10000
```

#### Using Jupyter Notebooks

For interactive exploration, see:
- `RECOMB_zebrafish_train.ipynb`: Minimal notebook demonstrating training a LDS model on the zebrafish dataset
- `RECOMB_mouse_train.ipynb`: Notebook for mouse embryogenesis data

