# Latent Dynamical Systems (LDS)

A computational framework for learning spatiotemporal dynamics from time-series spatial transcriptomics data using latent partial differential equations (PDEs) and spatial registration.

## Key Features

- **Latent PDE**: Fits reaction-diffusion PDE in a learned latent space to capture spatiotemporal dynamics
- **Decoder**: Learns decoder mapping latent variables to gene expression. The attributions of latents to genes from the decoder can be interpreted as the learned gene programs. 
- **Spatial Registration**: Registers latent trajectory to each sample via a learnable affine transformation

## Example of a learned latent trajectory of zebrafish embryogenesis

https://github.com/user-attachments/assets/245e13f0-da6e-4cf0-8218-157e19fc7419

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

### Data Download

Download the preprocessed datasets and place them in the root directory of the repository. Note that the zebrafish dataset is 200 MB while the mouse dataset is 23 GB. 

**Zebrafish Dataset:**
- Main data: [`zebrafish_spatial_fivetime_slice_stereoseq.h5ad`](https://storage.googleapis.com/lds_recomb_data/zebrafish_spatial_fivetime_slice_stereoseq.h5ad)

**Mouse Dataset:**
- Main data: [`Mouse_embryo_all_stage.h5ad`](https://storage.googleapis.com/lds_recomb_data/Mouse_embryo_all_stage.h5ad)
- Metadata files:
  - [`hg38_mm10_1k_features.pkl`](https://storage.googleapis.com/lds_recomb_data/hg38_mm10_1k_features.pkl)
  - [`mm10.ncbiRefSeq.gtf`](https://storage.googleapis.com/lds_recomb_data/mm10.ncbiRefSeq.gtf)

**References:**
- Mouse: Chen et al. *Spatiotemporal transcriptomic atlas of mouse organogenesis using DNA nanoball-patterned arrays.* Cell (2022)
- Zebrafish: Liu et al. *Spatiotemporal mapping of gene expression landscapes and developmental trajectories during zebrafish embryogenesis.* Science (2022) 

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
    --mix mlp \
    --chosen_tps "1,2,3,4"
```

Train on mouse data:

```bash
mkdir output
python run/mouse_train.py \
    --latent_dim 40 \
    --out_dir ./output \
    --timescale 25 \
    --mix mlp
```

#### Using Jupyter Notebooks

For interactive exploration, see:
- `RECOMB_zebrafish_train.ipynb`: Minimal notebook demonstrating training a LDS model on the zebrafish dataset
- `RECOMB_mouse_train.ipynb`: Notebook for mouse embryogenesis data

