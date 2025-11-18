# UvS - TS: Unlearning via Surrogation for Time Series Data
## Topic:
- By utilizing the motif(s) detection and data sequence generation, unlearning in Time Series can be achieved via the act of replacing corrupted/unwanted data with generated ones.
- This framework inlcudes:
  + PA-EW (Population-Aware Elastic Weight Consolidation):
  + T_GR (Temporal Generative Replay):
  + PEPA: A motif detection technique.

- The full Framework can be visualized as an Objective Function which of consists 3 Orders:
$$
\mathcal{L}_{SA-TS} = \mathbb{E}_{x_f \sim q(x|c_f)}[\log p(x_f|c_f)] - \lambda \sum_{i}\frac{F_i^{T}}{2}(\theta_{i}-\theta_{i}^{*})^{2} + \mathbb{E}_{x_r \sim p(x|c_r)}[\log p(x_r|c_r)]
$$

- **Where**:
  1. **FIRST ORDER**:  
  2. **SECOND ORDER**: Fisher Information is used to make sure the model performance is retain with the retain set $D_r$.
  3. **THIRD ORDER**: 

## TO DO LIST:
1. Load Datasets.
2. Train initial Models.
...

### Core:
1. Implement PEPA.
2. Optimize MMD, FiM calculations.
n?. MIA attacks.
...

## Pipeline
``` 
Data Loading → Model Training → Unlearning Process → Evaluation
     ↓              ↓              ↓              ↓
data_loader.py → stgcn.py → unlearn.py → evaluate.py
```
## Dir Tree:
```
|-- 📂 data/
|   |-- preprocess_ecg.py          # Script for ECG data prep
|   |-- preprocess_pemsbay.py      # Script for PEMS-BAY prep
|-- 📂 configs/
|   |-- stgcn_pemsbay.yaml         # Config for ST-GNN experiment
|-- 📂 src/
|   |-- 📂 models/
|   |   |-- stgcn.py               # STGCN model architecture
|   |-- 📂 unlearning/
|   |   |-- concept_definition.py  # Implements PEPA algorithm for motif discovery
|   |   |-- statistics.py          # Functions to calculate population stats (ACF, PSD, etc.)
|   |   |-- pa_ewc.py              # Calculates L_pop, MMD, and the Population-Aware FIM
|   |   |-- t_gr.py                # Implements Self-Contained Temporal Generative Replay
|   |-- 📂 utils/
|   |   |-- data_utils.py         # Support functions
|   |   |-- logger.py             # Setup for logging results
|   |   |-- losses.py             # Evaluation metrics (e.g., forgetting score, retain accuracy)
|   |-- train.py                   # Main script to train initial models
|   |-- unlearn.py                 # Main script to run the SA-TS unlearning process
|   |-- evaluate.py                # Script to evaluate models post-unlearning
|-- 📂 notebooks/
|   |-- 1_data_exploration.ipynb   # Visualize datasets
|   |-- 2_motif_discovery.ipynb    # Test and visualize PEPA algorithm results
|   |-- 3_model_testing.ipynb      # Debug individual model components
|   |-- 4_results_analysis.ipynb   # Plot final results and metrics
|
|-- requirements.txt               # Python package dependencies
|-- README.md                      # Project description
```

### NOTICE: For functions called from different file, use this for safety `import {folder_name}/{file_name}`

## How to run:
- Train the base model: \
`python src/train.py --enable-cuda --input C:/Users/rokeM/Downloads/"UvS-TS Data"/PEMSBAY --model C:/Users/rokeM/Downloads/"UvS-TS Data/Model"`

`python src/train.py --enable-cuda --input "/home/cs.aau.dk/tungkvt/Hung_Quyen/Data/PEMSBAY" --model "/home/cs.aau.dk/tungkvt/Hung_Quyen/Data/Model"`

`python src/train.py --enable-cuda --input "/q/storage/tung/Hung_Quyen/Data/PEMSBAY" --model "/q/storage/tung/Hung_Quyen/Data/Model"`

- Unlearn: \
  + Subsection:
`python src/unlearn.py --enable-cuda --input Data/PEMSBAY --model Data/Model --forget-set Data/forget_set.txt --node-idx 2 --type stgcn`
  + Node:
`python src/unlearn.py --enable-cuda --input Data/PEMSBAY --model Data/Model --unlearn-node --node-idx 2 --type stgcn`
  + Run all unlearning baselines and compare results
`python src/run_baselines.py --enable-cuda --input Data/PEMSBAY --model Data/Model --node-idx 10 --unlearn-node --type stgcn`
  + Viualize: 
`python src/visualize_spatio_data.py --input "Data/PEMSBAY" --original-model "Data\Model" --unlearned-model "Data/Model/Unlearn node 2" --node-idx 2`