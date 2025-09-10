# UvS - TS: Unlearning via Surrogation for Time Series Data
## Topic:
- By utilizing the motif(s) detection and data sequence generation, unlearning in Time Series can be achieved via the act of replacing corrupted/unwanted data with generated ones.
- This framework inlcudes:
  + PA-EW (Population-Aware Elastic Weight Consolidation):
  + T_GR (Temporal Generative Replay):
  + PEPA: A motif detection technique.
  
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