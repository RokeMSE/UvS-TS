# UvS-TS: Unlearning via Surrogation for Time Series

Unlearn corrupted or unwanted segments in spatio-temporal forecasting models (STGCN / STGAT / GWNet) by replacing target data with motif-conditioned surrogates, guided by a Fisher-regularised objective.

## Pipeline
```
Data Loading -> Model Training -> Unlearning -> Evaluation
```

## How to run

Replace `<DATA>` and `<MODEL>` with your dataset and model directories. All commands are run from `src/`.

### 1. Train base models
Train all three architectures:
```bash
python train.py --enable-cuda --all --input <DATA> --model <MODEL>
```
Single model:
```bash
python train.py --enable-cuda --type stgcn --input <DATA> --model <MODEL>
```

### 2. Run baselines + evaluation
Subset unlearning (requires `forget_set.json`):
```bash
python run_baselines.py --enable-cuda --all --input <DATA> --model <MODEL> \
    --forget-set <DATA>/forget_set.json --run-retrain
```
Node unlearning:
```bash
python run_baselines.py --enable-cuda --all --unlearn-node --node-idx 42 \
    --input <DATA> --model <MODEL> --run-retrain
```
Results are written to `<MODEL>/baselines_node_<type>_<idx>/` (CSV, JSON, `.pt`). Drop `--run-retrain` to skip the slow gold-standard retrain.

### 3. Run UvS-TS (our method)
Per-model; repeat for `stgat` and `gwnet`.

Subset:
```bash
python unlearn_logic_2.py --enable-cuda --type stgcn --input <DATA> --model <MODEL> \
    --forget-set <DATA>/forget_set.json
```
Node:
```bash
python unlearn_logic_2.py --enable-cuda --type stgcn --unlearn-node --node-idx 42 \
    --input <DATA> --model <MODEL>
```
Useful flags: `--epochs`, `--lr`, `--lambda-surr/retain/ewc/forget`, `--forget-margin`, `--retrain-baseline`, `--out-suffix` (for sweeps).

### Batch driver
[src/run_all_command.py](src/run_all_command.py) contains ready-to-use command lists — uncomment the rows you want and run `python run_all_command.py`.

Evaluation (`evaluate_unlearning`) is called inside steps 2 and 3, so no separate evaluation command is needed.

## NOTE
- Unlearn_subset: Use unlearn_logic_2, replace forget_set by surrogate + early stopping
- Unlearn node:
    - unlear_2: Use Linear Structural Model(graph base)
    - unlearn_3: Use Vector Autoregression + Impulse Response Function(dynamic time series base)

## Reason of unlearning
A camera can learn to continuosly to alert the correct situation -> unlearning the wrong signals (suddenly traffic jams on that day due to accidents) -> wrong learn -> wrong prediction -> need unlearn