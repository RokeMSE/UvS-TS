import subprocess

NODE_IDX = 1
UVS_EXTRA = "--lambda-forget 1.0 --forget-margin 0.5"

commands = [
    # Train PEMS-BAY
    "python train.py --enable-cuda --all --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/PEMS-BAY",

    # # Train METR-LA
    "python train.py --enable-cuda --all --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/METR-LA",

    # # Baseline subset PEMS-BAY
    "python run_baselines.py --enable-cuda --all --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/PEMS-BAY --run-retrain --forget-set /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY/forget_set.json",

    # # Baseline subset METR-LA
    "python run_baselines.py --enable-cuda --all --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/METR-LA --run-retrain --forget-set /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA/forget_set.json",

    # # Unlearn 2 subset PEMS-BAY
    "python unlearn_2.py --enable-cuda --all --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/PEMS-BAY --forget-set /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY/forget_set.json",

    # # Unlearn 2 subset METR-LA
    "python unlearn_2.py --enable-cuda --all --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/METR-LA --forget-set /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA/forget_set.json",

    # # Unlearn 3 subset PEMS-BAY
    "python unlearn_3.py --enable-cuda --all --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/PEMS-BAY --forget-set /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY/forget_set.json",

    # # Unlearn 3 subset METR-LA
    "python unlearn_3.py --enable-cuda --all --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/METR-LA --forget-set /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA/forget_set.json",

    # # Unlearn logic 2 subset PEMS-BAY model STGCN
    f"python unlearn_logic_2.py --enable-cuda --type stgcn --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/PEMS-BAY --forget-set /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY/forget_set.json {UVS_EXTRA}",

    # # Unlearn logic 2 subset METR-LA model STGCN
    f"python unlearn_logic_2.py --enable-cuda --type stgcn --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/METR-LA --forget-set /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA/forget_set.json {UVS_EXTRA}",

    # # Unlearn logic 2 subset PEMS-BAY model STGAT
    f"python unlearn_logic_2.py --enable-cuda --type stgat --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/PEMS-BAY --forget-set /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY/forget_set.json {UVS_EXTRA}",

    # # Unlearn logic 2 subset METR-LA model STGAT
    f"python unlearn_logic_2.py --enable-cuda --type stgat --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/METR-LA --forget-set /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA/forget_set.json {UVS_EXTRA}",

    # # Unlearn logic 2 subset PEMS-BAY model GWNET
    f"python unlearn_logic_2.py --enable-cuda --type gwnet --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/PEMS-BAY --forget-set /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY/forget_set.json {UVS_EXTRA}",

    # # Unlearn logic 2 subset METR-LA model GWNET
    f"python unlearn_logic_2.py --enable-cuda --type gwnet --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/METR-LA --forget-set /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA/forget_set.json {UVS_EXTRA}",

    # # Baseline node PEMS-BAY
    # f"python run_baselines.py --enable-cuda --all --unlearn-node --node-idx {NODE_IDX} --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/PEMS-BAY --run-retrain",

    # # Baseline node METR-LA
    # f"python run_baselines.py --enable-cuda --all --unlearn-node --node-idx {NODE_IDX} --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/METR-LA --run-retrain",

    # # Unlearn 2 node PEMS-BAY
    # f"python unlearn_2.py --enable-cuda --all --unlearn-node --node-idx {NODE_IDX} --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/PEMS-BAY",

    # # Unlearn 2 node METR-LA
    # f"python unlearn_2.py --enable-cuda --all --unlearn-node --node-idx {NODE_IDX} --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/METR-LA",

    # # Unlearn 3 node PEMS-BAY
    # f"python unlearn_3.py --enable-cuda --all --unlearn-node --node-idx {NODE_IDX} --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/PEMS-BAY",

    # # Unlearn 3 node METR-LA
    # f"python unlearn_3.py --enable-cuda --all --unlearn-node --node-idx {NODE_IDX} --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/METR-LA",

    # Unlearn logic 2 node PEMS-BAY model STGCN
    f"python unlearn_logic_2.py --enable-cuda --type stgcn --node-mode motif --threshold 0.5 --unlearn-node --node-idx {NODE_IDX} --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/PEMS-BAY {UVS_EXTRA}",

    # Unlearn logic 2 node METR-LA model STGCN
    f"python unlearn_logic_2.py --enable-cuda --type stgcn --node-mode motif --threshold 0.5 --unlearn-node --node-idx {NODE_IDX} --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/METR-LA {UVS_EXTRA}",

    # Unlearn logic 2 node PEMS-BAY model STGAT
    f"python unlearn_logic_2.py --enable-cuda --type stgat --node-mode motif --threshold 0.5 --unlearn-node --node-idx {NODE_IDX} --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/PEMS-BAY {UVS_EXTRA}",

    # Unlearn logic 2 node METR-LA model STGAT
    f"python unlearn_logic_2.py --enable-cuda --type stgat --node-mode motif --threshold 0.5 --unlearn-node --node-idx {NODE_IDX} --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/METR-LA {UVS_EXTRA}",

    # Unlearn logic 2 node PEMS-BAY model GWNET
    f"python unlearn_logic_2.py --enable-cuda --type gwnet --node-mode motif --threshold 0.5 --unlearn-node --node-idx {NODE_IDX} --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/PEMS-BAY {UVS_EXTRA}",

    # Unlearn logic 2 node METR-LA model GWNET
    f"python unlearn_logic_2.py --enable-cuda --type gwnet --node-mode motif --threshold 0.5 --unlearn-node --node-idx {NODE_IDX} --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/METR-LA {UVS_EXTRA}",
   
    #-- Unlearn logic 2 node PEMS-BAY model STGCN
    f"python unlearn_logic_2.py --enable-cuda --type stgcn --node-mode leverage --leverage-keep 0.2 --unlearn-node --node-idx {NODE_IDX} --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/PEMS-BAY {UVS_EXTRA}",

    # Unlearn logic 2 node METR-LA model STGCN
    f"python unlearn_logic_2.py --enable-cuda --type stgcn --node-mode leverage --leverage-keep 0.2 --unlearn-node --node-idx {NODE_IDX} --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/METR-LA {UVS_EXTRA}",

    # Unlearn logic 2 node PEMS-BAY model STGAT
    f"python unlearn_logic_2.py --enable-cuda --type stgat --node-mode leverage --leverage-keep 0.2 --unlearn-node --node-idx {NODE_IDX} --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/PEMS-BAY {UVS_EXTRA}",

    # Unlearn logic 2 node METR-LA model STGAT
    f"python unlearn_logic_2.py --enable-cuda --type stgat --node-mode leverage --leverage-keep 0.2 --unlearn-node --node-idx {NODE_IDX} --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/METR-LA {UVS_EXTRA}",

    # Unlearn logic 2 node PEMS-BAY model GWNET
    f"python unlearn_logic_2.py --enable-cuda --type gwnet --node-mode leverage --leverage-keep 0.2 --unlearn-node --node-idx {NODE_IDX} --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/PEMS-BAY {UVS_EXTRA}",

    # Unlearn logic 2 node METR-LA model GWNET
    f"python unlearn_logic_2.py --enable-cuda --type gwnet --unlearn-node --node-idx {NODE_IDX} --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/METR-LA {UVS_EXTRA}",
]

for i, cmd in enumerate(commands):
    print(f"\n Running command {i+1}/{len(commands)}:")
    print(cmd)

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"? Command failed at step {i+1}")
        break
