import subprocess

commands = [
    # Train PEMS-BAY
    # "python train.py --enable-cuda --all --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/PEMS-BAY",

    # Train METR-LA
    #"python train.py --enable-cuda --all --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/METR-LA",

    # Baseline PEMS-BAY
    #"python run_baselines.py --enable-cuda --all --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/PEMS-BAY --run-retrain --forget-set /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY/forget_set.json",

    # Baseline METR-LA
    #"python run_baselines.py --enable-cuda --all --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/METR-LA --run-retrain --forget-set /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA/forget_set.json",

    # Unlearn 2 PEMS-BAY
    #"python unlearn_2.py --enable-cuda --all --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/PEMS-BAY --forget-set /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY/forget_set.json",

    # Unlearn 2 METR-LA
    #"python unlearn_2.py --enable-cuda --all --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/METR-LA --forget-set /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA/forget_set.json",

    # Unlearn 3 PEMS-BAY
    #"python unlearn_3.py --enable-cuda --all --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/PEMS-BAY --forget-set /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY/forget_set.json",

    # Unlearn 3 METR-LA
    #"python unlearn_3.py --enable-cuda --all --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/METR-LA --forget-set /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA/forget_set.json",

    # Unlearn logic 2 PEMS-BAY model STGCN
    "python unlearn_logic_2.py --enable-cuda --type stgcn --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/PEMS-BAY --forget-set /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY/forget_set.json",

    # Unlearn logic 2 METR-LA model STGCN
    "python unlearn_logic_2.py --enable-cuda --type stgcn --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/METR-LA --forget-set /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA/forget_set.json",

    # Unlearn logic 2 PEMS-BAY model STGAT
    "python unlearn_logic_2.py --enable-cuda --type stgat --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/PEMS-BAY --forget-set /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY/forget_set.json",

    # Unlearn logic 2 METR-LA model STGAT
    "python unlearn_logic_2.py --enable-cuda --type stgat --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/METR-LA --forget-set /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA/forget_set.json",

    # Unlearn logic 2 PEMS-BAY model GWNET
    "python unlearn_logic_2.py --enable-cuda --type gwnet --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/PEMS-BAY --forget-set /user/cs.aau.dk/tungkvt/hung_quyen/Data/PEMS-BAY/forget_set.json",

    # Unlearn logic 2 METR-LA model GWNET
    "python unlearn_logic_2.py --enable-cuda --type gwnet --input /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA --model /user/cs.aau.dk/tungkvt/hung_quyen/Model/METR-LA --forget-set /user/cs.aau.dk/tungkvt/hung_quyen/Data/METR-LA/forget_set.json",
    
]

for i, cmd in enumerate(commands):
    print(f"\n Running command {i+1}/{len(commands)}:")
    print(cmd)

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"? Command failed at step {i+1}")
        break