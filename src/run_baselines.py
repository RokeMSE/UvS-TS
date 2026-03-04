import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import json
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

# Import models and utilities
from models.stgcn import STGCN
from models.stgat import STGAT
from models.gwn import gwnet
from utils.data_loader import load_data_PEMS_BAY
from data.preprocess_pemsbay import get_normalized_adj, generate_dataset
from evaluate import evaluate_unlearning
from unlearning_baselines import UnlearningBaselines


def prepare_data_loaders(train, test, A, args, num_timesteps_input, num_timesteps_output, forget_set=None):
    """Prepare all necessary data loaders for unlearning experiments"""

    
    # Generate FULL datasets (B, N, T, F)
    train_input, train_target = generate_dataset(
        train, num_timesteps_input, num_timesteps_output
    )
    nums_window = train_input.shape[0]
    window_starts = torch.arange(nums_window)
    window_ends = window_starts + num_timesteps_input + num_timesteps_output - 1

    test_input, test_target = generate_dataset(
        test, num_timesteps_input, num_timesteps_output
    )
    
    new_A_wave = None
    
    if args.unlearn_node:
        faulty_node_idx = args.node_idx
        print(f"Preparing data for Node Unlearning: Node {faulty_node_idx}")
        
        # Modify adjacency matrix for unlearning (isolate faulty node)
        new_A = A.copy()
        new_A[faulty_node_idx, :] = 0
        new_A[:, faulty_node_idx] = 0
        new_A_wave = get_normalized_adj(new_A)
        new_A_wave = torch.from_numpy(new_A_wave).float()
        
        # Retain Set = Train Set
        forget_input = train_input[:, faulty_node_idx:faulty_node_idx+1, :, :]
        forget_target = train_target[:, faulty_node_idx:faulty_node_idx+1, :, :]    

        # --- Retain set = mask faulty node ---
        retain_input = train_input.clone()
        retain_target = train_target.clone()

        retain_input[:, faulty_node_idx, :, :] = 0
        retain_target[:, faulty_node_idx, :, :] = 0

    else:
        # For subset unlearning (temporal split)
        mask_forget = torch.zeros(nums_window, dtype=torch.bool)
        for key, value in forget_set.items():
            for item in value:
                start = item[0]
                end = item[1]
                overlap = ~((window_ends < start) | (window_starts > end))
                mask_forget |= overlap

        Df_indices = torch.where(mask_forget)[0]
        Dr_indices = torch.where(~mask_forget)[0]

        retain_input = train_input[Dr_indices]
        retain_target = train_target[Dr_indices]

        forget_input = train_input[Df_indices]
        forget_target = train_target[Df_indices]
        
        new_A_wave = get_normalized_adj(A)
        new_A_wave = torch.from_numpy(new_A_wave).float()
    
    # Create data loaders
    batch_size = 512  # Adjust based on model type if needed
    train_loader = DataLoader(
        TensorDataset(train_input, train_target), 
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(test_input, test_target),
        batch_size=batch_size, shuffle=False
    )
    forget_loader = DataLoader(
        TensorDataset(forget_input, forget_target),
        batch_size=batch_size, shuffle=True
    )
    retain_loader = DataLoader(
        TensorDataset(retain_input, retain_target),
        batch_size=batch_size, shuffle=True
    )
    
    return {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'forget_loader': forget_loader,
        'retain_loader': retain_loader,
        'new_A_wave': new_A_wave
    }


def run_baselines(args, model_class, original_model, raw_config, loaders, A_wave, time_gen_loader):
    """Run all baseline methods and compare results"""
    baseline_runner = UnlearningBaselines(device=args.device)
    
    all_results = {}
    all_models = {}
    
    # Determine node to unlearn (if any)
    faulty_node_idx = args.node_idx if args.unlearn_node else None
    
    # 1. Retrain from Scratch (Gold Standard)
    if args.run_retrain:
        print("\n" + "="*80)
        print("BASELINE 1/6: Retrain from Scratch")
        print("="*80)
        try:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            retrained_model = baseline_runner.retrain_from_scratch(
                model_class, raw_config, loaders['retain_loader'], A_wave,
                num_epochs=100, learning_rate=1e-3,
                faulty_node_idx=faulty_node_idx
            )

            end.record()
            torch.cuda.synchronize()

            all_models['retrain'] = retrained_model
            
            results = evaluate_unlearning(
                retrained_model, original_model,
                loaders['retain_loader'], loaders['forget_loader'], loaders['test_loader'],
                loaders['new_A_wave'], A_wave, args.device, args.node_idx
            )
            results['time'] = start.elapsed_time(end) / 1000 + time_gen_loader
            all_results['retrain'] = results
            print("Retrain completed")
        except Exception as e:
            print(f"Retrain failed: {e}")
            import traceback
            traceback.print_exc()
            all_results['retrain'] = None
    
    # 2. Fine-tune on Retain Set
    print("\n" + "="*80)
    print("BASELINE 2/6: Fine-tune on Retain Set")
    print("="*80)
    try:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        finetuned_model, _ = baseline_runner.finetune_on_retain(
            original_model, loaders['retain_loader'], loaders['new_A_wave'],
            num_epochs=50, learning_rate=1e-4,
            faulty_node_idx=faulty_node_idx
        )
        end.record()
        torch.cuda.synchronize()

        all_models['finetune'] = finetuned_model
        
        results = evaluate_unlearning(
            finetuned_model, original_model,
            loaders['retain_loader'], loaders['forget_loader'], loaders['test_loader'],
            loaders['new_A_wave'], A_wave, args.device, args.node_idx
        )
        results['time'] = start.elapsed_time(end) / 1000 + time_gen_loader
        all_results['finetune'] = results
        print("Fine-tune completed")
    except Exception as e:
        print(f"Fine-tune failed: {e}")
        import traceback
        traceback.print_exc()
        all_results['finetune'] = None
    
    # 3. Gradient Ascent (NegGrad)
    print("\n" + "="*80)
    print("BASELINE 3/6: Gradient Ascent (NegGrad)")
    print("="*80)
    try:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        neggrad_model, _ = baseline_runner.gradient_ascent(
            original_model, loaders['forget_loader'], A_wave,
            num_epochs=20, learning_rate=1e-4,
            faulty_node_idx=faulty_node_idx
        )

        end.record()
        torch.cuda.synchronize()

        all_models['neggrad'] = neggrad_model
        
        results = evaluate_unlearning(
            neggrad_model, original_model,
            loaders['retain_loader'], loaders['forget_loader'], loaders['test_loader'],
            loaders['new_A_wave'], A_wave, args.device, args.node_idx
        )
        results['time'] = start.elapsed_time(end) / 1000 + time_gen_loader
        all_results['neggrad'] = results
        print("NegGrad completed")
    except Exception as e:
        print(f"NegGrad failed: {e}")
        import traceback
        traceback.print_exc()
        all_results['neggrad'] = None
    
    # 4. Gradient Ascent + Fine-tune
    print("\n" + "="*80)
    print("BASELINE 4/6: Gradient Ascent + Fine-tune")
    print("="*80)
    try:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        neggrad_ft_model, _ = baseline_runner.gradient_ascent_plus_finetune(
            original_model, loaders['forget_loader'], loaders['retain_loader'], A_wave,
            neggrad_epochs=20, finetune_epochs=30,
            faulty_node_idx=faulty_node_idx
        )

        end.record()
        torch.cuda.synchronize()

        all_models['neggrad_ft'] = neggrad_ft_model
        
        results = evaluate_unlearning(
            neggrad_ft_model, original_model,
            loaders['retain_loader'], loaders['forget_loader'], loaders['test_loader'],
            loaders['new_A_wave'], A_wave, args.device, args.node_idx
        )
        results['time'] = start.elapsed_time(end) / 1000 + time_gen_loader
        all_results['neggrad_ft'] = results
        print("NegGrad+FT completed")
    except Exception as e:
        print(f"NegGrad+FT failed: {e}")
        import traceback
        traceback.print_exc()
        all_results['neggrad_ft'] = None
    
    # 5. Influence Functions
    print("\n" + "="*80)
    print("BASELINE 5/6: Influence Functions")
    print("="*80)
    try:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        influence_model, _ = baseline_runner.influence_function_unlearning(
            original_model, loaders['forget_loader'], loaders['train_loader'], A_wave,
            faulty_node_idx=faulty_node_idx
        )

        end.record()
        torch.cuda.synchronize()

        all_models['influence'] = influence_model
        
        results = evaluate_unlearning(
            influence_model, original_model,
            loaders['retain_loader'], loaders['forget_loader'], loaders['test_loader'],
            loaders['new_A_wave'], A_wave, args.device, args.node_idx
        )

        results['time'] = start.elapsed_time(end) / 1000 + time_gen_loader
        all_results['influence'] = results
        print("Influence Functions completed")
    except Exception as e:
        print(f"Influence Functions failed: {e}")
        import traceback
        traceback.print_exc()
        all_results['influence'] = None
    
    # 6. Fisher Unlearning
    print("\n" + "="*80)
    print("BASELINE 6/6: Fisher Unlearning")
    print("="*80)
    try:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        fisher_model, _ = baseline_runner.fisher_unlearning(
            original_model, loaders['forget_loader'], loaders['retain_loader'], A_wave,
            num_epochs=50, lambda_fisher=10.0,
            faulty_node_idx=faulty_node_idx
        )

        end.record()
        torch.cuda.synchronize()

        all_models['fisher'] = fisher_model
        
        results = evaluate_unlearning(
            fisher_model, original_model,
            loaders['retain_loader'], loaders['forget_loader'], loaders['test_loader'],
            loaders['new_A_wave'], A_wave, args.device, args.node_idx
        )
        results['time'] = start.elapsed_time(end) / 1000 + time_gen_loader
        all_results['fisher'] = results
        print("Fisher Unlearning completed")
    except Exception as e:
        print(f"Fisher Unlearning failed: {e}")
        import traceback
        traceback.print_exc()
        all_results['fisher'] = None
    
    print("\n" + "="*80)
    print("GENERATING COMPARISON REPORT")
    print("="*80)
    
    generate_comparison_report(all_results, args)
    
    save_dir = os.path.join(args.model, f"baselines_node_{args.node_idx}")
    os.makedirs(save_dir, exist_ok=True)
    
    for name, model in all_models.items():
        if model is not None:
            model_config_to_save = getattr(model, 'config', raw_config)
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': model_config_to_save
            }, os.path.join(save_dir, f"{name}_model.pt"))
    
    print(f"\nResults saved to {save_dir}")
    return all_results, all_models


def generate_comparison_report(all_results, args):
    metrics = [
        'fidelity_score', 'forgetting_efficacy', 'generalization_score',
        'forget_set_mse', 'retain_set_mse', 'test_set_mse',
        'spatial_correlation_divergence', 'temporal_pattern_divergence'
    ]
    
    data = []
    for method, results in all_results.items():
        if results is not None:
            row = {'Method': method}
            for metric in metrics:
                row[metric] = results.get(metric, np.nan)
            data.append(row)
    
    df = pd.DataFrame(data)
    
    print("\n" + "="*80)
    print("BASELINE COMPARISON TABLE")
    print("="*80)
    print(df.to_string(index=False))
    
    save_dir = os.path.join(args.model, f"baselines_node_{args.node_idx}")
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, "baseline_comparison.csv"), index=False)
    
    with open(os.path.join(save_dir, "detailed_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    if not df.empty:
        if 'forgetting_efficacy' in df.columns:
            best_forget = df.loc[df['forgetting_efficacy'].idxmax()]
            print(f"\nBest Forgetting Efficacy: {best_forget['Method']} ({best_forget['forgetting_efficacy']:.4f})")
        if 'fidelity_score' in df.columns:
            best_fidelity = df.loc[df['fidelity_score'].idxmax()]
            print(f"Best Fidelity (Retain Performance): {best_fidelity['Method']} ({best_fidelity['fidelity_score']:.4f})")
        if 'generalization_score' in df.columns:
            best_gen = df.loc[df['generalization_score'].idxmax()]
            print(f"Best Generalization: {best_gen['Method']} ({best_gen['generalization_score']:.4f})")


def main():
    parser = argparse.ArgumentParser(description='Run Unlearning Baselines')
    parser.add_argument('--enable-cuda', action='store_true', help='Enable CUDA')
    parser.add_argument('--unlearn-node', action='store_true', help='Node unlearning mode')
    parser.add_argument('--all', action='store_true', help='Unlearn all model')
    parser.add_argument('--node-idx', type=int, help='Node index to unlearn')
    parser.add_argument('--input', type=str, required=True, help='Data directory')
    parser.add_argument('--type', type=str, default='stgcn', 
                        choices=['stgcn', 'stgat', 'gwnet'], help='Model type')
    parser.add_argument('--model', type=str, required=True, help='Model directory')
    parser.add_argument('--run-retrain', action='store_true', help='Run retrain baseline (slow)')
    parser.add_argument('--forget-set', type=str, help='Path to the directory containing forget dataset')
    
    args = parser.parse_args()
    args.device = torch.device('cuda' if args.enable_cuda and torch.cuda.is_available() else 'cpu')

    print("="*80)
    print("UNLEARNING BASELINE COMPARISON")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    A, train_original_data, test_original_data, means, stds = load_data_PEMS_BAY(args.input)
    
    if not args.all:
        # Load original model
        print("Loading original model...")
        checkpoint = torch.load(
            os.path.join(args.model, f"{args.type}_model.pt"),
            map_location=args.device
        )
        
        raw_config = checkpoint["config"]
        
        if args.type == 'stgcn':
            model_class = STGCN
        elif args.type == 'stgat':
            model_class = STGAT
        elif args.type == 'gwnet':
            model_class = gwnet
        else:
            raise ValueError(f"Unknown model type: {args.type}")

        original_model = model_class(**raw_config).to(args.device)

        original_model.load_state_dict({
            k: v.float() for k, v in checkpoint["model_state_dict"].items()
        })
        
        num_timesteps_input = raw_config.get("nums_timestep_in", 12)
        num_timesteps_output = raw_config.get("nums_step_out", 4)
        
        # Read forget_set
        with open(args.forget_set, 'r', encoding='utf8') as f:
            forget_set_json = json.load(f)
            
        
        print("Preparing data loaders...")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        loaders = prepare_data_loaders(
            train_original_data, test_original_data, A, args, num_timesteps_input, num_timesteps_output, forget_set_json
        )
        end.record()
        torch.cuda.synchronize()

        time_gen_loader = start.elapsed_time(end) / 1000

        
        run_baselines(args, model_class, original_model, raw_config, loaders, loaders['new_A_wave'], time_gen_loader)
        
        print("\n" + "="*80)
        print("BASELINE COMPARISON COMPLETED!")
        print("="*80)

    else:
        list_models = ['stgcn', 'stgat', 'gwnet']
        for model_name in list_models:
            args.type = model_name
            
            # Load original model
            print("Loading original model...")
            checkpoint = torch.load(
                os.path.join(args.model, f"{args.type}_model.pt"),
                map_location=args.device
            )
            
            raw_config = checkpoint["config"]
            
            if args.type == 'stgcn':
                model_class = STGCN
            elif args.type == 'stgat':
                model_class = STGAT
            elif args.type == 'gwnet':
                model_class = gwnet
            else:
                raise ValueError(f"Unknown model type: {args.type}")

            original_model = model_class(**raw_config).to(args.device)

            original_model.load_state_dict({
                k: v.float() for k, v in checkpoint["model_state_dict"].items()
            })
            
            num_timesteps_input = raw_config.get("nums_timestep_in", 12)
            num_timesteps_output = raw_config.get("nums_step_out", 4)
            
            # Read forget_set
            with open(args.forget_set, 'r', encoding='utf8') as f:
                forget_set_json = json.load(f)
                
            
            print("Preparing data loaders...")
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            loaders = prepare_data_loaders(
                train_original_data, test_original_data, A, args, num_timesteps_input, num_timesteps_output, forget_set_json
            )
            end.record()
            torch.cuda.synchronize()

            time_gen_loader = start.elapsed_time(end) / 1000

            
            run_baselines(args, model_class, original_model, raw_config, loaders, loaders['new_A_wave'], time_gen_loader)
            
            print("\n" + "="*80)
            print("BASELINE COMPARISON COMPLETED!")
            print("="*80)


if __name__ == "__main__":
    main()