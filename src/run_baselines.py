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
from utils.data_loader import load_data_PEMS_BAY
from data.preprocess_pemsbay import get_normalized_adj, generate_dataset
from evaluate import evaluate_unlearning
from unlearning_baselines import UnlearningBaselines


def prepare_data_loaders(X, A, args, num_timesteps_input, num_timesteps_output):
    """Prepare all necessary data loaders for unlearning experiments"""
    
    split_line = int(X.shape[2] * 0.8)
    train_data = X[:, :, :split_line]
    test_data = X[:, :, split_line:]
    
    # Generate datasets
    train_input, train_target = generate_dataset(
        train_data, num_timesteps_input, num_timesteps_output
    )
    test_input, test_target = generate_dataset(
        test_data, num_timesteps_input, num_timesteps_output
    )
    
    # For node unlearning: use neighbor nodes for retain, faulty node for forget
    if args.unlearn_node:
        faulty_node_idx = args.node_idx
        
        # Get neighbors
        row = A[faulty_node_idx]
        neighbor_indices = np.where(row > 0)[0]
        neighbor_indices = [idx for idx in neighbor_indices if idx != faulty_node_idx]
        
        if len(neighbor_indices) == 0:
            raise ValueError(f"Node {faulty_node_idx} has no neighbors!")
        
        # Retain: first neighbor's data
        retain_node_idx = neighbor_indices[0]
        retain_data = train_data[retain_node_idx:retain_node_idx+1, :, :]
        retain_input, retain_target = generate_dataset(
            retain_data, num_timesteps_input, num_timesteps_output
        )
        
        # Forget: faulty node's data
        forget_data = train_data[faulty_node_idx:faulty_node_idx+1, :, :]
        forget_input, forget_target = generate_dataset(
            forget_data, num_timesteps_input, num_timesteps_output
        )
        
        # Modify adjacency matrix
        new_A = A.copy()
        new_A[faulty_node_idx, :] = 0
        new_A[:, faulty_node_idx] = 0
        new_A_wave = get_normalized_adj(new_A)
        new_A_wave = torch.from_numpy(new_A_wave).float()
        
    else:
        # For subset unlearning: use motif discovery (simplified version)
        # For baseline comparison, we'll use random forget/retain split
        num_samples = train_input.shape[0]
        forget_ratio = 0.1  # 10% forget
        
        indices = torch.randperm(num_samples)
        forget_size = int(num_samples * forget_ratio)
        
        forget_indices = indices[:forget_size]
        retain_indices = indices[forget_size:]
        
        forget_input = train_input[forget_indices]
        forget_target = train_target[forget_indices]
        retain_input = train_input[retain_indices]
        retain_target = train_target[retain_indices]
        
        new_A_wave = get_normalized_adj(A)
        new_A_wave = torch.from_numpy(new_A_wave).float()
    
    # Create data loaders
    batch_size = 512
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


def run_baselines(args):
    """Run all baseline methods and compare results"""
    
    print("="*80)
    print("UNLEARNING BASELINE COMPARISON")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    A, X, means, stds = load_data_PEMS_BAY(args.input)
    X = X.astype(np.float32)
    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave).float().to(args.device)
    
    # Load original model
    print("Loading original model...")
    checkpoint = torch.load(
        os.path.join(args.model, f"{args.type}_model.pt"),
        map_location=args.device
    )
    
    if args.type == 'stgcn':
        model_class = STGCN
    elif args.type == 'stgat':
        model_class = STGAT
    else:
        raise ValueError(f"Unknown model type: {args.type}")
    
    original_model = model_class(**checkpoint["config"]).to(args.device)
    original_model.load_state_dict({
        k: v.float() for k, v in checkpoint["model_state_dict"].items()
    })
    
    config = checkpoint["config"]
    num_timesteps_input = config["num_timesteps_input"]
    num_timesteps_output = config["num_timesteps_output"]
    
    # Prepare data loaders
    print("Preparing data loaders...")
    loaders = prepare_data_loaders(
        X, A, args, num_timesteps_input, num_timesteps_output
    )
    
    # Initialize baseline runner
    baseline_runner = UnlearningBaselines(device=args.device)
    
    # Storage for results
    all_results = {}
    all_models = {}
    
    # ========== Run Each Baseline ==========
    
    # 1. Retrain from Scratch (Gold Standard)
    if args.run_retrain:
        print("\n" + "="*80)
        print("BASELINE 1/6: Retrain from Scratch")
        print("="*80)
        try:
            retrained_model = baseline_runner.retrain_from_scratch(
                model_class, config, loaders['retain_loader'], A_wave,
                num_epochs=100, learning_rate=1e-3
            )
            all_models['retrain'] = retrained_model
            
            results = evaluate_unlearning(
                retrained_model, original_model,
                loaders['retain_loader'], loaders['forget_loader'], loaders['test_loader'],
                loaders['new_A_wave'], A_wave, args.device, args.node_idx
            )
            all_results['retrain'] = results
            print("Retrain completed")
        except Exception as e:
            print(f"Retrain failed: {e}")
            all_results['retrain'] = None
    
    # 2. Fine-tune on Retain Set
    print("\n" + "="*80)
    print("BASELINE 2/6: Fine-tune on Retain Set")
    print("="*80)
    try:
        finetuned_model, _ = baseline_runner.finetune_on_retain(
            original_model, loaders['retain_loader'], A_wave,
            num_epochs=50, learning_rate=1e-4
        )
        all_models['finetune'] = finetuned_model
        
        results = evaluate_unlearning(
            finetuned_model, original_model,
            loaders['retain_loader'], loaders['forget_loader'], loaders['test_loader'],
            loaders['new_A_wave'], A_wave, args.device, args.node_idx
        )
        all_results['finetune'] = results
        print("Fine-tune completed")
    except Exception as e:
        print(f"Fine-tune failed: {e}")
        all_results['finetune'] = None
    
    # 3. Gradient Ascent (NegGrad)
    print("\n" + "="*80)
    print("BASELINE 3/6: Gradient Ascent (NegGrad)")
    print("="*80)
    try:
        neggrad_model, _ = baseline_runner.gradient_ascent(
            original_model, loaders['forget_loader'], A_wave,
            num_epochs=20, learning_rate=1e-4
        )
        all_models['neggrad'] = neggrad_model
        
        results = evaluate_unlearning(
            neggrad_model, original_model,
            loaders['retain_loader'], loaders['forget_loader'], loaders['test_loader'],
            loaders['new_A_wave'], A_wave, args.device, args.node_idx
        )
        all_results['neggrad'] = results
        print("NegGrad completed")
    except Exception as e:
        print(f"NegGrad failed: {e}")
        all_results['neggrad'] = None
    
    # 4. Gradient Ascent + Fine-tune
    print("\n" + "="*80)
    print("BASELINE 4/6: Gradient Ascent + Fine-tune")
    print("="*80)
    try:
        neggrad_ft_model, _ = baseline_runner.gradient_ascent_plus_finetune(
            original_model, loaders['forget_loader'], loaders['retain_loader'], A_wave,
            neggrad_epochs=20, finetune_epochs=30
        )
        all_models['neggrad_ft'] = neggrad_ft_model
        
        results = evaluate_unlearning(
            neggrad_ft_model, original_model,
            loaders['retain_loader'], loaders['forget_loader'], loaders['test_loader'],
            loaders['new_A_wave'], A_wave, args.device, args.node_idx
        )
        all_results['neggrad_ft'] = results
        print("NegGrad+FT completed")
    except Exception as e:
        print(f"NegGrad+FT failed: {e}")
        all_results['neggrad_ft'] = None
    
    # 5. Influence Functions
    print("\n" + "="*80)
    print("BASELINE 5/6: Influence Functions")
    print("="*80)
    try:
        influence_model, _ = baseline_runner.influence_function_unlearning(
            original_model, loaders['forget_loader'], loaders['train_loader'], A_wave
        )
        all_models['influence'] = influence_model
        
        results = evaluate_unlearning(
            influence_model, original_model,
            loaders['retain_loader'], loaders['forget_loader'], loaders['test_loader'],
            loaders['new_A_wave'], A_wave, args.device, args.node_idx
        )
        all_results['influence'] = results
        print("Influence Functions completed")
    except Exception as e:
        print(f"Influence Functions failed: {e}")
        all_results['influence'] = None
    
    # 6. Fisher Unlearning
    print("\n" + "="*80)
    print("BASELINE 6/6: Fisher Unlearning")
    print("="*80)
    try:
        fisher_model, _ = baseline_runner.fisher_unlearning(
            original_model, loaders['forget_loader'], loaders['retain_loader'], A_wave,
            num_epochs=50, lambda_fisher=10.0
        )
        all_models['fisher'] = fisher_model
        
        results = evaluate_unlearning(
            fisher_model, original_model,
            loaders['retain_loader'], loaders['forget_loader'], loaders['test_loader'],
            loaders['new_A_wave'], A_wave, args.device, args.node_idx
        )
        all_results['fisher'] = results
        print("Fisher Unlearning completed")
    except Exception as e:
        print(f"Fisher Unlearning failed: {e}")
        all_results['fisher'] = None
    
    # ========== Generate Comparison Report ==========
    print("\n" + "="*80)
    print("GENERATING COMPARISON REPORT")
    print("="*80)
    
    generate_comparison_report(all_results, args)
    
    # Save models
    save_dir = os.path.join(args.model, f"baselines_node_{args.node_idx}")
    os.makedirs(save_dir, exist_ok=True)
    
    for name, model in all_models.items():
        if model is not None:
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config
            }, os.path.join(save_dir, f"{name}_model.pt"))
    
    print(f"\nResults saved to {save_dir}")
    
    return all_results, all_models


def generate_comparison_report(all_results, args):
    """Generate a comprehensive comparison report"""
    
    # Create DataFrame for easy comparison
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
    
    # Print comparison table
    print("\n" + "="*80)
    print("BASELINE COMPARISON TABLE")
    print("="*80)
    print(df.to_string(index=False))
    
    # Save to CSV
    save_dir = os.path.join(args.model, f"baselines_node_{args.node_idx}")
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, "baseline_comparison.csv"), index=False)
    
    # Save detailed results to JSON
    with open(os.path.join(save_dir, "detailed_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    # Print key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    if not df.empty:
        # Best forgetting efficacy
        best_forget = df.loc[df['forgetting_efficacy'].idxmax()]
        print(f"\nBest Forgetting Efficacy: {best_forget['Method']} ({best_forget['forgetting_efficacy']:.4f})")
        
        # Best fidelity
        best_fidelity = df.loc[df['fidelity_score'].idxmax()]
        print(f"Best Fidelity (Retain Performance): {best_fidelity['Method']} ({best_fidelity['fidelity_score']:.4f})")
        
        # Best generalization
        best_gen = df.loc[df['generalization_score'].idxmax()]
        print(f"Best Generalization: {best_gen['Method']} ({best_gen['generalization_score']:.4f})")
        
        # Lowest forget MSE
        best_forget_mse = df.loc[df['forget_set_mse'].idxmax()]
        print(f"Highest Forget MSE (Best Unlearning): {best_forget_mse['Method']} ({best_forget_mse['forget_set_mse']:.4f})")


def main():
    parser = argparse.ArgumentParser(description='Run Unlearning Baselines')
    parser.add_argument('--enable-cuda', action='store_true', help='Enable CUDA')
    parser.add_argument('--unlearn-node', action='store_true', help='Node unlearning mode')
    parser.add_argument('--node-idx', type=int, required=True, help='Node index to unlearn')
    parser.add_argument('--input', type=str, required=True, help='Data directory')
    parser.add_argument('--type', type=str, default='stgcn', choices=['stgcn', 'stgat'], help='Model type')
    parser.add_argument('--model', type=str, required=True, help='Model directory')
    parser.add_argument('--run-retrain', action='store_true', help='Run retrain baseline (slow)')
    
    args = parser.parse_args()
    args.device = torch.device('cuda' if args.enable_cuda and torch.cuda.is_available() else 'cpu')
    
    # Run all baselines
    all_results, all_models = run_baselines(args)
    
    print("\n" + "="*80)
    print("BASELINE COMPARISON COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()