import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch.utils.data import TensorDataset, DataLoader
import os

# Assuming the project structure is as provided
from models.stgcn import STGCN
from utils.data_loader import load_data_PEMS_BAY
from data.preprocess_pemsbay import get_normalized_adj, generate_dataset
from evaluate import get_model_predictions

def plot_spatio_temporal_data(A, original_preds, unlearned_preds, sample_idx, timestep, faulty_node_idx=None):
    """
    Plots the graph with node values as a heatmap for a specific sample and timestep.
    Compares the original model's predictions with the unlearned model's predictions.
    """
    G = nx.from_numpy_array(A)
    pos = nx.spring_layout(G, seed=42)  # for consistent layout

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'Spatial Data Visualization for Sample {sample_idx} at Timestep {timestep}', fontsize=16)

    # --- Original Model Predictions ---
    axes[0].set_title('Before Unlearning (Original Model)')
    node_colors_original = original_preds[sample_idx, :, timestep, 0].cpu().numpy()
    vmin = min(node_colors_original)
    vmax = max(node_colors_original)

    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors_original, cmap=plt.cm.viridis, node_size=500, ax=axes[0], vmin=vmin, vmax=vmax)
    nx.draw_networkx_edges(G, pos, ax=axes[0], alpha=0.5)
    nx.draw_networkx_labels(G, pos, ax=axes[0], font_size=8, font_color='white')

    if faulty_node_idx is not None:
        nx.draw_networkx_nodes(G, pos, nodelist=[faulty_node_idx], node_color='red', node_size=600, ax=axes[0])


    # --- Unlearned Model Predictions ---
    axes[1].set_title('After Unlearning (Unlearned Model)')
    node_colors_unlearned = unlearned_preds[sample_idx, :, timestep, 0].cpu().numpy()

    nodes_unlearned = nx.draw_networkx_nodes(G, pos, node_color=node_colors_unlearned, cmap=plt.cm.viridis, node_size=500, ax=axes[1], vmin=vmin, vmax=vmax)
    nx.draw_networkx_edges(G, pos, ax=axes[1], alpha=0.5)
    nx.draw_networkx_labels(G, pos, ax=axes[1], font_size=8, font_color='white')

    if faulty_node_idx is not None:
        nx.draw_networkx_nodes(G, pos, nodelist=[faulty_node_idx], node_color='red', node_size=600, ax=axes[1])


    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', pad=0.05)
    cbar.set_label('Predicted Node Value')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(f'spatio_temporal_visualization_sample_{sample_idx}_timestep_{timestep}.png')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Spatio-Temporal Data Visualization')
    parser.add_argument('--enable-cuda', action='store_true', help='Enable CUDA')
    parser.add_argument('--input', type=str, required=True, help='Path to the directory containing dataset')
    parser.add_argument('--original-model', type=str, required=True, help='Path to the original trained model file (model.pt)')
    parser.add_argument('--unlearned-model', type=str, required=True, help='Path to the unlearned model file (model.pt)')
    parser.add_argument('--node-idx', type=int, default=None, help='Node index that was unlearned, to highlight it.')
    parser.add_argument('--sample-idx', type=int, default=0, help='Index of the sample to visualize')
    parser.add_argument('--timestep', type=int, default=0, help='Index of the timestep to visualize')


    args = parser.parse_args()
    args.device = torch.device('cuda' if args.enable_cuda and torch.cuda.is_available() else 'cpu')

    # Load data
    print("Loading PEMS-BAY data...")
    A, X, means, stds = load_data_PEMS_BAY(args.input)
    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave).float().to(args.device)

    # Load original model
    print("Loading original model...")
    checkpoint_orig = torch.load(args.original_model, map_location=args.device)
    original_model = STGCN(**checkpoint_orig["config"]).to(args.device)
    original_model.load_state_dict({k: v.float() for k, v in checkpoint_orig["model_state_dict"].items()})
    original_model.eval()


    # Load unlearned model
    print("Loading unlearned model...")
    checkpoint_unlearned = torch.load(args.unlearned_model, map_location=args.device)
    unlearned_model = STGCN(**checkpoint_unlearned["config"]).to(args.device)
    unlearned_model.load_state_dict({k: v.float() for k, v in checkpoint_unlearned["model_state_dict"].items()})
    unlearned_model.eval()


    config = checkpoint_orig["config"]
    num_timesteps_input = config["num_timesteps_input"]
    num_timesteps_output = config["num_timesteps_output"]

    # Create a test loader
    split_line = int(X.shape[2] * 0.8)
    test_original_data = X[:, :, split_line:]
    test_input, test_target = generate_dataset(test_original_data,
                                                num_timesteps_input=num_timesteps_input,
                                                num_timesteps_output=num_timesteps_output)
    test_dataset = TensorDataset(test_input, test_target)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Get predictions
    print("Generating predictions...")
    original_preds, _ = get_model_predictions(original_model, test_loader, A_wave, args.device)
    unlearned_preds, _ = get_model_predictions(unlearned_model, test_loader, A_wave, args.device)

    # Plot the data
    print("Plotting spatio-temporal data...")
    plot_spatio_temporal_data(A, original_preds, unlearned_preds, args.sample_idx, args.timestep, args.node_idx)
    print(f"Visualization saved to spatio_temporal_visualization_sample_{args.sample_idx}_timestep_{args.timestep}.png")

    # Save the visualization
    print("Visualization complete.")
    os.makedirs('visualizations', exist_ok=True)
    os.replace(f'spatio_temporal_visualization_sample_{args.sample_idx}_timestep_{args.timestep}.png', 
               os.path.join('visualizations', f'spatio_temporal_visualization_sample_{args.sample_idx}_timestep_{args.timestep}.png'))

if __name__ == '__main__':
    main()