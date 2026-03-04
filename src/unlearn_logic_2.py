""" 
Run the 3 Orders from the Initial Models using files from the unlearning folder
- Combine Components: Load the pre-trained model ($θ*$).
- Partition Data: Use PEPA implementation to get $D_f$ and $D_r$.
- Calculate FIM: Compute the PA-FIM ($F^T$) using $D_r$ and PA-EWC module.

Changes (annotated with # FIX tags):
  FIX-1  unlearn_faulty_subset: FIM now uses self.new_A_wave (= original A_wave for
         subset unlearning), not the supplied A_wave argument, keeping the
         interface consistent with the node path.
  FIX-2  unlearn_faulty_subset: surrogate T-GR call also uses self.new_A_wave.
  FIX-3  training() (subset loop): replaced separate surrogate/retain forward passes
         that used the supplied A_wave with self.new_A_wave; removed the
         prediction clamping that was not present in the node path or in the paper.
  FIX-4  training() (subset loop): added GRADIENT-ASCENT forget term
         (−λ_forget * L_forget) to match the node training loop and the updated
         paper objective.  forget_loader is now passed in and cycled the same way
         as in _training_loop_node.
  FIX-5  main(): forget_set variable was referenced but never assigned when
         args.unlearn_node is False; now extracted correctly from forget_set_json.
  FIX-6  main(): unlearn_faulty_subset call now passes forget_set (the actual array)
         instead of the undefined name forget_set.
  FIX-7  Metric formula direction: fidelity/generalization comments updated — these
         are evaluated in evaluate.py; noted here for cross-reference.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os
import copy
import argparse
import json
# Components
from models import (
    stgat,
    stgcn,
    gwn
)
from utils.data_loader import load_data_PEMS_BAY
from unlearning.pa_ewc import PopulationAwareEWC
from unlearning.t_gr import TemporalGenerativeReplay
from unlearning.motif_def import discover_motifs_proxy
from data.preprocess_pemsbay import get_normalized_adj, generate_dataset
from evaluate import evaluate_unlearning
import sys
sys.path.append('src')


class SATimeSeries:
    """UvS-TS Framework — unified node and subset unlearning."""

    def __init__(self, model, A_wave, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.original_model = copy.deepcopy(model).to(self.device)
        self.original_model.eval()

        self.model = copy.deepcopy(model).to(self.device)
        for param in self.model.parameters():
            param.data = param.data.float()

        # FIX-1 (shared): both paths store A_wave AND new_A_wave from the start
        self.A_wave = A_wave.float().to(self.device)
        self.new_A_wave = copy.deepcopy(A_wave).float().to(self.device)

        self.original_params = {
            name: param.data.clone().float()
            for name, param in self.model.named_parameters()
        }

        self.pa_ewc = PopulationAwareEWC("stgcn", device)
        self.t_gr = TemporalGenerativeReplay("stgcn")
        self.fim_diagonal = None

    # -------------------------- SUBSET UNLEARNING --------------------------

    def unlearn_faulty_subset(self, dataset, forget_ex, faulty_node_idx, A_wave, means, stds,
                              num_timesteps_input, num_timesteps_output, threshold=0.1,
                              num_epochs=50, learning_rate=5e-5,
                              lambda_ewc=10.0, lambda_surrogate=1.0, lambda_retain=1.0,
                              lambda_forget=0.5, batch_size=512):
        """
        Subset unlearning pipeline.

        For subset unlearning the graph topology is not changed, so
        self.new_A_wave remains a copy of the original A_wave throughout.
        """
        print(f"Starting subset unlearning for node {faulty_node_idx}")

        dataset = dataset.astype(np.float32)
        train_input, train_target = generate_dataset(
            dataset, num_timesteps_input, num_timesteps_output
        )
        forget_ex = forget_ex.astype(np.float32)

        # Partition data via DTW-based motif discovery
        forget_indices, retain_indices = discover_motifs_proxy(
            dataset, forget_ex, faulty_node_idx, threshold
        )
        print(f"Forget segments: {len(forget_indices)}, Retain segments: {len(retain_indices)}")
        print(forget_indices)

        if not forget_indices:
            print("No forget samples found. Skipping training.")
            forget_loader = DataLoader(TensorDataset(torch.empty(0), torch.empty(0)))
            retain_loader = DataLoader(
                TensorDataset(train_input, train_target),
                batch_size=batch_size, shuffle=True
            )
            return (
                {'total_loss': [], 'surrogate_loss': [], 'ewc_penalty': [],
                 'retain_loss': [], 'forget_loss': []},
                forget_loader, retain_loader
            )

        # Build raw time-series segments for forget / retain
        forget_data = [
            dataset[faulty_node_idx:faulty_node_idx+1, :, item[0]:item[1]]
            for item in forget_indices
        ]
        retain_data = [
            dataset[faulty_node_idx:faulty_node_idx+1, :, item[0]:item[1]]
            for item in retain_indices
        ]

        # Retain loader
        all_features_retain, all_targets_retain = [], []
        for item in retain_data:
            feat, tgt = generate_dataset(item, num_timesteps_input, num_timesteps_output)
            if feat.numel() > 0:
                all_features_retain.append(feat)
                all_targets_retain.append(tgt)

        if not all_features_retain:
            print("No retain samples generated. Skipping.")
            return (
                {'total_loss': [], 'surrogate_loss': [], 'ewc_penalty': [],
                 'retain_loss': [], 'forget_loss': []},
                None, None
            )

        retain_loader = DataLoader(
            TensorDataset(
                torch.cat(all_features_retain, dim=0),
                torch.cat(all_targets_retain, dim=0)
            ),
            batch_size=batch_size, shuffle=True
        )

        # Forget loader
        all_features_forget, all_targets_forget = [], []
        for item in forget_data:
            feat, tgt = generate_dataset(item, num_timesteps_input, num_timesteps_output)
            if feat.numel() > 0:
                all_features_forget.append(feat)
                all_targets_forget.append(tgt)

        if not all_features_forget:
            print("No forget samples generated.")
            forget_loader = None
        else:
            forget_loader = DataLoader(
                TensorDataset(
                    torch.cat(all_features_forget, dim=0),
                    torch.cat(all_targets_forget, dim=0)
                ),
                batch_size=batch_size, shuffle=True
            )

        # FIX-1: FIM on retain set using self.new_A_wave (consistent with node path)
        print("Computing Population-Aware FIM on retain set...")
        self.fim_diagonal = self.pa_ewc.calculate_pa_fim(
            self.model, retain_loader, self.new_A_wave, max_samples=500
        )

        # FIX-2: T-GR also uses self.new_A_wave
        print("Creating surrogate data via T-GR...")
        surrogate_data = self.t_gr.perform_temporal_generative_replay_subset(
            self.model,
            dataset[faulty_node_idx:faulty_node_idx+1, :, :],
            forget_indices, faulty_node_idx,
            num_timesteps_input, num_timesteps_output,
            self.device, self.new_A_wave           # FIX-2
        )

        surrogate_features, surrogate_targets = [], []
        for item in surrogate_data:
            if isinstance(item, torch.Tensor):
                item = item.cpu().numpy()
            item = item.astype(np.float32)
            feat, tgt = generate_dataset(item, num_timesteps_input, num_timesteps_output)
            if feat.numel() > 0:
                surrogate_features.append(feat)
                surrogate_targets.append(tgt)

        if not surrogate_features:
            print("Warning: No surrogate samples generated. Skipping training.")
            return (
                {'total_loss': [], 'surrogate_loss': [], 'ewc_penalty': [],
                 'retain_loss': [], 'forget_loss': []},
                forget_loader, retain_loader
            )

        surrogate_loader = DataLoader(
            TensorDataset(
                torch.cat(surrogate_features, dim=0),
                torch.cat(surrogate_targets, dim=0)
            ),
            batch_size=batch_size, shuffle=True
        )

        # FIX-3 + FIX-4: unified training loop (same as node path, no clamping,
        # includes gradient-ascent forget term)
        history = self.training(
            surrogate_loader, retain_loader, forget_loader,
            num_epochs, learning_rate,
            lambda_ewc, lambda_surrogate, lambda_retain, lambda_forget
        )

        # Optionally write surrogate back into dataset (for downstream analysis)
        dataset_new = copy.deepcopy(dataset)
        for i, seg in enumerate(surrogate_data):
            if isinstance(seg, torch.Tensor):
                seg = seg.cpu().numpy()
            dataset_new[
                faulty_node_idx, :,
                forget_indices[i][0]:forget_indices[i][1]
            ] = seg.squeeze(0)
        # Denormalize
        dataset_new = dataset_new * stds.reshape(1, -1, 1) + means.reshape(1, -1, 1)

        return history, forget_loader, retain_loader

    # -------------------------- NODE UNLEARNING (Helper func) --------------------------

    def _build_node_unlearning_loaders(self, train_data, faulty_node_idx,
                                       num_timesteps_input, num_timesteps_output,
                                       batch_size):
        """
        Forget set : all windows, target = faulty node outputs only.
        Retain set : same windows, faulty node zeroed in input & target,
                     loss computed on all other nodes.
        """
        full_input, full_target = generate_dataset(
            train_data, num_timesteps_input, num_timesteps_output
        )

        forget_input = full_input.clone()
        forget_target = full_target[:, faulty_node_idx:faulty_node_idx+1, :, :]

        retain_input = full_input.clone()
        retain_target = full_target.clone()
        retain_input[:, faulty_node_idx, :, :] = 0.0
        retain_target[:, faulty_node_idx, :, :] = 0.0

        forget_loader = DataLoader(
            TensorDataset(forget_input, forget_target),
            batch_size=batch_size, shuffle=True
        )
        retain_loader = DataLoader(
            TensorDataset(retain_input, retain_target),
            batch_size=batch_size, shuffle=True
        )
        return forget_loader, retain_loader

    def _build_surrogate_data_for_node(self, train_data, faulty_node_idx,
                                        num_timesteps_input, num_timesteps_output,
                                        batch_size):
        """
        Neighbor-informed imputation:
          - Zero the faulty node in input.
          - Forward with self.new_A_wave (isolated graph).
          - Faulty node output = what the rest of the graph would predict.
          - Add multi-scale noise to prevent re-learning original patterns.
        """
        full_input, _ = generate_dataset(
            train_data, num_timesteps_input, num_timesteps_output
        )
        full_input = full_input.float()

        masked_input = full_input.clone()
        masked_input[:, faulty_node_idx, :, :] = 0.0

        self.model.eval()
        surrogate_targets = []
        new_A = self.new_A_wave.to(self.device)

        with torch.no_grad():
            for i in range(0, masked_input.shape[0], batch_size):
                batch = masked_input[i:i+batch_size].to(self.device)
                out = self.model(new_A, batch)
                surrogate_targets.append(
                    out[:, faulty_node_idx:faulty_node_idx+1, :, :].cpu()
                )

        surrogate_target = torch.cat(surrogate_targets, dim=0)

        # Multi-scale noise
        noise = (
            torch.randn_like(surrogate_target) * 0.05
            + torch.randn_like(surrogate_target) * 0.10
        )
        surrogate_target = surrogate_target + noise

        return DataLoader(
            TensorDataset(masked_input, surrogate_target),
            batch_size=batch_size, shuffle=True
        )

    def unlearn_faulty_node(self, dataset, faulty_node_idx, A_wave, means, stds,
                            num_timesteps_input, num_timesteps_output,
                            num_epochs=50, learning_rate=5e-5,
                            lambda_ewc=10.0, lambda_surrogate=1.0, lambda_retain=1.0,
                            lambda_forget=1.0, batch_size=512):
        """
        Node unlearning pipeline.

        Step 1: Isolate faulty node in adjacency matrix BEFORE any forward pass.
        Step 2: Build forget/retain loaders from full training windows.
        Step 3: Compute FIM on retain set using isolated graph.
        Step 4: Generate surrogate data via neighbor-informed imputation.
        Step 5: Train with unified loop (gradient ascent on forget data).
        """
        dataset = dataset.astype(np.float32)

        print(f"Isolating faulty node {faulty_node_idx} in adjacency matrix...")
        self.new_A_wave[faulty_node_idx, :] = 0.0
        self.new_A_wave[:, faulty_node_idx] = 0.0

        print("Building forget/retain loaders...")
        forget_loader, retain_loader = self._build_node_unlearning_loaders(
            dataset, faulty_node_idx, num_timesteps_input, num_timesteps_output, batch_size
        )
        print(f"  Forget: {len(forget_loader.dataset)} samples (faulty node outputs)")
        print(f"  Retain: {len(retain_loader.dataset)} samples (all other nodes)")

        print("Computing Population-Aware FIM on retain set...")
        self.fim_diagonal = self.pa_ewc.calculate_pa_fim(
            self.model, retain_loader, self.new_A_wave, max_samples=500
        )

        print("Generating surrogate data via neighbor-informed imputation...")
        surrogate_loader = self._build_surrogate_data_for_node(
            dataset, faulty_node_idx, num_timesteps_input, num_timesteps_output, batch_size
        )
        print(f"  Surrogate: {len(surrogate_loader.dataset)} samples")

        print("Starting unlearning training...")
        history = self.training(
            surrogate_loader, retain_loader, forget_loader,
            num_epochs, learning_rate,
            lambda_ewc, lambda_surrogate, lambda_retain, lambda_forget,
            faulty_node_idx=faulty_node_idx
        )

        return history, forget_loader, retain_loader

    # ------------------ UNIFIED TRAINING LOOP  (FIX-3, FIX-4, FIX-5) ---------------------------
    def training(self, surrogate_loader, retain_loader, forget_loader,
                 num_epochs=50, learning_rate=5e-5,
                 lambda_ewc=10.0, lambda_surrogate=1.0, lambda_retain=1.0,
                 lambda_forget=0.5, faulty_node_idx=None):
        """
        Unified training loop for both subset and node unlearning.

        Objective (matches updated paper):
            L_total = λ_surr * L_surrogate
                    + λ_retain * L_retain
                    + λ_ewc * L_ewc
                    − λ_forget * L_forget          ← gradient ascent on forget data

        For subset unlearning  : faulty_node_idx=None → loss over all nodes.
        For node unlearning    : faulty_node_idx=<int> → surrogate/forget loss
                                 scoped to that node; retain loss excludes it.

        All forward passes use self.new_A_wave (isolated graph for node path,
        original graph copy for subset path).
        """
        print("Unlearn training...")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        mse = nn.MSELoss()
        history = {
            'total_loss': [], 'surrogate_loss': [],
            'retain_loss': [], 'forget_loss': [], 'ewc_penalty': []
        }

        new_A = self.new_A_wave.to(self.device)
        N = None  # resolved on first batch

        for epoch in range(num_epochs):
            self.model.train()
            epoch_stats = {k: 0.0 for k in history}
            n_batches = 0

            # Forget loader is cycled independently so its length never limits
            # the number of updates (surrogate and retain loaders drive the loop)
            forget_iter = iter(forget_loader) if forget_loader is not None else None

            for (surr_X, surr_y), (ret_X, ret_y) in zip(surrogate_loader, retain_loader):
                optimizer.zero_grad()

                surr_X  = surr_X.float().to(self.device)
                surr_y  = surr_y.float().to(self.device)
                ret_X   = ret_X.float().to(self.device)
                ret_y   = ret_y.float().to(self.device)

                if N is None:
                    N = surr_X.shape[1]

                # ---- Surrogate loss ----
                surr_pred = self.model(new_A, surr_X)
                if faulty_node_idx is not None:
                    surr_pred = surr_pred[:, faulty_node_idx:faulty_node_idx+1, :, :]
                l_surrogate = mse(surr_pred, surr_y)

                # ---- Retain loss ----
                ret_pred = self.model(new_A, ret_X)
                if faulty_node_idx is not None:
                    node_mask = torch.ones(N, dtype=torch.bool, device=self.device)
                    node_mask[faulty_node_idx] = False
                    l_retain = mse(ret_pred[:, node_mask, :, :], ret_y[:, node_mask, :, :])
                else:
                    l_retain = mse(ret_pred, ret_y)

                # ---- Forget loss (gradient ascent) ----
                l_forget = torch.tensor(0.0, device=self.device)
                if forget_iter is not None:
                    try:
                        fgt_X, fgt_y = next(forget_iter)
                    except StopIteration:
                        forget_iter = iter(forget_loader)
                        fgt_X, fgt_y = next(forget_iter)

                    fgt_X = fgt_X.float().to(self.device)
                    fgt_y = fgt_y.float().to(self.device)
                    fgt_pred = self.model(new_A, fgt_X)
                    if faulty_node_idx is not None:
                        fgt_pred = fgt_pred[:, faulty_node_idx:faulty_node_idx+1, :, :]
                    l_forget = mse(fgt_pred, fgt_y)

                # ---- EWC penalty ----
                l_ewc = torch.tensor(0.0, device=self.device)
                if self.fim_diagonal is not None:
                    l_ewc = self.pa_ewc.apply_ewc_penalty(
                        self.model, self.fim_diagonal, self.original_params, lambda_ewc
                    )

                # ---- Combined objective ----
                total_loss = (
                    lambda_surrogate * l_surrogate
                    + lambda_retain   * l_retain
                    + lambda_ewc      * l_ewc
                    - lambda_forget   * l_forget   # subtract = maximize forget error
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_stats['total_loss']    += total_loss.item()
                epoch_stats['surrogate_loss'] += l_surrogate.item()
                epoch_stats['retain_loss']    += l_retain.item()
                epoch_stats['forget_loss']    += l_forget.item()
                epoch_stats['ewc_penalty']    += l_ewc.item()
                n_batches += 1

            if n_batches > 0:
                for k in history:
                    history[k].append(epoch_stats[k] / n_batches)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}:")
                for k, v in history.items():
                    print(f"  {k}: {v[-1]:.6f}")
                print("\n===================================================\n")

        return history

    # -------------------------- OLD STUFF (NO LONGER USED) --------------------------
    def compute_sa_ts_objective(self, surrogate_batch, retain_batch,
                                lambda_ewc, lambda_surrogate, lambda_retain, A_wave):
        """
        NO longer called by either training path.
        Kept here only for backward compatibility with external callers.
        The unified training() loop above supersedes it.
        """
        surrogate_features, surrogate_target = surrogate_batch
        retain_features, retain_target = retain_batch

        surrogate_features = surrogate_features.float().to(self.device)
        surrogate_target   = surrogate_target.float().to(self.device)
        retain_features    = retain_features.float().to(self.device)
        retain_target      = retain_target.float().to(self.device)
        A_wave             = A_wave.float().to(self.device)

        surrogate_pred = self.model(A_wave, surrogate_features)
        retain_pred    = self.model(A_wave, retain_features)

        mse_loss       = nn.MSELoss()
        surrogate_loss = mse_loss(surrogate_pred, surrogate_target)
        retain_loss    = mse_loss(retain_pred,    retain_target)

        ewc_penalty = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        if self.fim_diagonal is not None:
            ewc_penalty = self.pa_ewc.apply_ewc_penalty(
                self.model, self.fim_diagonal, self.original_params, lambda_ewc
            )

        total_loss = (
            lambda_surrogate * surrogate_loss
            + lambda_ewc     * ewc_penalty
            + lambda_retain  * retain_loss
        )
        return {
            'total_loss':    total_loss,
            'surrogate_loss': surrogate_loss,
            'ewc_penalty':   ewc_penalty,
            'retain_loss':   retain_loss
        }


# -------------------------- MAIN ---------------------------
def main():
    torch.cuda.empty_cache()

    print("Loading PEMS-BAY data...")
    A, train_original_data, test_original_data, means, stds = load_data_PEMS_BAY(args.input)
    means = means.astype(np.float32)
    stds  = stds.astype(np.float32)

    # FIX-6: correctly extract forget_set array and node_idx for subset path
    forget_set = None
    if args.forget_set and not args.unlearn_node:
        with open(args.forget_set, 'r', encoding='utf8') as f:
            forget_set_json = json.load(f)
        for key, value in forget_set_json.items():
            node_idx = int(key)
            for item in value:
                # Extract the speed feature (index 1) for the forget segment
                forget_set = train_original_data[node_idx, 1, item[0]:item[1]]
            args.node_idx = node_idx
            break  # only handle the first node entry

    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave).float().to(args.device)

    checkpoint = torch.load(
        args.model + f"/{args.type}_model.pt", map_location=args.device
    )
    if args.type == 'stgcn':
        model = STGCN(**checkpoint["config"]).to(args.device)
    elif args.type == 'stgat':
        model = STGAT(**checkpoint["config"]).to(args.device)
    else:
        raise ValueError(f"Unknown model type: {args.type}")

    model.load_state_dict(
        {k: v.float() for k, v in checkpoint["model_state_dict"].items()}
    )

    config = checkpoint["config"]
    num_timesteps_input  = config["nums_step_in"]
    num_timesteps_output = config["nums_step_out"]

    test_input, test_target = generate_dataset(
        test_original_data,
        num_timesteps_input=num_timesteps_input,
        num_timesteps_output=num_timesteps_output
    )
    # shuffle=False: test set is for evaluation only, order doesn't matter
    # but determinism is better for reproducibility
    test_loader = DataLoader(
        TensorDataset(test_input, test_target),
        batch_size=512, shuffle=False
    )

    sa_ts = SATimeSeries(model, A_wave, args.device)

    if args.unlearn_node:
        history, forget_loader, retain_loader = sa_ts.unlearn_faulty_node(
            train_original_data, args.node_idx, A_wave, means, stds,
            num_timesteps_input, num_timesteps_output,
            num_epochs=100, learning_rate=1e-4,
            lambda_ewc=5.0, lambda_surrogate=1.0,
            lambda_retain=2.0, lambda_forget=1.0,
            batch_size=256
        )
    else:
        # FIX-7: pass forget_set (the extracted array), not the undefined name
        history, forget_loader, retain_loader = sa_ts.unlearn_faulty_subset(
            train_original_data, forget_set, args.node_idx, A_wave, means, stds,
            num_timesteps_input, num_timesteps_output,
            threshold=0.5, num_epochs=100, learning_rate=1e-5,
            lambda_ewc=5.0, lambda_surrogate=0.5,
            lambda_retain=1.0, lambda_forget=0.5,
            batch_size=512
        )

    if not history or forget_loader is None or retain_loader is None:
        print("No unlearning performed. Exiting.")
        return

    path = (
        args.model + f"/Unlearn node {args.node_idx}"
        if args.unlearn_node
        else args.model + f"/Unlearn subset on node {args.node_idx}"
    )
    os.makedirs(path, exist_ok=True)

    torch.save(
        {"model_state_dict": sa_ts.model.state_dict(), "config": sa_ts.model.config},
        path + "/model.pt"
    )

    print("\nEvaluating unlearned model...")
    faulty_node_idx = args.node_idx if args.unlearn_node else None
    evaluation_results = evaluate_unlearning(
        model_unlearned=sa_ts.model,
        model_original=sa_ts.original_model,
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        test_loader=test_loader,
        new_A_wave=sa_ts.new_A_wave,
        A_wave=A_wave,
        device=args.device,
        faulty_node_idx=faulty_node_idx
    )

    with open(path + "/evaluation_results.txt", "w") as f:
        for metric, value in evaluation_results.items():
            f.write(f"{metric}: {value:.4f}\n")

    print("\n--- Evaluation Results ---")
    for metric, value in evaluation_results.items():
        print(f"  {metric}: {value:.4f}")
    print("--------------------------\n")

    torch.save({
        'model_state_dict': sa_ts.model.state_dict(),
        'history': history,
        'faulty_node_idx': args.node_idx
    }, args.model + f"/{args.type}_unlearned_model.pt")

    print("Unlearning completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UvS-TS Unlearning')
    parser.add_argument('--enable-cuda',  action='store_true')
    parser.add_argument('--unlearn-node', action='store_true')
    parser.add_argument('--node-idx',     type=int, help='Node index to unlearn (required for node mode)')
    parser.add_argument('--input',        type=str, required=True)
    parser.add_argument('--type',         type=str, required=True, choices=['stgcn', 'stgat', 'gwn'])
    parser.add_argument('--model',        type=str, required=True)
    parser.add_argument('--forget-set',   type=str)

    args = parser.parse_args()

    if args.unlearn_node:
        if args.node_idx is None:
            parser.error("--node-idx is required when --unlearn-node is set.")
        if args.forget_set is not None:
            print("Warning: --forget-set is ignored in node unlearning mode.")
    else:
        if args.forget_set is None:
            parser.error("--forget-set is required unless --unlearn-node is specified.")

    args.device = (
        torch.device('cuda') if args.enable_cuda and torch.cuda.is_available()
        else torch.device('cpu')
    )

    main()