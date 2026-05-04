"""
UvS-TS Unlearning — consolidated implementation.

Replaces: unlearn.py, unlearn_2.py, unlearn_3.py, unlearn_logic_2.py

Key design choices (vs. the three predecessors):

  1. ONE unified training loop. Three clearly-named loaders:
        surrogate_loader  - what the model should LEARN (new targets)
        retain_loader     - clean retain windows (preserve performance)
        forget_loader_tr  - original forget windows for gradient ascent
     These are never reused or swapped.

  2. TWO forget loaders:
        forget_loader_tr   - used INSIDE training for −λ_forget·L_forget
        forget_loader_eval - used ONLY for evaluation, ALWAYS holds the
                             ORIGINAL (unmodified) forget targets so
                             forget-set MSE measures real forgetting,
                             not "how well we fit our own surrogates"
                             (this was the tautology bug in unlearn_3).

  3. TWO surrogate strategies for SUBSET unlearning, selectable via
     --surrogate-mode:
        'self'     - self-imputation via T-GR (the original paper's path).
                     Kept for ablation / backward compatibility.
        'patch'    - real forget inputs with PARTIALLY-patched targets
                     (only forget-timestep slots in the output horizon are
                     overwritten with surrogate values; the rest stay real).
                     Avoids the self-distillation circularity and is what
                     the research review recommended as the safer default.

  4. Bounded forget-ascent term (margin loss). Instead of raw
     −λ_forget·L_forget (which diverges on MSE regression — documented
     in Zhang 2024 "Negative Preference Optimization"), we use:

        L_forget_term = λ_forget · max(0, margin − L_forget)

     This stops contributing once forget MSE exceeds the margin, preventing
     catastrophic collapse. Set `forget_margin` via the CLI.

  5. Graph WaveNet node-vector isolation. When node-unlearning a gwnet
     model, we also zero rows/cols of nodevec1/nodevec2 so the adaptive
     adjacency bypass is actually closed. This was the Graph WaveNet
     leak identified in the research review.

  6. PopulationAwareEWC receives the actual model_type (was hardcoded
     "stgcn" in all three predecessors, harmless but confusing).

  7. Consistent FIM / T-GR adjacency: both use self.new_A_wave.

  8. No prediction clamping (was silently dropping signal in unlearn_2/3).
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append("src")

from models.stgcn import STGCN
from models.stgat import STGAT
from models.gwn import gwnet
from utils.data_loader import load_data_PEMS_BAY
from unlearning.pa_ewc import PopulationAwareEWC
from unlearning.tgr_test import TemporalGenerativeReplay  # the newer T-GR
from unlearning.motif_def import discover_motifs_proxy
from utils.filter_forget import filter_forget
from utils.replace_surrogate import replace_target
from data.preprocess_pemsbay import generate_dataset, get_normalized_adj
from evaluate import evaluate_unlearning
from retrain_baseline import retrain_from_scratch


# ============================================================================
#  Core framework
# ============================================================================
class UvSTS:
    """Unified UvS-TS unlearning framework."""

    def __init__(self, model, A_wave, model_type: str, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        # --- Snapshot the original model for evaluation ---
        self.original_model = copy.deepcopy(model).to(self.device)
        self.original_model.eval()

        # --- Working copy that will be modified ---
        self.model = copy.deepcopy(model).to(self.device)
        for p in self.model.parameters():
            p.data = p.data.float()

        # --- Adjacency matrices ---
        self.A_wave = A_wave.float().to(self.device)
        self.new_A_wave = copy.deepcopy(A_wave).float().to(self.device)

        # --- EWC state ---
        self.original_params = {
            n: p.data.clone().float() for n, p in self.model.named_parameters()
        }
        self.pa_ewc = PopulationAwareEWC(model_type, device)
        self.t_gr = TemporalGenerativeReplay(model_type)
        self.fim_diagonal = None

    # ------------------------------------------------------------------
    #  Graph WaveNet: isolate faulty node in the adaptive adjacency
    # ------------------------------------------------------------------
    def _isolate_gwnet_node_embeddings(self, faulty_node_idx: int):
        """
        Closes the adaptive-adjacency leak for Graph WaveNet.

        gwnet forms an adaptive adjacency as
            Ã_adp = softmax(relu(nodevec1 @ nodevec2))
        If we only zero the static A_wave but leave nodevec1/nodevec2 intact,
        the faulty node is still connected through the learned adaptive path.
        """
        if not isinstance(self.model, gwnet):
            return
        with torch.no_grad():
            if hasattr(self.model, "nodevec1") and self.model.nodevec1 is not None:
                self.model.nodevec1.data[faulty_node_idx, :] = 0.0
            if hasattr(self.model, "nodevec2") and self.model.nodevec2 is not None:
                self.model.nodevec2.data[:, faulty_node_idx] = 0.0
        print(f"  [gwnet] zeroed nodevec1[{faulty_node_idx},:] and "
              f"nodevec2[:,{faulty_node_idx}]")

    # ==================================================================
    #  SUBSET UNLEARNING
    # ==================================================================
    def unlearn_subset(
        self,
        dataset,
        forget_example,
        faulty_node_idx,
        means,
        stds,
        num_timesteps_input,
        num_timesteps_output,
        threshold=0.5,
        num_epochs=100,
        learning_rate=1e-5,
        lambda_ewc=5.0,
        lambda_surrogate=1.0,
        lambda_retain=1.0,
        lambda_forget=0.5,
        forget_margin=2.0,
        batch_size=128,
        surrogate_mode="patch",
    ):
        """
        Subset unlearning on a single faulty node's temporal segments.

        surrogate_mode:
            'patch' - recommended default. Train on real forget INPUTS with
                      PARTIALLY patched targets (only forget-timestep slots
                      in the output horizon are replaced with surrogates).
                      Preserves real input context, no self-distillation loop.
            'self'  - legacy path. Build synthetic surrogate (input, target)
                      pairs from the model's own predictions on forget data.
        """
        print(f"\n[SUBSET] Starting on node {faulty_node_idx}")
        print(f"  surrogate_mode={surrogate_mode}  epochs={num_epochs}  lr={learning_rate}")
        print(f"  λ_surr={lambda_surrogate}  λ_retain={lambda_retain}  "
              f"λ_ewc={lambda_ewc}  λ_forget={lambda_forget}  margin={forget_margin}")

        dataset = dataset.astype(np.float32)
        forget_example = forget_example.astype(np.float32)

        train_input, train_target = generate_dataset(
            dataset, num_timesteps_input, num_timesteps_output
        )

        # ---- 1. DTW-based partitioning ----
        forget_indices, retain_indices = discover_motifs_proxy(
            dataset, forget_example, faulty_node_idx, threshold
        )
        print(f"  forget segments: {len(forget_indices)}  "
              f"retain segments: {len(retain_indices)}")
        if not forget_indices:
            print("  no forget windows found — aborting")
            return None, None, None, None

        # ---- 2. Split windows into forget / retain by overlap ----
        retain_input, retain_target, _, forget_input, forget_target_orig, Df_idx = (
            filter_forget(
                train_input, train_target, forget_indices,
                num_timesteps_input, num_timesteps_output,
            )
        )
        print(f"  retain windows: {len(retain_input)}  forget windows: {len(forget_input)}")

        retain_loader = DataLoader(
            TensorDataset(retain_input, retain_target),
            batch_size=batch_size, shuffle=True,
        )

        # ---- 3. Build EVALUATION forget loader using ORIGINAL targets ----
        #       (this is the fix for the unlearn_3 tautology bug)
        forget_loader_eval = DataLoader(
            TensorDataset(forget_input, forget_target_orig),
            batch_size=batch_size, shuffle=False,
        )

        # ---- 4. Compute FIM on retain set with consistent adjacency ----
        print("  computing PA-FIM on retain set...")
        self.fim_diagonal = self.pa_ewc.calculate_pa_fim(
            self.model, retain_loader, self.new_A_wave, max_samples=500
        )

        # ---- 5. Generate surrogate data via T-GR ----
        print(f"  generating surrogates via T-GR (mode={surrogate_mode})...")
        surrogate_segs = self.t_gr.perform_temporal_generative_replay_subset(
            self.model, forget_input, forget_indices, faulty_node_idx,
            num_timesteps_input, num_timesteps_output, self.device, self.new_A_wave,
        )

        # ---- 6. Build TRAINING surrogate / forget loaders per mode ----
        if surrogate_mode == "patch":
            # Patch only the forget-timestep slots inside forget_target.
            forget_target_patched = replace_target(
                forget_target_orig.clone(), surrogate_segs, Df_idx,
                forget_indices, faulty_node_idx, num_timesteps_input,
            )
            # surrogate_loader = real forget inputs, patched targets
            surrogate_loader = DataLoader(
                TensorDataset(forget_input, forget_target_patched),
                batch_size=batch_size, shuffle=True,
            )
            # forget_loader_tr (for gradient ascent) = original targets
            forget_loader_tr = DataLoader(
                TensorDataset(forget_input, forget_target_orig),
                batch_size=batch_size, shuffle=True,
            )
        elif surrogate_mode == "self":
            # Legacy path: build synthetic (input, target) from raw segments.
            surr_features, surr_targets = [], []
            for seg in surrogate_segs:
                if isinstance(seg, torch.Tensor):
                    seg_np = seg.cpu().numpy()
                else:
                    seg_np = np.asarray(seg)
                seg_np = seg_np.astype(np.float32)
                # shape expected by generate_dataset: (N, F, T)
                if seg_np.ndim == 2:
                    seg_np = seg_np[np.newaxis, :, :]  # (1, F, T) or (1, T, F)
                feat, tgt = generate_dataset(
                    seg_np, num_timesteps_input, num_timesteps_output
                )
                if feat.numel() > 0:
                    surr_features.append(feat)
                    surr_targets.append(tgt)
            if not surr_features:
                print("  'self' mode produced no surrogate windows — falling back to 'patch'")
                return self.unlearn_subset(
                    dataset, forget_example, faulty_node_idx, means, stds,
                    num_timesteps_input, num_timesteps_output,
                    threshold, num_epochs, learning_rate,
                    lambda_ewc, lambda_surrogate, lambda_retain, lambda_forget,
                    forget_margin, batch_size, surrogate_mode="patch",
                )
            surrogate_loader = DataLoader(
                TensorDataset(torch.cat(surr_features), torch.cat(surr_targets)),
                batch_size=batch_size, shuffle=True,
            )
            forget_loader_tr = DataLoader(
                TensorDataset(forget_input, forget_target_orig),
                batch_size=batch_size, shuffle=True,
            )
        else:
            raise ValueError(f"Unknown surrogate_mode: {surrogate_mode}")

        # ---- 7. Unified training loop ----
        # Pass faulty_node_idx so the surrogate-slicing guard fires when
        # 'self' mode emits (B, 1, T, F) surrogate targets; retain loss
        # still runs over all nodes because mask_faulty_in_retain=False.
        history = self._training_loop(
            surrogate_loader, retain_loader, forget_loader_tr,
            num_epochs, learning_rate,
            lambda_ewc, lambda_surrogate, lambda_retain, lambda_forget,
            forget_margin, faulty_node_idx=faulty_node_idx,
            mask_faulty_in_retain=False,
        )

        return history, retain_loader, forget_loader_tr, forget_loader_eval

    # ==================================================================
    #  NODE UNLEARNING
    # ==================================================================
    def _compute_leverage_scores(self, full_input, faulty_node_idx, batch_size):
        """
        Per-window leverage score for node v:
            score(w) = mean( (f(A,w)[:,v] - f(A, mask_v(w))[:,v])^2 )
        High score = v's own input is load-bearing for v's output in this
        window. Used by node_mode='leverage' to prune D_f to high-influence
        windows, cutting node-mode runtime without weakening the forget signal.
        """
        self.model.eval()
        scores = torch.empty(full_input.shape[0], dtype=torch.float32)
        with torch.no_grad():
            for i in range(0, full_input.shape[0], batch_size):
                X = full_input[i:i + batch_size].to(self.device)
                pred_real = self.model(self.A_wave, X)[
                    :, faulty_node_idx:faulty_node_idx + 1, :, :
                ]
                X_masked = X.clone()
                X_masked[:, faulty_node_idx, :, :] = 0.0
                pred_masked = self.model(self.A_wave, X_masked)[
                    :, faulty_node_idx:faulty_node_idx + 1, :, :
                ]
                diff = (pred_real - pred_masked).pow(2).mean(dim=(1, 2, 3))
                scores[i:i + diff.shape[0]] = diff.cpu()
        return scores

    def unlearn_node(
        self,
        dataset,
        faulty_node_idx,
        means,
        stds,
        num_timesteps_input,
        num_timesteps_output,
        num_epochs=100,
        learning_rate=1e-4,
        lambda_ewc=5.0,
        lambda_surrogate=1.0,
        lambda_retain=2.0,
        lambda_forget=1.0, # 1.0 is best
        forget_margin=2.0, # 0.5 is best
        batch_size=256,
        # ---- scope-narrowing controls (suggested in the methodology review) ----
        node_mode="full",          # "full" | "leverage" | "motif"
        leverage_keep=0.2,         # top fraction of windows kept in 'leverage' mode
        motif_example=None,        # 1D array, faulty pattern on node v's series
        motif_threshold=0.5,       # DTW threshold for 'motif' mode
    ):
        """
        Node unlearning: remove a sensor's influence from the model.

        node_mode:
            'full'     - paper-default. D_f = ALL windows × {node v}, graph
                         isolated. Slow but matches the "sensor permanently
                         removed" scenario.
            'leverage' - same as 'full' but D_f is pruned to the top
                         `leverage_keep` fraction of windows by influence
                         score (windows where v's own input is load-bearing
                         for v's output). Cuts wall-clock by ~1/leverage_keep
                         without sacrificing forget signal.
            'motif'    - "sensor went bad after date X" scenario. DTW finds
                         windows whose v-series matches `motif_example`;
                         D_f = those windows, D_r = the rest with v's data
                         INTACT. Graph is NOT isolated, since v stays in
                         operation outside the faulty period.
        """
        if node_mode not in ("full", "leverage", "motif"):
            raise ValueError(f"Unknown node_mode: {node_mode}")
        if node_mode == "motif" and motif_example is None:
            raise ValueError("node_mode='motif' requires motif_example")

        print(f"\n[NODE] Starting on node {faulty_node_idx}  mode={node_mode}")

        dataset = dataset.astype(np.float32)

        # ---- 1. Build windows ----
        full_input, full_target = generate_dataset(
            dataset, num_timesteps_input, num_timesteps_output
        )

        # ============================================================
        #  Branch: motif-restricted scope (no graph isolation)
        # ============================================================
        if node_mode == "motif":
            print(f"  motif DTW: threshold={motif_threshold}, "
                  f"reference len={len(motif_example)}")
            forget_intervals, _ = discover_motifs_proxy(
                dataset, motif_example, faulty_node_idx, motif_threshold
            )
            if not forget_intervals:
                print("  no faulty motifs found — aborting")
                return None, None, None, None
            print(f"  faulty intervals: {len(forget_intervals)}")

            (retain_input, retain_target, _, forget_input,
             forget_target_full, _) = filter_forget(
                full_input, full_target, forget_intervals,
                num_timesteps_input, num_timesteps_output,
            )
            # In motif mode the forget target is sliced to node v
            forget_target = forget_target_full[
                :, faulty_node_idx:faulty_node_idx + 1, :, :
            ]
            print(f"  retain windows: {len(retain_input)}  "
                  f"forget windows: {len(forget_input)}")

            # Surrogate: neighbor-informed but ONLY on the faulty windows
            # (rest of v's history is kept intact via retain).
            print("  generating neighbor-informed surrogate on faulty windows...")
            masked_input = forget_input.clone()
            masked_input[:, faulty_node_idx, :, :] = 0.0
            self.model.eval()
            surrogate_chunks = []
            with torch.no_grad():
                for i in range(0, masked_input.shape[0], batch_size):
                    batch = masked_input[i:i + batch_size].to(self.device)
                    out = self.model(self.A_wave, batch)
                    surrogate_chunks.append(
                        out[:, faulty_node_idx:faulty_node_idx + 1, :, :].cpu()
                    )
            surrogate_target = torch.cat(surrogate_chunks, dim=0)
            surrogate_target = surrogate_target + (
                torch.randn_like(surrogate_target) * 0.05
                + torch.randn_like(surrogate_target) * 0.10
            )

            # NOTE: graph stays connected — v is still operational.
            print("  graph isolation SKIPPED (motif mode)")

            forget_loader_tr = DataLoader(
                TensorDataset(forget_input, forget_target),
                batch_size=batch_size, shuffle=True,
            )
            forget_loader_eval = DataLoader(
                TensorDataset(forget_input, forget_target),
                batch_size=batch_size, shuffle=False,
            )
            retain_loader = DataLoader(
                TensorDataset(retain_input, retain_target),
                batch_size=batch_size, shuffle=True,
            )
            surrogate_loader = DataLoader(
                TensorDataset(masked_input, surrogate_target),
                batch_size=batch_size, shuffle=True,
            )

            print("  computing PA-FIM on retain set (original graph)...")
            self.fim_diagonal = self.pa_ewc.calculate_pa_fim(
                self.model, retain_loader, self.new_A_wave, max_samples=500
            )

            history = self._training_loop(
                surrogate_loader, retain_loader, forget_loader_tr,
                num_epochs, learning_rate,
                lambda_ewc, lambda_surrogate, lambda_retain, lambda_forget,
                forget_margin, faulty_node_idx=faulty_node_idx,
                mask_faulty_in_retain=False,   # v is intact in retain
            )
            return history, retain_loader, forget_loader_tr, forget_loader_eval

        # ============================================================
        #  Modes 'full' and 'leverage' share the rest of the pipeline
        # ============================================================

        # ---- 2. Neighbor-informed surrogate BEFORE isolation ----
        # Must happen before zeroing the faulty row/col of the adjacency and
        # the gwnet nodevecs; otherwise the faulty node's output degenerates
        # to ~0 and the "imputation" collapses to pure noise.
        print("  generating neighbor-informed surrogate (pre-isolation)...")
        masked_input = full_input.clone()
        masked_input[:, faulty_node_idx, :, :] = 0.0

        self.model.eval()
        surrogate_chunks = []
        with torch.no_grad():
            for i in range(0, masked_input.shape[0], batch_size):
                batch = masked_input[i:i + batch_size].to(self.device)
                out = self.model(self.A_wave, batch)   # ORIGINAL adjacency
                surrogate_chunks.append(
                    out[:, faulty_node_idx:faulty_node_idx + 1, :, :].cpu()
                )
        surrogate_target = torch.cat(surrogate_chunks, dim=0)
        # Multi-scale noise to prevent re-learning the original pattern
        surrogate_target = surrogate_target + (
            torch.randn_like(surrogate_target) * 0.05
            + torch.randn_like(surrogate_target) * 0.10
        )

        # ---- 2b. Leverage-based subsampling of D_f (and matching surrogate) ----
        keep_indices = None
        if node_mode == "leverage":
            print(f"  scoring window leverage for node {faulty_node_idx}...")
            scores = self._compute_leverage_scores(
                full_input, faulty_node_idx, batch_size
            )
            n_keep = max(1, int(round(leverage_keep * scores.shape[0])))
            keep_indices = torch.topk(scores, n_keep).indices.sort().values
            print(f"  leverage_keep={leverage_keep}  "
                  f"kept {n_keep}/{scores.shape[0]} windows  "
                  f"score range=[{scores[keep_indices].min().item():.4g}, "
                  f"{scores[keep_indices].max().item():.4g}]")

        # ---- 3. Isolate faulty node in adjacency / gwnet embeddings ----
        print(f"  isolating node {faulty_node_idx} in adjacency...")
        self.new_A_wave[faulty_node_idx, :] = 0.0
        self.new_A_wave[:, faulty_node_idx] = 0.0
        self._isolate_gwnet_node_embeddings(faulty_node_idx)

        # ---- 4. Build loaders ----
        if keep_indices is not None:
            forget_input = full_input[keep_indices].clone()
            forget_target = full_target[keep_indices][
                :, faulty_node_idx:faulty_node_idx + 1, :, :
            ]
            masked_input = masked_input[keep_indices]
            surrogate_target = surrogate_target[keep_indices]
        else:
            forget_input = full_input.clone()
            forget_target = full_target[:, faulty_node_idx:faulty_node_idx + 1, :, :]

        forget_loader_tr = DataLoader(
            TensorDataset(forget_input, forget_target),
            batch_size=batch_size, shuffle=True,
        )
        forget_loader_eval = DataLoader(
            TensorDataset(forget_input, forget_target),
            batch_size=batch_size, shuffle=False,
        )

        retain_input = full_input.clone()
        retain_target = full_target.clone()
        retain_input[:, faulty_node_idx, :, :] = 0.0
        retain_target[:, faulty_node_idx, :, :] = 0.0
        retain_loader = DataLoader(
            TensorDataset(retain_input, retain_target),
            batch_size=batch_size, shuffle=True,
        )

        surrogate_loader = DataLoader(
            TensorDataset(masked_input, surrogate_target),
            batch_size=batch_size, shuffle=True,
        )

        print(f"  retain windows: {len(retain_input)}  "
              f"forget windows: {len(forget_input)}")

        # ---- 5. FIM on retain set (isolated graph) ----
        print("  computing PA-FIM on retain set (isolated graph)...")
        self.fim_diagonal = self.pa_ewc.calculate_pa_fim(
            self.model, retain_loader, self.new_A_wave, max_samples=500
        )

        # ---- 6. Unified training loop ----
        history = self._training_loop(
            surrogate_loader, retain_loader, forget_loader_tr,
            num_epochs, learning_rate,
            lambda_ewc, lambda_surrogate, lambda_retain, lambda_forget,
            forget_margin, faulty_node_idx=faulty_node_idx,
            mask_faulty_in_retain=True,
        )

        return history, retain_loader, forget_loader_tr, forget_loader_eval

    # ==================================================================
    #  UNIFIED TRAINING LOOP
    # ==================================================================
    def _training_loop(
        self,
        surrogate_loader,
        retain_loader,
        forget_loader_tr,
        num_epochs,
        learning_rate,
        lambda_ewc,
        lambda_surrogate,
        lambda_retain,
        lambda_forget,
        forget_margin,
        faulty_node_idx=None,
        mask_faulty_in_retain=False,
    ):
        """
        Shared loop for subset + node unlearning.

        Objective:
            L = λ_surr·L_surr + λ_retain·L_retain + λ_ewc·L_ewc
                + λ_forget·max(0, margin − L_forget)     ← BOUNDED ascent

        The bounded forget term stops pushing once L_forget exceeds the
        margin, preventing catastrophic collapse on MSE regression.
        """
        print("\n[TRAIN] starting unified unlearning loop")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        mse = nn.MSELoss()
        history = {
            "total_loss": [], "surrogate_loss": [], "retain_loss": [],
            "forget_loss": [], "forget_term": [], "ewc_penalty": [],
        }

        new_A = self.new_A_wave.to(self.device)
        N = None

        for epoch in range(num_epochs):
            self.model.train()
            stats = {k: 0.0 for k in history}
            n_batches = 0

            forget_iter = iter(forget_loader_tr) if forget_loader_tr is not None else None

            for (surr_X, surr_y), (ret_X, ret_y) in zip(surrogate_loader, retain_loader):
                optimizer.zero_grad()

                surr_X = surr_X.float().to(self.device)
                surr_y = surr_y.float().to(self.device)
                ret_X = ret_X.float().to(self.device)
                ret_y = ret_y.float().to(self.device)

                if N is None:
                    N = surr_X.shape[1]

                # ---- Surrogate loss ----
                surr_pred = self.model(new_A, surr_X)
                if faulty_node_idx is not None and surr_pred.shape[1] > 1 and surr_y.shape[1] == 1:
                    surr_pred = surr_pred[:, faulty_node_idx:faulty_node_idx + 1, :, :]
                l_surr = mse(surr_pred, surr_y)

                # ---- Retain loss ----
                ret_pred = self.model(new_A, ret_X)
                if mask_faulty_in_retain and faulty_node_idx is not None:
                    mask = torch.ones(N, dtype=torch.bool, device=self.device)
                    mask[faulty_node_idx] = False
                    l_retain = mse(ret_pred[:, mask, :, :], ret_y[:, mask, :, :])
                else:
                    l_retain = mse(ret_pred, ret_y)

                # ---- Bounded forget term ----
                l_forget = torch.tensor(0.0, device=self.device)
                l_forget_term = torch.tensor(0.0, device=self.device)
                if forget_iter is not None:
                    try:
                        fgt_X, fgt_y = next(forget_iter)
                    except StopIteration:
                        forget_iter = iter(forget_loader_tr)
                        fgt_X, fgt_y = next(forget_iter)

                    fgt_X = fgt_X.float().to(self.device)
                    fgt_y = fgt_y.float().to(self.device)
                    fgt_pred = self.model(new_A, fgt_X)
                    if faulty_node_idx is not None and fgt_pred.shape[1] > 1 and fgt_y.shape[1] == 1:
                        fgt_pred = fgt_pred[:, faulty_node_idx:faulty_node_idx + 1, :, :]
                    l_forget = mse(fgt_pred, fgt_y)
                    # Margin loss: push L_forget up to `margin`, then stop.
                    l_forget_term = torch.clamp(forget_margin - l_forget, min=0.0)

                # ---- EWC penalty ----
                l_ewc = torch.tensor(0.0, device=self.device)
                if self.fim_diagonal is not None:
                    l_ewc = self.pa_ewc.apply_ewc_penalty(
                        self.model, self.fim_diagonal, self.original_params, lambda_ewc
                    )

                total = (
                    lambda_surrogate * l_surr
                    + lambda_retain * l_retain
                    + lambda_ewc * l_ewc
                    + lambda_forget * l_forget_term
                )

                total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                stats["total_loss"] += total.item()
                stats["surrogate_loss"] += l_surr.item()
                stats["retain_loss"] += l_retain.item()
                stats["forget_loss"] += l_forget.item()
                stats["forget_term"] += l_forget_term.item()
                stats["ewc_penalty"] += l_ewc.item()
                n_batches += 1

            if n_batches > 0:
                for k in history:
                    history[k].append(stats[k] / n_batches)

            if (epoch + 1) % 10 == 0:
                print(f"  epoch {epoch + 1}/{num_epochs}  "
                      f"total={history['total_loss'][-1]:.4f}  "
                      f"surr={history['surrogate_loss'][-1]:.4f}  "
                      f"retain={history['retain_loss'][-1]:.4f}  "
                      f"L_forget={history['forget_loss'][-1]:.4f}  "
                      f"forget_term={history['forget_term'][-1]:.4f}  "
                      f"ewc={history['ewc_penalty'][-1]:.4f}")

        # ---- Post-training diagnostic: did the forget term ever fire? ----
        if forget_loader_tr is not None and history["forget_loss"]:
            max_lf = max(history["forget_loss"])
            final_lf = history["forget_loss"][-1]
            final_term = history["forget_term"][-1]
            print(f"\n[DIAG] forget-set MSE  max={max_lf:.4f}  final={final_lf:.4f}  "
                  f"margin={forget_margin}  final_term={final_term:.4f}")
            if max_lf < forget_margin:
                print(f"  WARNING: L_forget never reached margin ({forget_margin}). "
                      f"Bounded ascent stayed active the whole run but produced no "
                      f"forgetting above the target. Consider:")
                print(f"    - raising --lambda-forget (current contribution may be "
                      f"drowned by EWC / retain / surrogate terms)")
                print(f"    - lowering --lambda-ewc (EWC pins params near original)")
                print(f"    - lowering --forget-margin toward a realistic target "
                      f"(e.g. 2-3x the original forget MSE)")
            elif final_term > 0.0:
                print(f"  NOTE: forget term still non-zero at end — more epochs may "
                      f"continue pushing L_forget up.")
            else:
                print(f"  OK: L_forget reached margin and the bounded term saturated.")

        return history


# ============================================================================
#  Driver
# ============================================================================
def _build_model(model_type, config, device):
    if model_type == "stgcn":
        return STGCN(**config).to(device)
    if model_type == "stgat":
        return STGAT(**config).to(device)
    if model_type == "gwnet":
        config = dict(config)
        config["device"] = device
        return gwnet(**config).to(device)
    raise ValueError(f"Unknown model_type: {model_type}")


def run(args):
    torch.cuda.empty_cache()
    print("Loading PEMS-BAY...")
    A, train_data, test_data, means, stds = load_data_PEMS_BAY(args.input)
    means = means.astype(np.float32)
    stds = stds.astype(np.float32)

    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave).float().to(args.device)

    # Extract forget example for subset mode (and node-mode='motif').
    # In subset mode the JSON's node id overrides args.node_idx; in node-mode
    # 'motif' the user has already pinned --node-idx, so we keep that but pull
    # the example slice from THAT node's series.
    forget_example = None
    needs_forget_set = (not args.unlearn_node) or (
        args.unlearn_node and args.node_mode == "motif"
    )
    if needs_forget_set:
        with open(args.forget_set, "r", encoding="utf8") as f:
            forget_set_json = json.load(f)
        if args.unlearn_node:
            node_for_motif = args.node_idx
            entries = forget_set_json.get(str(node_for_motif))
            if not entries:
                # fall back to the first available entry
                first_key = next(iter(forget_set_json))
                node_for_motif = int(first_key)
                entries = forget_set_json[first_key]
                print(f"  [motif] no entry for node {args.node_idx} in "
                      f"forget-set; using node {node_for_motif} as reference")
            item = entries[0]
            forget_example = train_data[node_for_motif, 0, item[0]:item[1]]
        else:
            for key, value in forget_set_json.items():
                node_idx = int(key)
                for item in value:
                    forget_example = train_data[node_idx, 0, item[0]:item[1]]
                args.node_idx = node_idx
                break

    # Load trained model
    ckpt_path = os.path.join(args.model, f"{args.type}_model.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=args.device)
    config = ckpt["config"]

    model = _build_model(args.type, config, args.device)
    model.load_state_dict({k: v.float() for k, v in ckpt["model_state_dict"].items()})

    num_in = config["nums_step_in"]
    num_out = config["nums_step_out"]

    # Test loader
    test_in, test_tg = generate_dataset(test_data, num_in, num_out)
    test_loader = DataLoader(
        TensorDataset(test_in, test_tg), batch_size=128, shuffle=False
    )

    # Framework
    framework = UvSTS(model, A_wave, args.type, args.device)

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()

    if args.unlearn_node:
        history, retain_loader, forget_loader_tr, forget_loader_eval = framework.unlearn_node(
            train_data, args.node_idx, means, stds, num_in, num_out,
            num_epochs=args.epochs, learning_rate=args.lr,
            lambda_ewc=args.lambda_ewc, lambda_surrogate=args.lambda_surr,
            lambda_retain=args.lambda_retain, lambda_forget=args.lambda_forget,
            forget_margin=args.forget_margin, batch_size=args.batch_size,
            node_mode=args.node_mode,
            leverage_keep=args.leverage_keep,
            motif_example=forget_example,
            motif_threshold=args.threshold,
        )
        if history is None:
            print("Nothing to unlearn (no faulty motifs found). Exiting.")
            return
    else:
        result = framework.unlearn_subset(
            train_data, forget_example, args.node_idx, means, stds,
            num_in, num_out, threshold=args.threshold,
            num_epochs=args.epochs, learning_rate=args.lr,
            lambda_ewc=args.lambda_ewc, lambda_surrogate=args.lambda_surr,
            lambda_retain=args.lambda_retain, lambda_forget=args.lambda_forget,
            forget_margin=args.forget_margin, batch_size=args.batch_size,
            surrogate_mode=args.surrogate_mode,
        )
        if result[0] is None:
            print("Nothing to unlearn. Exiting.")
            return
        history, retain_loader, forget_loader_tr, forget_loader_eval = result

    end_evt.record()
    torch.cuda.synchronize()
    elapsed = start_evt.elapsed_time(end_evt) / 1000.0

    # Save
    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    out_dir = os.path.join(
        args.model,
        f"Unlearn_{'node' if args.unlearn_node else 'subset'}_{args.node_idx}{suffix}",
    )
    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {"model_state_dict": framework.model.state_dict(),
         "config": framework.model.config},
        os.path.join(out_dir, f"{args.type}_unlearned.pt"),
    )

    # Optional: retrain-from-scratch gold-standard baseline.
    # NODE mode masks the faulty node in retain-loss (matches training). SUBSET
    # mode does not (retain_loader is already forget-window-filtered).
    mask_faulty = bool(args.unlearn_node)
    model_retrained = None
    if args.retrain_baseline:
        print("\n[RETRAIN] training gold-standard baseline from scratch on retain loader...")
        model_retrained = retrain_from_scratch(
            model_type=args.type,
            config=config,
            retain_loader=retain_loader,
            A_wave=framework.new_A_wave,
            device=args.device,
            epochs=args.retrain_epochs,
            lr=args.retrain_lr,
            faulty_node_idx=args.node_idx if args.unlearn_node else None,
            mask_faulty_in_retain=mask_faulty,
        )

    # Evaluate using the CLEAN eval forget loader
    print("\n[EVAL] evaluating unlearned model...")
    results = evaluate_unlearning(
        model_unlearned=framework.model,
        model_original=framework.original_model,
        retain_loader=retain_loader,
        forget_loader=forget_loader_eval,   # ORIGINAL forget targets
        test_loader=test_loader,
        new_A_wave=framework.new_A_wave,
        A_wave=A_wave,
        device=args.device,
        faulty_node_idx=args.node_idx if args.unlearn_node else None,
        mask_faulty_in_retain=mask_faulty,
        model_retrained=model_retrained,
        forget_margin=args.forget_margin,
    )
    results["time"] = elapsed

    eval_path = os.path.join(out_dir, f"evaluation_results_{args.type}.txt")
    with open(eval_path, "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")

    print("\n--- Evaluation Results ---")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    print(f"saved to {eval_path}")


def main():
    parser = argparse.ArgumentParser(description="UvS-TS unlearning (consolidated)")
    parser.add_argument("--enable-cuda", action="store_true")
    parser.add_argument("--unlearn-node", action="store_true")
    parser.add_argument("--node-idx", type=int)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--type", type=str, required=True,
                        choices=["stgcn", "stgat", "gwnet"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--forget-set", type=str)

    # Surrogate mode (subset only)
    parser.add_argument("--surrogate-mode", type=str, default="patch",
                        choices=["patch", "self"],
                        help="'patch' (default, recommended) or 'self' (legacy)")

    # Node-mode scope-narrowing (no-op in subset mode)
    parser.add_argument("--node-mode", type=str, default="full",
                        choices=["full", "leverage", "motif"],
                        help="Node-unlearning scope. 'full' (paper default) "
                             "uses every window. 'leverage' keeps the top "
                             "--leverage-keep fraction of windows by influence "
                             "score (faster, same isolation). 'motif' uses DTW "
                             "to find faulty windows on node v's series, does "
                             "NOT isolate the graph, and keeps v intact in "
                             "retain — for the 'sensor went bad' scenario.")
    parser.add_argument("--leverage-keep", type=float, default=0.2,
                        help="Top fraction of windows to keep as D_f when "
                             "--node-mode=leverage (default 0.2 = top 20%%).")

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="DTW threshold for motif discovery (subset mode)")
    parser.add_argument("--lambda-surr", type=float, default=1.0)
    parser.add_argument("--lambda-retain", type=float, default=1.0)
    parser.add_argument("--lambda-ewc", type=float, default=5.0)
    parser.add_argument("--lambda-forget", type=float, default=2.0)
    parser.add_argument("--forget-margin", type=float, default=0.5,
                        help="Bounded-ascent margin. Once L_forget exceeds "
                             "this, the forget term stops contributing. "
                             "Pick a realistic multiple of the original forget "
                             "MSE — too high causes generalization collapse.")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Optional suffix appended to the output directory "
                             "name (useful for hyperparameter sweeps).")

    # Retrain-from-scratch baseline (gold standard for unlearning eval).
    parser.add_argument("--retrain-baseline", action="store_true",
                        help="Train a fresh model on the retain loader and "
                             "report gap-to-retrain metrics. Expensive but "
                             "produces the defensible unlearning signal.")
    parser.add_argument("--retrain-epochs", type=int, default=100)
    parser.add_argument("--retrain-lr", type=float, default=1e-3)

    args = parser.parse_args()

    if args.unlearn_node:
        if args.node_idx is None:
            parser.error("--node-idx required with --unlearn-node")
        if args.node_mode == "motif":
            if args.forget_set is None:
                parser.error("--forget-set required when --node-mode=motif "
                             "(provides the reference faulty motif)")
        elif args.forget_set is not None:
            print("Warning: --forget-set ignored in node mode "
                  f"(--node-mode={args.node_mode})")
        if not 0.0 < args.leverage_keep <= 1.0:
            parser.error("--leverage-keep must be in (0, 1]")
    else:
        if args.forget_set is None:
            parser.error("--forget-set required unless --unlearn-node is set")

    args.device = (torch.device("cuda") if args.enable_cuda and torch.cuda.is_available()
                   else torch.device("cpu"))

    run(args)


if __name__ == "__main__":
    main()
