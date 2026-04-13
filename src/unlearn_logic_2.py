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
        history = self._training_loop(
            surrogate_loader, retain_loader, forget_loader_tr,
            num_epochs, learning_rate,
            lambda_ewc, lambda_surrogate, lambda_retain, lambda_forget,
            forget_margin, faulty_node_idx=None,  # subset: loss over all nodes
        )

        return history, retain_loader, forget_loader_tr, forget_loader_eval

    # ==================================================================
    #  NODE UNLEARNING
    # ==================================================================
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
        lambda_forget=1.0,
        forget_margin=2.0,
        batch_size=256,
    ):
        """Node unlearning: remove an entire sensor's influence."""
        print(f"\n[NODE] Starting on node {faulty_node_idx}")

        dataset = dataset.astype(np.float32)

        # ---- 1. Isolate faulty node in adjacency BEFORE any forward pass ----
        print(f"  isolating node {faulty_node_idx} in adjacency...")
        self.new_A_wave[faulty_node_idx, :] = 0.0
        self.new_A_wave[:, faulty_node_idx] = 0.0
        self._isolate_gwnet_node_embeddings(faulty_node_idx)

        # ---- 2. Build loaders ----
        full_input, full_target = generate_dataset(
            dataset, num_timesteps_input, num_timesteps_output
        )

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

        print(f"  retain windows: {len(retain_input)}  "
              f"forget windows: {len(forget_input)}")

        # ---- 3. FIM on retain set (isolated graph) ----
        print("  computing PA-FIM on retain set (isolated graph)...")
        self.fim_diagonal = self.pa_ewc.calculate_pa_fim(
            self.model, retain_loader, self.new_A_wave, max_samples=500
        )

        # ---- 4. Neighbor-informed surrogate ----
        print("  generating surrogate via neighbor-informed imputation...")
        masked_input = full_input.clone()
        masked_input[:, faulty_node_idx, :, :] = 0.0

        self.model.eval()
        surrogate_targets = []
        with torch.no_grad():
            for i in range(0, masked_input.shape[0], batch_size):
                batch = masked_input[i:i + batch_size].to(self.device)
                out = self.model(self.new_A_wave, batch)
                surrogate_targets.append(
                    out[:, faulty_node_idx:faulty_node_idx + 1, :, :].cpu()
                )
        surrogate_target = torch.cat(surrogate_targets, dim=0)
        # Multi-scale noise to prevent re-learning the original pattern
        surrogate_target = surrogate_target + (
            torch.randn_like(surrogate_target) * 0.05
            + torch.randn_like(surrogate_target) * 0.10
        )
        surrogate_loader = DataLoader(
            TensorDataset(masked_input, surrogate_target),
            batch_size=batch_size, shuffle=True,
        )

        # ---- 5. Unified training loop ----
        history = self._training_loop(
            surrogate_loader, retain_loader, forget_loader_tr,
            num_epochs, learning_rate,
            lambda_ewc, lambda_surrogate, lambda_retain, lambda_forget,
            forget_margin, faulty_node_idx=faulty_node_idx,
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
                if faulty_node_idx is not None:
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

    # Extract forget example for subset mode
    forget_example = None
    if not args.unlearn_node:
        with open(args.forget_set, "r", encoding="utf8") as f:
            forget_set_json = json.load(f)
        for key, value in forget_set_json.items():
            node_idx = int(key)
            for item in value:
                forget_example = train_data[node_idx, 1, item[0]:item[1]]
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
        )
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
    out_dir = os.path.join(
        args.model,
        f"Unlearn_{'node' if args.unlearn_node else 'subset'}_{args.node_idx}",
    )
    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {"model_state_dict": framework.model.state_dict(),
         "config": framework.model.config},
        os.path.join(out_dir, f"{args.type}_unlearned.pt"),
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

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="DTW threshold for motif discovery (subset mode)")
    parser.add_argument("--lambda-surr", type=float, default=1.0)
    parser.add_argument("--lambda-retain", type=float, default=1.0)
    parser.add_argument("--lambda-ewc", type=float, default=5.0)
    parser.add_argument("--lambda-forget", type=float, default=0.5)
    parser.add_argument("--forget-margin", type=float, default=2.0,
                        help="Bounded-ascent margin. Once L_forget exceeds "
                             "this, the forget term stops contributing.")

    args = parser.parse_args()

    if args.unlearn_node:
        if args.node_idx is None:
            parser.error("--node-idx required with --unlearn-node")
        if args.forget_set is not None:
            print("Warning: --forget-set ignored in node mode")
    else:
        if args.forget_set is None:
            parser.error("--forget-set required unless --unlearn-node is set")

    args.device = (torch.device("cuda") if args.enable_cuda and torch.cuda.is_available()
                   else torch.device("cpu"))

    run(args)


if __name__ == "__main__":
    main()
