import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import copy
from typing import Dict, Tuple, Optional


class UnlearningBaselines:
    """Collection of baseline unlearning methods for ST-GNNs"""
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def _compute_loss(self, output, target, loss_fn, faulty_node_idx=None, mode='all'):
        """
        Helper to compute loss with optional node masking.
        mode: 'all' (standard), 'retain' (exclude faulty), 'forget' (only faulty)
        """
        if faulty_node_idx is None or mode == 'all':
            return loss_fn(output, target)
        
        if mode == 'retain':
            # Calculate loss on all nodes EXCEPT faulty_node_idx
            # We construct a mask
            num_nodes = output.shape[1]
            if num_nodes <= 1: 
                # If input is already sliced or 1-node graph, assume it's valid retain data
                return loss_fn(output, target)
                
            # Create a boolean mask for nodes to keep
            node_mask = torch.ones(num_nodes, dtype=torch.bool, device=self.device)
            node_mask[faulty_node_idx] = False
            
            # Select nodes: (B, N, ...) -> (B, N_retain, ...)
            out_masked = output[:, node_mask, ...]
            tgt_masked = target[:, node_mask, ...]
            return loss_fn(out_masked, tgt_masked)
            
        elif mode == 'forget':
            # Calculate loss ONLY on faulty_node_idx
            out_sliced = output[:, faulty_node_idx:faulty_node_idx+1, ...]
            tgt_sliced = target[:, faulty_node_idx:faulty_node_idx+1, ...]
            return loss_fn(out_sliced, tgt_sliced)
            
        return loss_fn(output, target)
    
    # ========== BASELINE 1: Retrain from Scratch ==========
    def retrain_from_scratch(self, model_class, model_config: dict, 
                            retain_loader: DataLoader, A_wave: torch.Tensor,
                            num_epochs: int = 100, learning_rate: float = 1e-3,
                            batch_size: int = 512, faulty_node_idx: int = None) -> nn.Module:
        """
        Retrain from scratch. If faulty_node_idx is provided, loss is calculated 
        only on retain nodes (mode='retain').
        """
        print("="*80)
        print("BASELINE: Retrain from Scratch")
        print("="*80)
        
        # Initialize fresh model
        model = model_class(**model_config).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        
        A_wave = A_wave.to(self.device)
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            for X_batch, y_batch in retain_loader:
                X_batch = X_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)
                
                optimizer.zero_grad()
                output = model(A_wave, X_batch)
                
                # Compute loss on retain set (exclude faulty node if specified)
                loss = self._compute_loss(output, y_batch, loss_fn, faulty_node_idx, mode='retain')
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
        
        print("Retraining completed!")
        return model
    
    
    # ========== BASELINE 2: Fine-tune on Retain Set Only ==========
    def finetune_on_retain(self, model: nn.Module, retain_loader: DataLoader, 
                          A_wave: torch.Tensor, num_epochs: int = 50,
                          learning_rate: float = 1e-4, faulty_node_idx: int = None) -> Tuple[nn.Module, Dict]:
        """
        Fine-tune on retain set. If faulty_node_idx provided, exclude it from loss.
        """
        print("="*80)
        print("BASELINE: Fine-tune on Retain Set Only")
        print("="*80)
        
        model = copy.deepcopy(model).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        
        A_wave = A_wave.to(self.device)
        history = {'loss': []}
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            for X_batch, y_batch in retain_loader:
                X_batch = X_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)
                
                optimizer.zero_grad()
                output = model(A_wave, X_batch)
                
                # Compute loss on retain set (exclude faulty node)
                loss = self._compute_loss(output, y_batch, loss_fn, faulty_node_idx, mode='retain')
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                history['loss'].append(avg_loss)
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
        
        print("Fine-tuning completed!")
        return model, history
    
    
    # ========== BASELINE 3: Gradient Ascent (NegGrad) ==========
    def gradient_ascent(self, model: nn.Module, forget_loader: DataLoader,
                       A_wave: torch.Tensor, num_epochs: int = 20,
                       learning_rate: float = 1e-4, 
                       max_gradient_norm: float = 1.0, 
                       faulty_node_idx: int = None) -> Tuple[nn.Module, Dict]:
        """
        Gradient ascent. If faulty_node_idx provided, maximize loss ONLY on that node.
        """
        print("="*80)
        print("BASELINE: Gradient Ascent (NegGrad)")
        print("="*80)
        
        model = copy.deepcopy(model).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        
        A_wave = A_wave.to(self.device)
        history = {'forget_loss': []}
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            for X_batch, y_batch in forget_loader:
                X_batch = X_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)
                
                optimizer.zero_grad()
                output = model(A_wave, X_batch)
                
                # Maximize loss on Forget Set
                mode = 'forget' if faulty_node_idx is not None else 'all'
                loss = self._compute_loss(output, y_batch, loss_fn, faulty_node_idx, mode=mode)
                
                # NEGATIVE gradient (gradient ascent)
                (-loss).backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_gradient_norm)
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                history['forget_loss'].append(avg_loss)
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Forget Loss: {avg_loss:.6f}")
        
        print("Gradient ascent completed!")
        return model, history
    
    
    # ========== BASELINE 4: Gradient Ascent + Fine-tuning ==========
    def gradient_ascent_plus_finetune(self, model: nn.Module, 
                                     forget_loader: DataLoader,
                                     retain_loader: DataLoader,
                                     A_wave: torch.Tensor,
                                     neggrad_epochs: int = 20,
                                     finetune_epochs: int = 30,
                                     neggrad_lr: float = 1e-4,
                                     finetune_lr: float = 5e-5,
                                     faulty_node_idx: int = None) -> Tuple[nn.Module, Dict]:
        
        print("="*80)
        print("BASELINE: Gradient Ascent + Fine-tuning")
        print("="*80)
        
        # Stage 1: Gradient Ascent
        print("\n--- Stage 1: Gradient Ascent ---")
        model, history_neggrad = self.gradient_ascent(
            model, forget_loader, A_wave, neggrad_epochs, neggrad_lr, 
            faulty_node_idx=faulty_node_idx
        )
        
        # Stage 2: Fine-tuning
        print("\n--- Stage 2: Fine-tuning on Retain Set ---")
        model, history_finetune = self.finetune_on_retain(
            model, retain_loader, A_wave, finetune_epochs, finetune_lr,
            faulty_node_idx=faulty_node_idx
        )
        
        history = {
            'neggrad_loss': history_neggrad['forget_loss'],
            'finetune_loss': history_finetune['loss']
        }
        
        print("\nGradient Ascent + Fine-tuning completed!")
        return model, history
    
    
    # ========== BASELINE 5: Influence Functions ==========
    def influence_function_unlearning(self, model: nn.Module, 
                                     forget_loader: DataLoader,
                                     train_loader: DataLoader,
                                     A_wave: torch.Tensor,
                                     damping: float = 0.01,
                                     scaling: float = 1.0,
                                     faulty_node_idx: int = None) -> Tuple[nn.Module, Dict]:
        
        print("="*80)
        print("BASELINE: Influence Functions")
        print("="*80)
        
        model = copy.deepcopy(model).to(self.device)
        loss_fn = nn.MSELoss()
        A_wave = A_wave.to(self.device)
        
        # Step 1: Compute gradients on forget set
        print("Computing gradients on forget set...")
        model.eval()
        forget_grads = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
        forget_count = 0
        
        for X_batch, y_batch in forget_loader:
            X_batch = X_batch.float().to(self.device)
            y_batch = y_batch.float().to(self.device)
            
            model.zero_grad()
            output = model(A_wave, X_batch)
            
            # Gradients based on Forget Set loss
            mode = 'forget' if faulty_node_idx is not None else 'all'
            loss = self._compute_loss(output, y_batch, loss_fn, faulty_node_idx, mode=mode)
            
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    forget_grads[name] += param.grad.data * X_batch.shape[0]
            
            forget_count += X_batch.shape[0]
        
        # Average gradients
        for name in forget_grads:
            forget_grads[name] /= (forget_count + 1e-8)
        
        # Step 2: Compute inverse Hessian-vector product using conjugate gradient
        # Note: Hessian is usually computed on Training set (or Retain set)
        print("Computing Hessian approximation...")
        ihvp = self._compute_ihvp_cg(model, train_loader, A_wave, forget_grads, 
                                     loss_fn, damping, max_iterations=50)
        
        # Step 3: Update parameters
        print("Applying parameter updates...")
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in ihvp:
                    param.data += scaling * ihvp[name]
        
        info = {
            'forget_samples': forget_count,
            'damping': damping,
            'scaling': scaling
        }
        
        print("Influence function unlearning completed!")
        return model, info
    
    def _compute_ihvp_cg(self, model: nn.Module, train_loader: DataLoader,
                        A_wave: torch.Tensor, v: Dict[str, torch.Tensor],
                        loss_fn: nn.Module, damping: float, 
                        max_iterations: int = 50, tolerance: float = 1e-5) -> Dict[str, torch.Tensor]:
        
        x = {name: torch.zeros_like(tensor) for name, tensor in v.items()}
        r = {name: tensor.clone() for name, tensor in v.items()}
        p = {name: tensor.clone() for name, tensor in v.items()}
        
        rs_old = sum((r[name] ** 2).sum() for name in r)
        
        for i in range(max_iterations):
            Hp = self._hessian_vector_product(model, train_loader, A_wave, p, loss_fn, damping)
            pHp = sum((p[name] * Hp[name]).sum() for name in p)
            alpha = rs_old / (pHp + 1e-10)
            
            for name in x:
                x[name] += alpha * p[name]
                r[name] -= alpha * Hp[name]
            
            rs_new = sum((r[name] ** 2).sum() for name in r)
            if torch.sqrt(rs_new) < tolerance:
                break
            
            beta = rs_new / (rs_old + 1e-10)
            for name in p:
                p[name] = r[name] + beta * p[name]
            rs_old = rs_new
        
        return x
    
    def _hessian_vector_product(self, model: nn.Module, train_loader: DataLoader,
                               A_wave: torch.Tensor, v: Dict[str, torch.Tensor],
                               loss_fn: nn.Module, damping: float) -> Dict[str, torch.Tensor]:
        epsilon = 1e-3
        original_params = {name: param.clone() for name, param in model.named_parameters()}
        
        model.zero_grad()
        # Gradient at theta - use full training set (no node masking usually for Hessian)
        grad_theta = self._compute_gradient(model, train_loader, A_wave, loss_fn)
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in v:
                    param.data += epsilon * v[name]
        
        model.zero_grad()
        grad_theta_plus = self._compute_gradient(model, train_loader, A_wave, loss_fn)
        
        hvp = {}
        for name in v:
            hvp[name] = (grad_theta_plus[name] - grad_theta[name]) / epsilon
            hvp[name] += damping * v[name]
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.data = original_params[name]
        
        return hvp
    
    def _compute_gradient(self, model: nn.Module, data_loader: DataLoader,
                         A_wave: torch.Tensor, loss_fn: nn.Module,
                         max_samples: int = 500) -> Dict[str, torch.Tensor]:
        model.eval()
        gradients = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
        sample_count = 0
        
        for X_batch, y_batch in data_loader:
            if sample_count >= max_samples:
                break
            X_batch = X_batch.float().to(self.device)
            y_batch = y_batch.float().to(self.device)
            
            model.zero_grad()
            output = model(A_wave, X_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data * X_batch.shape[0]
            
            sample_count += X_batch.shape[0]
        
        if sample_count > 0:
            for name in gradients:
                gradients[name] /= sample_count
        
        return gradients
    
    
    # ========== BASELINE 6: Fisher Unlearning ==========
    def fisher_unlearning(self, model: nn.Module, forget_loader: DataLoader,
                         retain_loader: DataLoader, A_wave: torch.Tensor,
                         num_epochs: int = 50, learning_rate: float = 1e-4,
                         lambda_fisher: float = 10.0,
                         faulty_node_idx: int = None) -> Tuple[nn.Module, Dict]:
        
        print("="*80)
        print("BASELINE: Fisher Unlearning")
        print("="*80)
        
        model = copy.deepcopy(model).to(self.device)
        A_wave = A_wave.to(self.device)
        
        # Compute Fisher Info on Retain Set (exclude faulty node if specified)
        print("Computing Fisher Information Matrix on retain set...")
        fisher_diagonal = self._compute_fisher_diagonal(model, retain_loader, A_wave, faulty_node_idx)
        
        original_params = {name: param.clone() for name, param in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        
        history = {'total_loss': [], 'forget_loss': [], 'fisher_penalty': []}
        
        for epoch in range(num_epochs):
            model.train()
            epoch_total_loss = 0.0
            epoch_forget_loss = 0.0
            epoch_fisher_penalty = 0.0
            batch_count = 0
            
            for X_batch, y_batch in forget_loader:
                X_batch = X_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)
                
                optimizer.zero_grad()
                output = model(A_wave, X_batch)
                
                # Maximize loss on Forget Set (only faulty node)
                mode = 'forget' if faulty_node_idx is not None else 'all'
                forget_loss = self._compute_loss(output, y_batch, loss_fn, faulty_node_idx, mode=mode)
                
                fisher_penalty = torch.tensor(0.0, device=self.device)
                for name, param in model.named_parameters():
                    if name in fisher_diagonal:
                        diff = (param - original_params[name]) ** 2
                        fisher_penalty += (fisher_diagonal[name] * diff).sum()
                
                total_loss = -forget_loss + lambda_fisher * fisher_penalty
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_total_loss += total_loss.item()
                epoch_forget_loss += forget_loss.item()
                epoch_fisher_penalty += fisher_penalty.item()
                batch_count += 1
            
            if batch_count > 0:
                history['total_loss'].append(epoch_total_loss / batch_count)
                history['forget_loss'].append(epoch_forget_loss / batch_count)
                history['fisher_penalty'].append(epoch_fisher_penalty / batch_count)
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Total: {history['total_loss'][-1]:.4f}, Forget: {history['forget_loss'][-1]:.4f}")
        
        print("Fisher unlearning completed!")
        return model, history
    
    def _compute_fisher_diagonal(self, model: nn.Module, data_loader: DataLoader,
                                A_wave: torch.Tensor, max_samples: int = 1000,
                                faulty_node_idx: int = None) -> Dict[str, torch.Tensor]:
        model.eval()
        loss_fn = nn.MSELoss()
        
        fisher = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
        sample_count = 0
        
        for X_batch, y_batch in data_loader:
            if sample_count >= max_samples:
                break
            
            X_batch = X_batch.float().to(self.device)
            y_batch = y_batch.float().to(self.device)
            
            # Compute gradients sample-by-sample or batch (approximation)
            # Here we do batch approximation for speed if needed, but per-sample is correct for Fisher
            # We stick to per-sample loop as in original code
            
            for i in range(X_batch.shape[0]):
                if sample_count >= max_samples:
                    break
                
                model.zero_grad()
                output = model(A_wave, X_batch[i:i+1])
                
                # Compute loss on retain nodes only
                loss = self._compute_loss(output, y_batch[i:i+1], loss_fn, faulty_node_idx, mode='retain')
                
                loss.backward()
                
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        fisher[name] += param.grad.data ** 2
                
                sample_count += 1
        
        if sample_count > 0:
            for name in fisher:
                fisher[name] /= sample_count
                fisher[name] += 1e-8
        
        return fisher