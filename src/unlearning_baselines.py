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
    
    # ========== BASELINE 1: Retrain from Scratch ==========
    def retrain_from_scratch(self, model_class, model_config: dict, 
                            retain_loader: DataLoader, A_wave: torch.Tensor,
                            num_epochs: int = 100, learning_rate: float = 1e-3,
                            batch_size: int = 512) -> nn.Module:
        """
        Retrain the model from scratch on only the retain set.
        This is the gold standard baseline.
        
        Args:
            model_class: Model class (STGCN, STGAT, etc.)
            model_config: Configuration dict for model initialization
            retain_loader: DataLoader containing only retain data
            A_wave: Adjacency matrix
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Retrained model
        """
        print("="*80)
        print("BASELINE: Retrain from Scratch")
        print("="*80)
        
        # Initialize fresh model
        model = model_class(**model_config).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        
        A_wave = A_wave.to(self.device)
        training_losses = []
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            for X_batch, y_batch in retain_loader:
                X_batch = X_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)
                
                optimizer.zero_grad()
                output = model(A_wave, X_batch)
                loss = loss_fn(output, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count
            training_losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
        
        print("Retraining completed!")
        return model
    
    
    # ========== BASELINE 2: Fine-tune on Retain Set Only ==========
    def finetune_on_retain(self, model: nn.Module, retain_loader: DataLoader, 
                          A_wave: torch.Tensor, num_epochs: int = 50,
                          learning_rate: float = 1e-4) -> Tuple[nn.Module, Dict]:
        """
        Fine-tune the original model on retain set only.
        Uses lower learning rate to avoid catastrophic forgetting of retain data.
        
        Args:
            model: Pre-trained model
            retain_loader: DataLoader with retain data
            A_wave: Adjacency matrix
            num_epochs: Number of fine-tuning epochs
            learning_rate: Learning rate (typically lower than initial training)
            
        Returns:
            Fine-tuned model and training history
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
                loss = loss_fn(output, y_batch)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
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
                       max_gradient_norm: float = 1.0) -> Tuple[nn.Module, Dict]:
        """
        Perform gradient ascent on forget set to maximize loss.
        This encourages the model to "forget" the forget set.
        
        Args:
            model: Pre-trained model
            forget_loader: DataLoader with forget data
            A_wave: Adjacency matrix
            num_epochs: Number of gradient ascent epochs
            learning_rate: Learning rate for ascent
            max_gradient_norm: Maximum gradient norm for stability
            
        Returns:
            Updated model and training history
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
                loss = loss_fn(output, y_batch)
                
                # NEGATIVE gradient (gradient ascent)
                (-loss).backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_gradient_norm)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
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
                                     finetune_lr: float = 5e-5) -> Tuple[nn.Module, Dict]:
        """
        Two-stage process:
        1. Gradient ascent on forget set
        2. Fine-tune on retain set to recover performance
        
        Args:
            model: Pre-trained model
            forget_loader: DataLoader with forget data
            retain_loader: DataLoader with retain data
            A_wave: Adjacency matrix
            neggrad_epochs: Epochs for gradient ascent
            finetune_epochs: Epochs for fine-tuning
            neggrad_lr: Learning rate for gradient ascent
            finetune_lr: Learning rate for fine-tuning
            
        Returns:
            Updated model and training history
        """
        print("="*80)
        print("BASELINE: Gradient Ascent + Fine-tuning")
        print("="*80)
        
        # Stage 1: Gradient Ascent
        print("\n--- Stage 1: Gradient Ascent ---")
        model, history_neggrad = self.gradient_ascent(
            model, forget_loader, A_wave, neggrad_epochs, neggrad_lr
        )
        
        # Stage 2: Fine-tuning
        print("\n--- Stage 2: Fine-tuning on Retain Set ---")
        model, history_finetune = self.finetune_on_retain(
            model, retain_loader, A_wave, finetune_epochs, finetune_lr
        )
        
        # Combine histories
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
                                     scaling: float = 1.0) -> Tuple[nn.Module, Dict]:
        """
        Influence function based unlearning.
        Approximates the effect of removing forget samples using influence functions.
        
        Based on: "Understanding Black-box Predictions via Influence Functions" (Koh & Liang, 2017)
        
        Args:
            model: Pre-trained model
            forget_loader: DataLoader with forget data
            train_loader: DataLoader with training data (for Hessian approximation)
            A_wave: Adjacency matrix
            damping: Damping factor for Hessian inverse approximation
            scaling: Scaling factor for parameter updates
            
        Returns:
            Updated model and computation info
        """
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
            loss = loss_fn(output, y_batch)
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    forget_grads[name] += param.grad.data * X_batch.shape[0]
            
            forget_count += X_batch.shape[0]
        
        # Average gradients
        for name in forget_grads:
            forget_grads[name] /= forget_count
        
        # Step 2: Compute inverse Hessian-vector product using conjugate gradient
        print("Computing Hessian approximation...")
        ihvp = self._compute_ihvp_cg(model, train_loader, A_wave, forget_grads, 
                                     loss_fn, damping, max_iterations=50)
        
        # Step 3: Update parameters
        print("Applying parameter updates...")
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in ihvp:
                    # Remove influence of forget samples
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
        """
        Compute inverse Hessian-vector product using Conjugate Gradient method.
        Approximates H^{-1} * v where H is the Hessian.
        """
        # Initialize result
        x = {name: torch.zeros_like(tensor) for name, tensor in v.items()}
        r = {name: tensor.clone() for name, tensor in v.items()}
        p = {name: tensor.clone() for name, tensor in v.items()}
        
        # Initial residual norm
        rs_old = sum((r[name] ** 2).sum() for name in r)
        
        for i in range(max_iterations):
            # Compute Hessian-vector product Hp
            Hp = self._hessian_vector_product(model, train_loader, A_wave, p, loss_fn, damping)
            
            # Compute step size
            pHp = sum((p[name] * Hp[name]).sum() for name in p)
            alpha = rs_old / (pHp + 1e-10)
            
            # Update solution and residual
            for name in x:
                x[name] += alpha * p[name]
                r[name] -= alpha * Hp[name]
            
            # Check convergence
            rs_new = sum((r[name] ** 2).sum() for name in r)
            if torch.sqrt(rs_new) < tolerance:
                print(f"CG converged at iteration {i+1}")
                break
            
            # Update search direction
            beta = rs_new / (rs_old + 1e-10)
            for name in p:
                p[name] = r[name] + beta * p[name]
            
            rs_old = rs_new
        
        return x
    
    def _hessian_vector_product(self, model: nn.Module, train_loader: DataLoader,
                               A_wave: torch.Tensor, v: Dict[str, torch.Tensor],
                               loss_fn: nn.Module, damping: float) -> Dict[str, torch.Tensor]:
        """
        Compute Hessian-vector product using finite differences.
        Approximates: H * v ≈ (∇L(θ + εv) - ∇L(θ)) / ε
        """
        epsilon = 1e-3
        
        # Save original parameters
        original_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Compute gradient at θ
        model.zero_grad()
        grad_theta = self._compute_gradient(model, train_loader, A_wave, loss_fn)
        
        # Perturb parameters: θ + εv
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in v:
                    param.data += epsilon * v[name]
        
        # Compute gradient at θ + εv
        model.zero_grad()
        grad_theta_plus = self._compute_gradient(model, train_loader, A_wave, loss_fn)
        
        # Compute Hessian-vector product with damping
        hvp = {}
        for name in v:
            hvp[name] = (grad_theta_plus[name] - grad_theta[name]) / epsilon
            hvp[name] += damping * v[name]  # Add damping term
        
        # Restore original parameters
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.data = original_params[name]
        
        return hvp
    
    def _compute_gradient(self, model: nn.Module, data_loader: DataLoader,
                         A_wave: torch.Tensor, loss_fn: nn.Module,
                         max_samples: int = 500) -> Dict[str, torch.Tensor]:
        """Compute gradient on a subset of data for efficiency"""
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
        
        # Average gradients
        if sample_count > 0:
            for name in gradients:
                gradients[name] /= sample_count
        
        return gradients
    
    
    # ========== BASELINE 6: Fisher Unlearning ==========
    def fisher_unlearning(self, model: nn.Module, forget_loader: DataLoader,
                         retain_loader: DataLoader, A_wave: torch.Tensor,
                         num_epochs: int = 50, learning_rate: float = 1e-4,
                         lambda_fisher: float = 10.0) -> Tuple[nn.Module, Dict]:
        """
        Fisher Information based unlearning.
        Similar to EWC but focuses on forgetting instead of retention.
        
        Uses Fisher Information Matrix to guide which parameters to modify
        while maintaining retain set performance.
        
        Args:
            model: Pre-trained model
            forget_loader: DataLoader with forget data
            retain_loader: DataLoader with retain data
            A_wave: Adjacency matrix
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            lambda_fisher: Weight for Fisher penalty
            
        Returns:
            Updated model and training history
        """
        print("="*80)
        print("BASELINE: Fisher Unlearning")
        print("="*80)
        
        model = copy.deepcopy(model).to(self.device)
        A_wave = A_wave.to(self.device)
        
        # Compute Fisher Information Matrix on retain set
        print("Computing Fisher Information Matrix on retain set...")
        fisher_diagonal = self._compute_fisher_diagonal(model, retain_loader, A_wave)
        
        # Store original parameters
        original_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        
        history = {'total_loss': [], 'forget_loss': [], 'fisher_penalty': []}
        
        for epoch in range(num_epochs):
            model.train()
            epoch_total_loss = 0.0
            epoch_forget_loss = 0.0
            epoch_fisher_penalty = 0.0
            batch_count = 0
            
            # Iterate over forget set
            for X_batch, y_batch in forget_loader:
                X_batch = X_batch.float().to(self.device)
                y_batch = y_batch.float().to(self.device)
                
                optimizer.zero_grad()
                
                # Forget loss (maximize this - use negative)
                output = model(A_wave, X_batch)
                forget_loss = loss_fn(output, y_batch)
                
                # Fisher penalty (penalize changes to important parameters)
                fisher_penalty = torch.tensor(0.0, device=self.device)
                for name, param in model.named_parameters():
                    if name in fisher_diagonal:
                        diff = (param - original_params[name]) ** 2
                        fisher_penalty += (fisher_diagonal[name] * diff).sum()
                
                # Total loss: maximize forget loss while staying close to original on important params
                total_loss = -forget_loss + lambda_fisher * fisher_penalty
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_total_loss += total_loss.item()
                epoch_forget_loss += forget_loss.item()
                epoch_fisher_penalty += fisher_penalty.item()
                batch_count += 1
            
            # Record history
            history['total_loss'].append(epoch_total_loss / batch_count)
            history['forget_loss'].append(epoch_forget_loss / batch_count)
            history['fisher_penalty'].append(epoch_fisher_penalty / batch_count)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}]")
                print(f"  Total Loss: {history['total_loss'][-1]:.6f}")
                print(f"  Forget Loss: {history['forget_loss'][-1]:.6f}")
                print(f"  Fisher Penalty: {history['fisher_penalty'][-1]:.6f}")
        
        print("Fisher unlearning completed!")
        return model, history
    
    def _compute_fisher_diagonal(self, model: nn.Module, data_loader: DataLoader,
                                A_wave: torch.Tensor, max_samples: int = 1000) -> Dict[str, torch.Tensor]:
        """
        Compute diagonal Fisher Information Matrix.
        FIM_ii = E[(∂log p(y|x,θ)/∂θ_i)^2]
        """
        model.eval()
        loss_fn = nn.MSELoss()
        
        fisher = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
        sample_count = 0
        
        for X_batch, y_batch in data_loader:
            if sample_count >= max_samples:
                break
            
            X_batch = X_batch.float().to(self.device)
            y_batch = y_batch.float().to(self.device)
            
            # Per-sample gradient computation
            for i in range(X_batch.shape[0]):
                if sample_count >= max_samples:
                    break
                
                model.zero_grad()
                output = model(A_wave, X_batch[i:i+1])
                loss = loss_fn(output, y_batch[i:i+1])
                loss.backward()
                
                # Accumulate squared gradients
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        fisher[name] += param.grad.data ** 2
                
                sample_count += 1
        
        # Average and add small constant for numerical stability
        if sample_count > 0:
            for name in fisher:
                fisher[name] /= sample_count
                fisher[name] += 1e-8
        
        return fisher



def run_all_baselines(model, model_class, model_config,
                     forget_loader, retain_loader, train_loader, test_loader,
                     A_wave, device="cuda") -> Dict[str, nn.Module]:
    """
    Run all baseline methods and return the results.
    
    Returns:
        Dictionary mapping baseline name to unlearned model
    """
    baselines = UnlearningBaselines(device=device)
    results = {}
    
    # 1. Retrain from Scratch
    print("\n" + "="*80)
    print("Running: Retrain from Scratch")
    print("="*80)
    results['retrain'] = baselines.retrain_from_scratch(
        model_class, model_config, retain_loader, A_wave
    )
    
    # 2. Fine-tune
    print("\n" + "="*80)
    print("Running: Fine-tune on Retain Set")
    print("="*80)
    results['finetune'], _ = baselines.finetune_on_retain(
        model, retain_loader, A_wave
    )
    
    # 3. Gradient Ascent
    print("\n" + "="*80)
    print("Running: Gradient Ascent (NegGrad)")
    print("="*80)
    results['neggrad'], _ = baselines.gradient_ascent(
        model, forget_loader, A_wave
    )
    
    # 4. Gradient Ascent + Fine-tune
    print("\n" + "="*80)
    print("Running: Gradient Ascent + Fine-tune")
    print("="*80)
    results['neggrad_ft'], _ = baselines.gradient_ascent_plus_finetune(
        model, forget_loader, retain_loader, A_wave
    )
    
    # 5. Influence Functions
    print("\n" + "="*80)
    print("Running: Influence Functions")
    print("="*80)
    results['influence'], _ = baselines.influence_function_unlearning(
        model, forget_loader, train_loader, A_wave
    )
    
    # 6. Fisher Unlearning
    print("\n" + "="*80)
    print("Running: Fisher Unlearning")
    print("="*80)
    results['fisher'], _ = baselines.fisher_unlearning(
        model, forget_loader, retain_loader, A_wave
    )
    
    print("\n" + "="*80)
    print("All baselines completed!")
    print("="*80)
    
    return results