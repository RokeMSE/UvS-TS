import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data.preprocess_pemsbay import generate_dataset

class PEMSBAYDataset(Dataset):
    """
    PEMS-BAY dataset wrapper
    """
    def __init__(self, input_data, target_data=None):
        """
        Args:
            input_data: Input data tensor of shape (num_samples, num_nodes, num_timesteps_input, num_features)
            target_data: Target data tensor of shape (num_samples, num_nodes, num_timesteps_output) [optional]
        """
        self.input_data = input_data
        self.target_data = target_data
        
    def __len__(self):
        return self.input_data.shape[0]
    
    def __getitem__(self, idx):
        if self.target_data is not None:
            return self.input_data[idx], self.target_data[idx]
        return self.input_data[idx]


def prepare_unlearning_data(X, faulty_node_idx, num_timesteps_input=12, 
                                    num_timesteps_output=3, train_split=0.8, 
                                    batch_size=32, forget_strategy="node_activity"):
    """
    Prepare local PEMS-BAY data for unlearning experiments
    
    Args:
        X: Raw PEMS-BAY data of shape (N, F, T) - nodes, features, time
        faulty_node_idx: Index of node to unlearn (can be int or list)
        num_timesteps_input: Input sequence length
        num_timesteps_output: Output/prediction sequence length
        train_split: Fraction for training (rest becomes test)
        batch_size: Batch size for DataLoaders
        forget_strategy: Strategy to identify forget samples
        
    Returns:
        dict containing train_loader, test_loader, forget_loader, retain_loader, 
        and the split indices for debugging
    """
    # Step 1: Convert raw data to training format 
    training_input, training_target = generate_dataset(
        X, num_timesteps_input=num_timesteps_input, 
        num_timesteps_output=num_timesteps_output
    )
    
    num_samples = training_input.shape[0]
    train_size = int(num_samples * train_split)
    
    # Step 2: Split into train/test
    indices = torch.randperm(num_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_input = training_input[train_indices]
    train_target = training_target[train_indices]
    test_input = training_input[test_indices]
    test_target = training_target[test_indices]
    
    # Step 3: Identify forget/retain samples from training set
    forget_indices_rel, retain_indices_rel = identify_samples_to_forget(
        train_input, train_target, faulty_node_idx, strategy=forget_strategy
    )
    
    # Get actual data for forget/retain sets
    forget_input = train_input[forget_indices_rel]
    forget_target = train_target[forget_indices_rel]
    retain_input = train_input[retain_indices_rel] 
    retain_target = train_target[retain_indices_rel]
    
    # Step 4: Create datasets
    train_dataset = PEMSBAYDataset(train_input, train_target)
    test_dataset = PEMSBAYDataset(test_input, test_target)
    forget_dataset = PEMSBAYDataset(forget_input, forget_target)
    retain_dataset = PEMSBAYDataset(retain_input, retain_target)
    
    # Step 5: Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=True)
    retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=False)
    
    # Return comprehensive data info
    data_info = {
        'train_loader': train_loader,
        'test_loader': test_loader, 
        'forget_loader': forget_loader,
        'retain_loader': retain_loader,
        'train_indices': train_indices,
        'test_indices': test_indices,
        'forget_indices_rel': forget_indices_rel,
        'retain_indices_rel': retain_indices_rel,
        'data_stats': {
            'total_samples': num_samples,
            'train_samples': len(train_indices),
            'test_samples': len(test_indices),
            'forget_samples': len(forget_indices_rel),
            'retain_samples': len(retain_indices_rel),
            'faulty_node_idx': faulty_node_idx
        }
    }
    
    return data_info


def identify_samples_to_forget(input_data, target_data, faulty_node_idx, 
                              strategy="node_activity", threshold=0.5):
    """
    Identify samples that should be forgotten based on different strategies
    
    Args:
        input_data: Input tensor (num_samples, num_nodes, num_timesteps, num_features) 
        target_data: Target tensor (num_samples, num_nodes, num_timesteps_output)
        faulty_node_idx: Index or list of indices of faulty node(s)
        strategy: Strategy to identify samples ("node_activity", "random", "temporal_pattern")
        threshold: Threshold for activity-based selection
        
    Returns:
        forget_indices, retain_indices (relative to the input data)
    """
    num_samples = input_data.shape[0]
    
    if isinstance(faulty_node_idx, int):
        faulty_node_idx = [faulty_node_idx]
    
    forget_indices = []
    retain_indices = []
    
    if strategy == "node_activity":
        # Forget samples where faulty node(s) have high activity
        for i in range(num_samples):
            should_forget = False
            for node_idx in faulty_node_idx:
                if node_idx < input_data.shape[1]:
                    # Calculate activity in both input and target
                    input_activity = torch.mean(torch.abs(input_data[i, node_idx, :, :]))
                    target_activity = torch.mean(torch.abs(target_data[i, node_idx, :]))
                    avg_activity = (input_activity + target_activity) / 2.0
                    
                    if avg_activity > threshold:
                        should_forget = True
                        break
            
            if should_forget:
                forget_indices.append(i)
            else:
                retain_indices.append(i)
                
    elif strategy == "random":
        # Random selection for forget set (4 baseline comparisons)
        forget_ratio = 0.1  # Forget 10% of samples randomly (change this if the current res output are not good)
        forget_size = int(num_samples * forget_ratio)
        all_indices = list(range(num_samples))
        np.random.shuffle(all_indices)
        forget_indices = all_indices[:forget_size]
        retain_indices = all_indices[forget_size:]
        
    elif strategy == "temporal_pattern":
        # Forget samples with specific temporal patterns in faulty nodes
        for i in range(num_samples):
            should_forget = False
            for node_idx in faulty_node_idx:
                if node_idx < input_data.shape[1]:
                    # Look for monotonic increasing pattern (example pattern)
                    node_series = input_data[i, node_idx, :, 0]  # First feature
                    if len(node_series) > 3:
                        # Check if generally increasing
                        diff = torch.diff(node_series)
                        increasing_ratio = (diff > 0).float().mean()
                        if increasing_ratio > 0.7:  # 70% increasing
                            should_forget = True
                            break
            
            if should_forget:
                forget_indices.append(i)
            else:
                retain_indices.append(i)
                
    elif strategy == "high_variance":
        # Forget samples where faulty nodes have high variance
        for i in range(num_samples):
            should_forget = False
            for node_idx in faulty_node_idx:
                if node_idx < input_data.shape[1]:
                    node_variance = torch.var(input_data[i, node_idx, :, 0])
                    if node_variance > threshold:
                        should_forget = True
                        break
            
            if should_forget:
                forget_indices.append(i)
            else:
                retain_indices.append(i)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return forget_indices, retain_indices


def create_simple_loaders(input_tensor, target_tensor, batch_size=32, shuffle=True):
    """
    Simple utility to create DataLoader from tensor data
    
    Args:
        input_tensor: Input data tensor
        target_tensor: Target data tensor  
        batch_size: Batch size
        shuffle: Whether to shuffle
        
    Returns:
        DataLoader
    """
    dataset = PEMSBAYDataset(input_tensor, target_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_data_subset(input_data, target_data, indices):
    """
    Extract subset of data based on indices
    
    Args:
        input_data: Input tensor
        target_data: Target tensor
        indices: List of indices to extract
        
    Returns:
        subset_input, subset_target
    """
    if len(indices) == 0:
        # Return empty tensors with correct shape
        empty_input = torch.empty(0, *input_data.shape[1:])
        empty_target = torch.empty(0, *target_data.shape[1:])
        return empty_input, empty_target
    
    return input_data[indices], target_data[indices]


def split_data_by_nodes(input_data, target_data, node_indices, strategy="exclude"):
    """
    Split data based on node involvement
    
    Args:
        input_data: Input tensor (num_samples, num_nodes, num_timesteps, num_features)
        target_data: Target tensor (num_samples, num_nodes, num_timesteps_output)  
        node_indices: List of node indices to focus on
        strategy: "exclude" to remove node data, "only" to keep only these nodes
        
    Returns:
        modified_input, modified_target
    """
    if strategy == "exclude":
        # Remove specified nodes
        remaining_nodes = [i for i in range(input_data.shape[1]) if i not in node_indices]
        return input_data[:, remaining_nodes], target_data[:, remaining_nodes]
    
    elif strategy == "only":
        # Keep only specified nodes
        return input_data[:, node_indices], target_data[:, node_indices]
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def analyze_data_distribution(input_data, target_data, faulty_node_idx):
    """
    Analyze the distribution of data for debugging unlearning
    
    Args:
        input_data: Input tensor
        target_data: Target tensor
        faulty_node_idx: Faulty node index
        
    Returns:
        dict with analysis results
    """
    num_samples, num_nodes, num_timesteps, num_features = input_data.shape
    
    analysis = {
        'data_shape': {
            'input': input_data.shape,
            'target': target_data.shape
        },
        'faulty_node_stats': {},
        'overall_stats': {}
    }
    
    # Analyze faulty node
    if faulty_node_idx < num_nodes:
        faulty_input = input_data[:, faulty_node_idx, :, :]
        faulty_target = target_data[:, faulty_node_idx, :]
        
        analysis['faulty_node_stats'] = {
            'input_mean': torch.mean(faulty_input).item(),
            'input_std': torch.std(faulty_input).item(), 
            'input_min': torch.min(faulty_input).item(),
            'input_max': torch.max(faulty_input).item(),
            'target_mean': torch.mean(faulty_target).item(),
            'target_std': torch.std(faulty_target).item(),
            'target_min': torch.min(faulty_target).item(),
            'target_max': torch.max(faulty_target).item(),
        }
    
    # Overall stats
    analysis['overall_stats'] = {
        'input_mean': torch.mean(input_data).item(),
        'input_std': torch.std(input_data).item(),
        'target_mean': torch.mean(target_data).item(), 
        'target_std': torch.std(target_data).item(),
    }
    
    return analysis