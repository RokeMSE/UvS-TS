import logging
import time
import os
import json
import torch
import psutil
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import sys

class UnlearningLogger:
    """
    Comprehensive logger for STGCN unlearning experiments.
    Tracks runtime, memory usage, GPU utilization, and training metrics.
    """
    
    def __init__(self, log_dir: str, experiment_name: str, 
                 log_to_console: bool = True, log_level: int = logging.INFO):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to save log files
            experiment_name: Name of the experiment
            log_to_console: Whether to print logs to console
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_to_console = log_to_console
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
        self.metrics_file = self.log_dir / f"{experiment_name}_{timestamp}_metrics.json"
        self.runtime_file = self.log_dir / f"{experiment_name}_{timestamp}_runtime.json"
        
        # Setup logger
        self.logger = logging.getLogger(f"{experiment_name}_{timestamp}")
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # Clear existing handlers
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_formatter = logging.Formatter(
                '%(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Runtime tracking
        self.start_time = None
        self.phase_times = {}
        self.phase_start = None
        self.current_phase = None
        
        # Metrics tracking
        self.metrics_history = {
            'training': [],
            'evaluation': [],
            'memory': [],
            'gpu': []
        }
        
        # System info
        self.system_info = self._get_system_info()
        
        self.logger.info("="*80)
        self.logger.info(f"Experiment: {experiment_name}")
        self.logger.info(f"Timestamp: {timestamp}")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info("="*80)
        self._log_system_info()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return info
    
    def _log_system_info(self):
        """Log system information"""
        self.logger.info("System Information:")
        for key, value in self.system_info.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("-"*80)
    
    def start_experiment(self):
        """Start timing the entire experiment"""
        self.start_time = time.time()
        self.logger.info("Experiment started")
    
    def end_experiment(self):
        """End timing the entire experiment"""
        if self.start_time is None:
            self.logger.warning("Experiment not started!")
            return
        
        total_time = time.time() - self.start_time
        self.logger.info("="*80)
        self.logger.info(f"Experiment completed in {self._format_time(total_time)}")
        
        # Log phase breakdown
        if self.phase_times:
            self.logger.info("\nPhase Breakdown:")
            for phase, duration in self.phase_times.items():
                percentage = (duration / total_time) * 100
                self.logger.info(f"  {phase}: {self._format_time(duration)} ({percentage:.1f}%)")
        
        # Save runtime summary
        runtime_summary = {
            'total_time_seconds': total_time,
            'total_time_formatted': self._format_time(total_time),
            'phase_times': {k: v for k, v in self.phase_times.items()},
            'timestamp': datetime.now().isoformat()
        }
        self._save_json(runtime_summary, self.runtime_file)
        
        self.logger.info("="*80)
    
    def start_phase(self, phase_name: str):
        """Start timing a phase (e.g., 'data_loading', 'training', 'evaluation')"""
        if self.current_phase is not None:
            self.logger.warning(f"Phase '{self.current_phase}' not ended before starting '{phase_name}'")
            self.end_phase()
        
        self.current_phase = phase_name
        self.phase_start = time.time()
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Starting phase: {phase_name}")
        self.logger.info(f"{'='*80}")
    
    def end_phase(self):
        """End timing the current phase"""
        if self.current_phase is None:
            self.logger.warning("No active phase to end")
            return
        
        duration = time.time() - self.phase_start
        self.phase_times[self.current_phase] = duration
        
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Phase '{self.current_phase}' completed in {self._format_time(duration)}")
        self.logger.info(f"{'='*80}\n")
        
        self.current_phase = None
        self.phase_start = None
    
    def log_epoch(self, epoch: int, total_epochs: int, metrics: Dict[str, float]):
        """Log training epoch metrics"""
        self.logger.info(f"Epoch [{epoch+1}/{total_epochs}]")
        
        for metric_name, value in metrics.items():
            self.logger.info(f"  {metric_name}: {value:.6f}")
        
        # Store in history
        epoch_data = {
            'epoch': epoch,
            'timestamp': time.time(),
            **metrics
        }
        self.metrics_history['training'].append(epoch_data)
        
        # Log memory usage
        if epoch % 10 == 0:  # Log every 10 epochs to avoid overhead
            self.log_memory_usage()
    
    def log_memory_usage(self):
        """Log current memory usage"""
        mem = psutil.virtual_memory()
        mem_used_gb = mem.used / (1024**3)
        mem_percent = mem.percent
        
        memory_info = {
            'timestamp': time.time(),
            'ram_used_gb': mem_used_gb,
            'ram_percent': mem_percent
        }
        
        if torch.cuda.is_available():
            gpu_mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            gpu_mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            memory_info['gpu_allocated_gb'] = gpu_mem_allocated
            memory_info['gpu_reserved_gb'] = gpu_mem_reserved
            
            self.logger.debug(
                f"Memory - RAM: {mem_used_gb:.2f}GB ({mem_percent:.1f}%), "
                f"GPU: {gpu_mem_allocated:.2f}GB allocated, {gpu_mem_reserved:.2f}GB reserved"
            )
        else:
            self.logger.debug(f"Memory - RAM: {mem_used_gb:.2f}GB ({mem_percent:.1f}%)")
        
        self.metrics_history['memory'].append(memory_info)
    
    def log_gpu_utilization(self):
        """Log GPU utilization (requires nvidia-ml-py3 package)"""
        if not torch.cuda.is_available():
            return
        
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            gpu_info = {
                'timestamp': time.time(),
                'gpu_utilization_percent': util.gpu,
                'memory_utilization_percent': util.memory
            }
            
            self.logger.debug(f"GPU Utilization: {util.gpu}%, Memory: {util.memory}%")
            self.metrics_history['gpu'].append(gpu_info)
            
            pynvml.nvmlShutdown()
        except ImportError:
            self.logger.debug("pynvml not installed. Install with: pip install nvidia-ml-py3")
        except Exception as e:
            self.logger.debug(f"Could not get GPU utilization: {e}")
    
    def log_evaluation_results(self, results: Dict[str, float], phase: str = "evaluation"):
        """Log evaluation results"""
        self.logger.info(f"\n{phase.upper()} RESULTS:")
        self.logger.info("-"*80)
        
        for metric_name, value in results.items():
            self.logger.info(f"  {metric_name}: {value:.6f}")
        
        self.logger.info("-"*80)
        
        # Store in history
        eval_data = {
            'phase': phase,
            'timestamp': time.time(),
            **results
        }
        self.metrics_history['evaluation'].append(eval_data)
    
    def log_model_info(self, model: torch.nn.Module, model_name: str = "Model"):
        """Log model architecture information"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"\n{model_name} Information:")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        self.logger.info(f"  Non-trainable parameters: {total_params - trainable_params:,}")
        
        # Log model size
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        size_mb = (param_size + buffer_size) / (1024**2)
        self.logger.info(f"  Model size: {size_mb:.2f} MB")
    
    def log_data_info(self, data_info: Dict[str, Any]):
        """Log dataset information"""
        self.logger.info("\nDataset Information:")
        for key, value in data_info.items():
            if isinstance(value, (int, float, str)):
                self.logger.info(f"  {key}: {value}")
            elif isinstance(value, dict):
                self.logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    self.logger.info(f"    {sub_key}: {sub_value}")
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """Log hyperparameters"""
        self.logger.info("\nHyperparameters:")
        for key, value in hyperparameters.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_warning(self, message: str):
        """Log a warning message"""
        self.logger.warning(message)
    
    def log_error(self, message: str):
        """Log an error message"""
        self.logger.error(message)
    
    def log_info(self, message: str):
        """Log an info message"""
        self.logger.info(message)
    
    def log_debug(self, message: str):
        """Log a debug message"""
        self.logger.debug(message)
    
    def save_metrics(self):
        """Save all metrics to JSON file"""
        metrics_data = {
            'experiment_name': self.experiment_name,
            'system_info': self.system_info,
            'metrics_history': self.metrics_history,
            'phase_times': self.phase_times,
            'timestamp': datetime.now().isoformat()
        }
        self._save_json(metrics_data, self.metrics_file)
        self.logger.info(f"Metrics saved to {self.metrics_file}")
    
    def _save_json(self, data: Dict, filepath: Path):
        """Save data to JSON file"""
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def create_summary_report(self) -> str:
        """Create a summary report of the experiment"""
        report_lines = [
            "="*80,
            f"EXPERIMENT SUMMARY: {self.experiment_name}",
            "="*80,
            "",
            "System Information:",
        ]
        
        for key, value in self.system_info.items():
            report_lines.append(f"  {key}: {value}")
        
        report_lines.extend([
            "",
            "Runtime Breakdown:",
        ])
        
        if self.phase_times:
            total_time = sum(self.phase_times.values())
            for phase, duration in sorted(self.phase_times.items(), key=lambda x: x[1], reverse=True):
                percentage = (duration / total_time) * 100
                report_lines.append(f"  {phase}: {self._format_time(duration)} ({percentage:.1f}%)")
        
        report_lines.extend([
            "",
            "Final Evaluation Metrics:",
        ])
        
        if self.metrics_history['evaluation']:
            last_eval = self.metrics_history['evaluation'][-1]
            for key, value in last_eval.items():
                if key not in ['phase', 'timestamp']:
                    report_lines.append(f"  {key}: {value:.6f}")
        
        report_lines.append("="*80)
        
        report = "\n".join(report_lines)
        
        # Save report
        report_file = self.log_dir / f"{self.experiment_name}_summary.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        self.logger.info(f"\nSummary report saved to {report_file}")
        
        return report
    
    def __enter__(self):
        """Context manager entry"""
        self.start_experiment()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if exc_type is not None:
            self.log_error(f"Exception occurred: {exc_type.__name__}: {exc_val}")
        
        self.end_experiment()
        self.save_metrics()
        self.create_summary_report()


# ============ Example Usage ============

def example_usage():
    """Example of how to use the UnlearningLogger"""
    
    # Initialize logger
    logger = UnlearningLogger(
        log_dir="logs",
        experiment_name="unlearn_node_10",
        log_to_console=True,
        log_level=logging.INFO
    )
    
    # Start experiment
    logger.start_experiment()
    
    # Log hyperparameters
    logger.log_hyperparameters({
        'learning_rate': 1e-3,
        'batch_size': 64,
        'epochs': 100,
        'lambda_ewc': 10.0,
        'lambda_surrogate': 1.0,
        'faulty_node_idx': 10
    })
    
    # Phase 1: Data Loading
    logger.start_phase("data_loading")
    time.sleep(2)  # Simulate data loading
    logger.log_data_info({
        'total_samples': 10000,
        'train_samples': 8000,
        'test_samples': 2000,
        'num_nodes': 325,
        'num_features': 3
    })
    logger.end_phase()
    
    # Phase 2: Training
    logger.start_phase("training")
    for epoch in range(5):
        # Simulate training
        time.sleep(0.5)
        metrics = {
            'total_loss': np.random.uniform(0.5, 1.0),
            'surrogate_loss': np.random.uniform(0.2, 0.5),
            'ewc_penalty': np.random.uniform(0.1, 0.3),
            'retain_loss': np.random.uniform(0.2, 0.4)
        }
        logger.log_epoch(epoch, 5, metrics)
        
        if epoch % 2 == 0:
            logger.log_memory_usage()
    
    logger.end_phase()
    
    # Phase 3: Evaluation
    logger.start_phase("evaluation")
    time.sleep(1)
    evaluation_results = {
        'fidelity_score': 0.95,
        'forgetting_efficacy': 2.34,
        'generalization_score': 0.98,
        'spatial_correlation_divergence': 0.67,
        'forget_set_mse': 1.23,
        'retain_set_mse': 0.45,
        'test_set_mse': 0.48
    }
    logger.log_evaluation_results(evaluation_results)
    logger.end_phase()
    
    # End experiment
    logger.end_experiment()
    logger.save_metrics()
    logger.create_summary_report()


# ============ Context Manager Usage ============

def example_with_context_manager():
    """Example using context manager"""
    
    with UnlearningLogger("logs", "unlearn_subset_5") as logger:
        logger.log_info("Starting unlearning experiment")
        
        logger.start_phase("training")
        # Your training code here
        time.sleep(2)
        logger.end_phase()
        
        logger.log_info("Experiment completed successfully")


if __name__ == "__main__":
    example_usage()
    # example_with_context_manager()