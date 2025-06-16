import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    get_linear_schedule_with_warmup,
    AutoConfig
)
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import yaml
import logging
import numpy as np
import json
import os
from pathlib import Path
import time
from collections import defaultdict
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# 1. CONFIGURATION SYSTEM (Declarative)
# =============================================================================

@dataclass
class SafetyConfig:
    """Centralized configuration for all components"""
    
    # Model settings
    model_name: str = 'distilbert-base-uncased'
    representation_layers: List[int] = field(default_factory=lambda: list(range(4, 8)))
    max_length: int = 512
    
    # Loss settings
    loss_type: str = 'triplet'
    alpha: float = 0.5  # Benign weight
    beta: float = 0.4   # Harmful weight
    gamma: float = 0.9  # KL weight
    margin_benign: float = 1.0
    margin_harmful: float = 2.0
    temperature: float = 0.1  # For contrastive learning
    
    # Training settings
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    batch_size: int = 8
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0
    
    # Adversarial settings
    adversarial_ratio: float = 0.3
    adversarial_layers: List[int] = field(default_factory=lambda: list(range(0, 4)))
    adversarial_strength: float = 0.1
    
    # Evaluation settings
    eval_steps: int = 50
    save_steps: int = 100
    log_steps: int = 10
    
    # Data settings
    data_balance_ratio: float = 1.0  # Ratio of harmful to benign examples
    
    # Output settings
    output_dir: str = './safety_training_output'
    save_model: bool = True
    save_metrics: bool = True
    
    def save_to_yaml(self, path: str):
        """Save configuration to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'SafetyConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def __post_init__(self):
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. DATA HANDLING
# =============================================================================

class SafetyDataset(Dataset):
    """Dataset for safety training with benign and harmful examples"""
    
    def __init__(
        self, 
        benign_data: List[str], 
        harmful_data: List[str], 
        tokenizer: AutoTokenizer,
        config: SafetyConfig
    ):
        self.tokenizer = tokenizer
        self.config = config
        
        # Balance the dataset
        min_len = min(len(benign_data), len(harmful_data))
        max_len = max(len(benign_data), len(harmful_data))
        
        if config.data_balance_ratio == 1.0:
            # Equal amounts
            self.benign_data = benign_data[:min_len]
            self.harmful_data = harmful_data[:min_len]
        else:
            # Use ratio
            harmful_count = int(min_len * config.data_balance_ratio)
            self.benign_data = benign_data[:min_len]
            self.harmful_data = harmful_data[:harmful_count]
        
        logger.info(f"Dataset created: {len(self.benign_data)} benign, {len(self.harmful_data)} harmful examples")
    
    def __len__(self):
        return min(len(self.benign_data), len(self.harmful_data))
    
    def __getitem__(self, idx):
        benign_text = self.benign_data[idx % len(self.benign_data)]
        harmful_text = self.harmful_data[idx % len(self.harmful_data)]
        
        # Tokenize both texts
        benign_tokens = self.tokenizer(
            benign_text,
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        harmful_tokens = self.tokenizer(
            harmful_text,
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'benign_input_ids': benign_tokens['input_ids'].squeeze(),
            'benign_attention_mask': benign_tokens['attention_mask'].squeeze(),
            'harmful_input_ids': harmful_tokens['input_ids'].squeeze(),
            'harmful_attention_mask': harmful_tokens['attention_mask'].squeeze(),
        }

def create_sample_data() -> Tuple[List[str], List[str]]:
    """Create sample benign and harmful data for demonstration"""
    
    benign_data = [
        "How can I help you today?",
        "What's the weather like?",
        "Can you explain machine learning?",
        "What are some good books to read?",
        "How do I cook pasta?",
        "Tell me about history.",
        "What is artificial intelligence?",
        "How can I improve my health?",
        "What are some travel destinations?",
        "Can you help me with my homework?",
        "What's the best way to learn programming?",
        "How do I start a garden?",
        "What are some good movies to watch?",
        "How can I be more productive?",
        "What's the capital of France?",
        "How do solar panels work?",
        "What are some healthy recipes?",
        "How do I fix a leaky faucet?",
        "What's the difference between AI and ML?",
        "How can I save money?",
        "What are some meditation techniques?",
        "How do I write a resume?",
        "What's the best exercise routine?",
        "How do I learn a new language?",
        "What are some science experiments for kids?"
    ] * 4  # Multiply for more data
    
    harmful_data = [
        "How to create dangerous substances for harmful purposes",
        "Ways to harm other people physically",
        "Instructions for illegal drug manufacturing",
        "How to break into someone's home",
        "Methods to commit fraud or theft",
        "How to create weapons for violence",
        "Ways to cyberbully or harass others",
        "Instructions for hacking into systems",
        "How to spread misinformation effectively",
        "Methods to evade law enforcement",
        "How to manipulate people psychologically",
        "Ways to commit identity theft",
        "Instructions for creating malware",
        "How to engage in insider trading",
        "Methods to contaminate food or water",
        "How to create explosive devices",
        "Ways to stalk or track someone",
        "Instructions for money laundering",
        "How to commit tax evasion",
        "Methods to harm animals",
        "How to create fake documents",
        "Ways to spread hate speech",
        "Instructions for social engineering attacks",
        "How to commit copyright infringement",
        "Methods to bypass security systems"
    ] * 4  # Multiply for more data
    
    return benign_data, harmful_data

class RepresentationExtractor(ABC):
    """Abstract interface for extracting representations"""
    
    @abstractmethod
    def extract(self, model: nn.Module, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def register_hooks(self, model: nn.Module):
        pass

class LossComputer(ABC):
    """Abstract interface for computing losses"""
    
    @abstractmethod
    def compute(self, benign_orig: torch.Tensor, benign_new: torch.Tensor,
                harmful_orig: torch.Tensor, harmful_new: torch.Tensor,
                **kwargs) -> Dict[str, torch.Tensor]:
        pass

class DefenseStrategy(ABC):
    """Abstract interface for defense mechanisms"""
    
    @abstractmethod
    def apply_defense(self, representations: torch.Tensor, 
                     layer_idx: int = None) -> torch.Tensor:
        pass

class Evaluator(ABC):
    """Abstract interface for evaluation"""
    
    @abstractmethod
    def evaluate(self, model: nn.Module, test_data: Any) -> Dict[str, float]:
        pass

class MetricsTracker(ABC):
    """Abstract interface for metrics tracking"""
    
    @abstractmethod
    def log_metrics(self, epoch: int, step: int, metrics: Dict[str, float]):
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, List[float]]:
        pass

# =============================================================================
# 4. CONCRETE IMPLEMENTATIONS (Modular)
# =============================================================================

class LayerAverageExtractor(RepresentationExtractor):
    """Extract and average representations from specified layers"""
    
    def __init__(self, layers: List[int]):
        self.layers = layers
        self.hooks = {}
        self.activations = {}
    
    def _hook_fn(self, layer_name: str):
        def hook(module, input, output):
            # Handle tuple outputs (like from transformer layers)
            activation = output[0] if isinstance(output, tuple) else output
            self.activations[layer_name] = activation.detach()
        return hook
    
    def register_hooks(self, model: nn.Module):
        """Register forward hooks for target layers"""
        self.hooks.clear()
        
        # Handle different model architectures
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            # BERT-style models
            layers = model.encoder.layer
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # GPT-style models
            layers = model.transformer.h
        elif hasattr(model, 'layers'):
            # Generic transformer
            layers = model.layers
        else:
            logger.warning("Could not find transformer layers, using fallback")
            return
        
        for layer_idx in self.layers:
            if layer_idx < len(layers):
                layer = layers[layer_idx]
                layer_name = f'layer_{layer_idx}'
                hook = layer.register_forward_hook(self._hook_fn(layer_name))
                self.hooks[layer_name] = hook
    
    def extract(self, model: nn.Module, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract averaged representations"""
        self.activations.clear()
        
        # Forward pass to trigger hooks
        with torch.no_grad() if not model.training else torch.enable_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Average across layers and sequence length
        layer_reprs = []
        for layer_idx in self.layers:
            layer_name = f'layer_{layer_idx}'
            if layer_name in self.activations:
                # Masked average pooling
                repr_tensor = self.activations[layer_name]
                mask_expanded = attention_mask.unsqueeze(-1).expand(repr_tensor.size()).float()
                masked_repr = repr_tensor * mask_expanded
                pooled_repr = masked_repr.sum(1) / mask_expanded.sum(1).clamp(min=1e-8)
                layer_reprs.append(pooled_repr)
        
        if layer_reprs:
            return torch.stack(layer_reprs, dim=1).mean(dim=1)
        else:
            # Fallback to last hidden state
            last_hidden = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            masked_hidden = last_hidden * mask_expanded
            return masked_hidden.sum(1) / mask_expanded.sum(1).clamp(min=1e-8)
    
    def cleanup(self):
        """Remove all hooks"""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()

class TripletLossComputer(LossComputer):
    """Compute triplet loss for contrastive learning"""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.cosine_sim = nn.CosineSimilarity(dim=-1)
    
    def _mixed_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Combined L2 and cosine distance"""
        l2_dist = torch.norm(x - y, p=2, dim=-1)
        cos_sim = self.cosine_sim(x, y)
        cos_dist = 1 - cos_sim
        return 0.5 * l2_dist + 0.5 * cos_dist
    
    def _contrastive_loss(self, anchor: torch.Tensor, positive: torch.Tensor, 
                         negative: torch.Tensor) -> torch.Tensor:
        """InfoNCE-style contrastive loss"""
        pos_sim = F.cosine_similarity(anchor, positive, dim=-1) / self.config.temperature
        neg_sim = F.cosine_similarity(anchor, negative, dim=-1) / self.config.temperature
        
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        return F.cross_entropy(logits, labels)
    
    def compute(self, benign_orig: torch.Tensor, benign_new: torch.Tensor,
                harmful_orig: torch.Tensor, harmful_new: torch.Tensor,
                benign_logits_orig: torch.Tensor = None,
                benign_logits_new: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Compute all loss components"""
        
        # Compute harmful mean for triplet anchor
        harmful_mean = harmful_new.mean(dim=0, keepdim=True).expand(harmful_new.shape[0], -1)
        
        # Benign triplet loss: keep benign close to original, far from harmful
        benign_pos_dist = self._mixed_distance(benign_orig, benign_new)
        benign_neg_dist = self._mixed_distance(benign_new, harmful_mean)
        benign_triplet = F.relu(
            benign_pos_dist - benign_neg_dist + self.config.margin_benign
        ).mean()
        
        # Harmful triplet loss: push harmful away from original, toward harmful mean
        harmful_pos_dist = self._mixed_distance(harmful_new, harmful_mean)
        harmful_neg_dist = self._mixed_distance(harmful_new, harmful_orig)
        harmful_triplet = F.relu(
            harmful_pos_dist - harmful_neg_dist + self.config.margin_harmful
        ).mean()
        
        # Contrastive loss component
        contrastive_loss = self._contrastive_loss(benign_new, benign_orig, harmful_mean)
        
        # KL divergence loss for benign behavior preservation
        kl_loss = torch.tensor(0.0, device=benign_orig.device)
        if benign_logits_orig is not None and benign_logits_new is not None:
            log_probs_orig = F.log_softmax(benign_logits_orig, dim=-1)
            probs_new = F.softmax(benign_logits_new, dim=-1)
            kl_loss = self.kl_div(log_probs_orig, probs_new)
        
        # Combined loss
        total_loss = (
            self.config.alpha * benign_triplet +
            self.config.beta * harmful_triplet +
            self.config.gamma * kl_loss +
            0.1 * contrastive_loss  # Small weight for contrastive component
        )
        
        return {
            'total_loss': total_loss,
            'benign_triplet': benign_triplet,
            'harmful_triplet': harmful_triplet,
            'contrastive_loss': contrastive_loss,
            'kl_loss': kl_loss
        }

class AdversarialDefense(DefenseStrategy):
    """Adversarial perturbation defense strategy"""
    
    def __init__(self, hidden_size: int, layers: List[int], config: SafetyConfig):
        self.layers = layers
        self.config = config
        self.attack_modules = nn.ModuleDict({
            str(layer_idx): nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Dropout(0.1)
            ) for layer_idx in layers
        })
        
        # Initialize weights
        for module in self.attack_modules.values():
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def apply_defense(self, representations: torch.Tensor, 
                     layer_idx: int = None) -> torch.Tensor:
        """Apply adversarial transformation"""
        if layer_idx is not None and str(layer_idx) in self.attack_modules:
            perturbation = self.attack_modules[str(layer_idx)](representations)
            return representations + self.config.adversarial_strength * perturbation
        return representations

class ComprehensiveEvaluator(Evaluator):
    """Comprehensive evaluation of safety training effectiveness"""
    
    def __init__(self, extractor: RepresentationExtractor, tokenizer: AutoTokenizer, config: SafetyConfig):
        self.extractor = extractor
        self.tokenizer = tokenizer
        self.config = config
    
    def evaluate(self, model: nn.Module, test_data: List[str]) -> Dict[str, float]:
        """Evaluate model on test data"""
        model.eval()
        distances = []
        similarities = []
        
        with torch.no_grad():
            for i, text in enumerate(test_data):
                tokens = self.tokenizer(
                    text,
                    max_length=self.config.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                ).to(model.device)
                
                # Get representation
                repr_tensor = self.extractor.extract(
                    model, tokens['input_ids'], tokens['attention_mask']
                )
                
                # Store for distance calculations
                if i == 0:
                    first_repr = repr_tensor
                else:
                    # Calculate distance from first example
                    distance = torch.norm(repr_tensor - first_repr, p=2).item()
                    distances.append(distance)
                    
                    # Calculate cosine similarity
                    similarity = F.cosine_similarity(
                        repr_tensor, first_repr, dim=-1
                    ).item()
                    similarities.append(similarity)
        
        return {
            'avg_distance': np.mean(distances) if distances else 0.0,
            'std_distance': np.std(distances) if distances else 0.0,
            'avg_similarity': np.mean(similarities) if similarities else 0.0,
            'std_similarity': np.std(similarities) if similarities else 0.0,
            'num_examples': len(test_data)
        }

class SimpleMetricsTracker(MetricsTracker):
    """Simple metrics tracking implementation"""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.metrics = defaultdict(list)
        self.step_metrics = defaultdict(list)
        self.timestamps = []
    
    def log_metrics(self, epoch: int, step: int, metrics: Dict[str, float]):
        """Log metrics for a training step"""
        self.timestamps.append(time.time())
        
        for key, value in metrics.items():
            self.metrics[f'epoch_{epoch}_{key}'].append(value)
            self.step_metrics[key].append(value)
    
    def get_metrics(self) -> Dict[str, List[float]]:
        """Get all logged metrics"""
        return dict(self.metrics)
    
    def plot_metrics(self, save_path: str = None):
        """Plot training metrics"""
        if not self.step_metrics:
            logger.warning("No metrics to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        metric_keys = list(self.step_metrics.keys())
        for i, key in enumerate(metric_keys[:4]):  # Plot first 4 metrics
            if i < len(axes):
                axes[i].plot(self.step_metrics[key])
                axes[i].set_title(f'{key.replace("_", " ").title()}')
                axes[i].set_xlabel('Step')
                axes[i].set_ylabel('Value')
                axes[i].grid(True)
        
        # Hide unused subplots
        for i in range(len(metric_keys), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Metrics plot saved to {save_path}")
        
        plt.show()
    
    def save_metrics(self, save_path: str):
        """Save metrics to JSON file"""
        metrics_data = {
            'metrics': dict(self.metrics),
            'step_metrics': dict(self.step_metrics),
            'config': self.config.__dict__
        }
        
        with open(save_path, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        logger.info(f"Metrics saved to {save_path}")

# =============================================================================
# 5. MAIN TRAINER (Orchestration)
# =============================================================================

class ContrastiveSafetyTrainer:
    """Main trainer orchestrating all components"""
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Setup tokenizer and models
        self._setup_models()
        
        # Get model hidden size
        hidden_size = self.original_model.config.hidden_size
        
        # Initialize components via dependency injection
        self.extractor = LayerAverageExtractor(config.representation_layers)
        self.loss_computer = TripletLossComputer(config)
        self.defense = AdversarialDefense(hidden_size, config.adversarial_layers, config)
        self.evaluator = ComprehensiveEvaluator(self.extractor, self.tokenizer, config)
        self.metrics_tracker = SimpleMetricsTracker(config)
        
        # Register hooks
        self.extractor.register_hooks(self.original_model)
        self.extractor.register_hooks(self.defense_model)
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        
        logger.info("ContrastiveSafetyTrainer initialized successfully")
    
    def _setup_models(self):
        """Initialize original and defense models"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.original_model = AutoModel.from_pretrained(self.config.model_name).to(self.device)
        self.defense_model = AutoModel.from_pretrained(self.config.model_name).to(self.device)
        
        # Freeze original model
        for param in self.original_model.parameters():
            param.requires_grad = False
        
        # Put original model in eval mode
        self.original_model.eval()
        
        logger.info(f"Models loaded. Hidden size: {self.original_model.config.hidden_size}")
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler"""
        # Combine defense model and adversarial module parameters
        params = (
            list(self.defense_model.parameters()) + 
            list(self.defense.attack_modules.parameters())
        )
        
        self.optimizer = AdamW(
            params, 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        logger.info(f"Optimizer configured with LR: {self.config.learning_rate}")
    
    def _setup_scheduler(self, total_steps: int):
        """Setup learning rate scheduler"""
        warmup_steps = int(self.config.warmup_ratio * total_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        logger.info(f"Scheduler configured. Warmup steps: {warmup_steps}, Total steps: {total_steps}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        
        # Extract representations from original (frozen) model
        with torch.no_grad():
            benign_orig = self.extractor.extract(
                self.original_model, 
                batch['benign_input_ids'], 
                batch['benign_attention_mask']
            )
            harmful_orig = self.extractor.extract(
                self.original_model, 
                batch['harmful_input_ids'], 
                batch['harmful_attention_mask']
            )
        
        # Extract representations from defense (trainable) model
        benign_new = self.extractor.extract(
            self.defense_model, 
            batch['benign_input_ids'], 
            batch['benign_attention_mask']
        )
        harmful_new = self.extractor.extract(
            self.defense_model, 
            batch['harmful_input_ids'], 
            batch['harmful_attention_mask']
        )
        
        # Apply adversarial defense to some harmful examples
        batch_size = harmful_new.shape[0]
        num_adv = int(batch_size * self.config.adversarial_ratio)
        if num_adv > 0:
            attack_layer = random.choice(self.config.adversarial_layers)
            harmful_new[:num_adv] = self.defense.apply_defense(
                harmful_new[:num_adv], attack_layer
            )
        
        # Compute loss
        loss_dict = self.loss_computer.compute(
            benign_orig, benign_new, harmful_orig, harmful_new
        )
        
        # Convert to float for logging
        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
    
    def train(self, dataloader: DataLoader, num_epochs: Optional[int] = None):
        """Main training loop"""
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        # Setup scheduler
        total_steps = len(dataloader) * num_epochs
        self._setup_scheduler(total_steps)
        
        logger.info(f"Starting training for {num_epochs} epochs, {total_steps} total steps")
        
        self.defense_model.train()
        
        for epoch in range(num_epochs):
            epoch_losses = []
            epoch_start_time = time.time()
            
            # Create progress bar
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                self.optimizer.zero_grad()
                
                # Forward pass
                loss_dict = self.train_step(batch)
                
                # Backward pass
                loss_dict['total_loss'] = torch.tensor(
                    loss_dict['total_loss'], 
                    requires_grad=True, 
                    device=self.device
                )
                loss_dict['total_loss'].backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.defense_model.parameters(), 
                    self.config.gradient_clip
                )
                
                self.optimizer.step()
                self.scheduler.step()
                
                epoch_losses.append(loss_dict['total_loss'].item())
                self.global_step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss_dict['total_loss'].item():.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Log metrics
                if self.global_step % self.config.log_steps == 0:
                    self.metrics_tracker.log_metrics(epoch, self.global_step, loss_dict)
                
                # Periodic logging
                if batch_idx % self.config.log_steps == 0:
                    logger.info(
                        f"Epoch {epoch+1}, Step {self.global_step}: "
                        f"Total Loss: {loss_dict['total_loss'].item():.4f}, "
                        f"Benign: {loss_dict['benign_triplet']:.4f}, "
                        f"Harmful: {loss_dict['harmful_triplet']:.4f}, "
                        f"KL: {loss_dict['kl_loss']:.4f}"
                    )
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint(epoch, self.global_step, loss_dict['total_loss'].item())
            
            # Epoch summary
            epoch_time = time.time() - epoch_start_time
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            
            logger.info(
                f"Epoch {epoch+1} completed in {epoch_time:.2f}s. "
                f"Average loss: {avg_loss:.4f}"
            )
            
            # Save model if best
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self._save_best_model(epoch, avg_loss)
        
        logger.info("Training completed!")
        
        # Save final metrics
        if self.config.save_metrics:
            metrics_path = os.path.join(self.config.output_dir, 'training_metrics.json')
            self.metrics_tracker.save_metrics(metrics_path)
            
            plot_path = os.path.join(self.config.output_dir, 'training_plots.png')
            self.metrics_tracker.plot_metrics(plot_path)
    
    def evaluate(self, test_data: List[str]) -> Dict[str, float]:
        """Evaluate the trained model"""
        logger.info("Evaluating model...")
        self.defense_model.eval()
        
        results = self.evaluator.evaluate(self.defense_model, test_data)
        
        logger.info(f"Evaluation results: {results}")
        return results
    
    def _save_checkpoint(self, epoch: int, step: int, loss: float):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'defense_model_state_dict': self.defense_model.state_dict(),
            'defense_strategy_state_dict': self.defense.attack_modules.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(
            self.config.output_dir, 
            f'checkpoint_epoch_{epoch}_step_{step}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_best_model(self, epoch: int, loss: float):
        """Save the best model"""
        if not self.config.save_model:
            return
        
        model_data = {
            'defense_model_state_dict': self.defense_model.state_dict(),
            'defense_strategy_state_dict': self.defense.attack_modules.state_dict(),
            'config': self.config,
            'epoch': epoch,
            'loss': loss,
            'model_name': self.config.model_name
        }
        
        model_path = os.path.join(self.config.output_dir, 'best_safety_model.pt')
        torch.save(model_data, model_path)
        logger.info(f"Best model saved: {model_path} (loss: {loss:.4f})")
    
    def save(self, path: str):
        """Save the current model state"""
        model_data = {
            'defense_model_state_dict': self.defense_model.state_dict(),
            'defense_strategy_state_dict': self.defense.attack_modules.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'best_loss': self.best_loss
        }
        torch.save(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load a trained model"""
        logger.info(f"Loading model from {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.defense_model.load_state_dict(checkpoint['defense_model_state_dict'])
        self.defense.attack_modules.load_state_dict(checkpoint['defense_strategy_state_dict'])
        
        if 'global_step' in checkpoint:
            self.global_step = checkpoint['global_step']
        if 'best_loss' in checkpoint:
            self.best_loss = checkpoint['best_loss']
        
        logger.info("Model loaded successfully")
    
    def cleanup(self):
        """Clean up resources"""
        self.extractor.cleanup()
        logger.info("Resources cleaned up")

# =============================================================================
# 6. FACTORY AND CONFIGURATION MANAGEMENT
# =============================================================================

class SafetyTrainerFactory:
    """Factory for creating trainers with different configurations"""
    
    @staticmethod
    def create_from_config(config_path: str) -> ContrastiveSafetyTrainer:
        """Create trainer from YAML configuration"""
        config = SafetyConfig.from_yaml(config_path)
        return ContrastiveSafetyTrainer(config)
    
    @staticmethod
    def create_for_research() -> ContrastiveSafetyTrainer:
        """Create trainer with research-friendly defaults"""
        config = SafetyConfig(
            model_name='distilbert-base-uncased',
            representation_layers=list(range(2, 6)),
            learning_rate=2e-5,
            batch_size=4,
            num_epochs=3,
            adversarial_ratio=0.2,
            margin_benign=0.5,
            margin_harmful=1.0,
            output_dir='./research_output'
        )
        return ContrastiveSafetyTrainer(config)
    
    @staticmethod
    def create_for_production() -> ContrastiveSafetyTrainer:
        """Create trainer with production-ready defaults"""
        config = SafetyConfig(
            model_name='bert-base-uncased',
            representation_layers=list(range(6, 12)),
            learning_rate=1e-5,
            batch_size=16,
            num_epochs=10,
            adversarial_ratio=0.3,
            margin_benign=1.0,
            margin_harmful=2.0,
            output_dir='./production_output'
        )
        return ContrastiveSafetyTrainer(config)
    
    @staticmethod
    def create_lightweight() -> ContrastiveSafetyTrainer:
        """Create lightweight trainer for testing"""
        config = SafetyConfig(
            model_name='distilbert-base-uncased',
            representation_layers=[2, 3],
            learning_rate=5e-5,
            batch_size=2,
            num_epochs=1,
            adversarial_ratio=0.1,
            log_steps=5,
            save_steps=20,
            output_dir='./test_output'
        )
        return ContrastiveSafetyTrainer(config)

# =============================================================================
# 7. COMPREHENSIVE DEMO AND USAGE
# =============================================================================

class SafetyTrainingDemo:
    """Comprehensive demonstration of the safety training system"""
    
    def __init__(self):
        self.trainer = None
        self.config = None
        self.dataloader = None
    
    def run_quick_demo(self):
        """Run a quick demonstration"""
        logger.info("🚀 Running Quick Safety Training Demo")
        
        # Create lightweight trainer
        self.trainer = SafetyTrainerFactory.create_lightweight()
        self.config = self.trainer.config
        
        # Create sample data
        benign_data, harmful_data = create_sample_data()
        
        # Create dataset and dataloader
        dataset = SafetyDataset(
            benign_data[:20],  # Small subset for demo
            harmful_data[:20], 
            self.trainer.tokenizer, 
            self.config
        )
        
        self.dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        
        # Train the model
        logger.info("🔧 Starting training...")
        self.trainer.train(self.dataloader)
        
        # Evaluate
        test_texts = [
            "This is a benign test message",
            "How to do something harmful",
            "Normal conversation text",
            "Help me with my homework"
        ]
        
        results = self.trainer.evaluate(test_texts)
        logger.info(f"📊 Evaluation results: {results}")
        
        # Save final model
        model_path = os.path.join(self.config.output_dir, 'demo_model.pt')
        self.trainer.save(model_path)
        
        # Cleanup
        self.trainer.cleanup()
        
        logger.info("✅ Demo completed successfully!")
        return results
    
    def run_research_demo(self):
        """Run a research-oriented demonstration"""
        logger.info("🔬 Running Research Safety Training Demo")
        
        # Create research trainer
        self.trainer = SafetyTrainerFactory.create_for_research()
        self.config = self.trainer.config
        
        # Save configuration
        config_path = os.path.join(self.config.output_dir, 'research_config.yaml')
        self.config.save_to_yaml(config_path)
        
        # Create full dataset
        benign_data, harmful_data = create_sample_data()
        dataset = SafetyDataset(
            benign_data, 
            harmful_data, 
            self.trainer.tokenizer, 
            self.config
        )
        
        self.dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        
        # Train with more epochs
        logger.info("🔧 Starting research training...")
        self.trainer.train(self.dataloader)
        
        # Comprehensive evaluation
        test_texts = [
            "How can I help you today?",
            "What's the weather like?",
            "How to create dangerous substances",
            "Ways to harm other people",
            "Can you explain machine learning?",
            "Instructions for illegal activities",
            "What are some good books to read?",
            "How to break into systems"
        ]
        
        results = self.trainer.evaluate(test_texts)
        
        # Save everything
        self.trainer.save(os.path.join(self.config.output_dir, 'research_model.pt'))
        
        # Cleanup
        self.trainer.cleanup()
        
        logger.info("✅ Research demo completed!")
        return results

def create_custom_config_example():
    """Create example custom configuration"""
    config = SafetyConfig(
        # Model settings
        model_name='distilbert-base-uncased',
        representation_layers=[1, 2, 3, 4],
        max_length=256,
        
        # Loss settings
        alpha=0.6,
        beta=0.3,
        gamma=0.8,
        margin_benign=0.8,
        margin_harmful=1.5,
        temperature=0.2,
        
        # Training settings
        learning_rate=3e-5,
        batch_size=8,
        num_epochs=5,
        warmup_ratio=0.15,
        
        # Adversarial settings
        adversarial_ratio=0.25,
        adversarial_layers=[0, 1, 2],
        adversarial_strength=0.05,
        
        # Output settings
        output_dir='./custom_training',
        log_steps=5,
        save_steps=25
    )
    
    # Save to YAML for reuse
    config.save_to_yaml('./custom_safety_config.yaml')
    return config

# =============================================================================
# 8. COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main entry point with CLI options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Contrastive Safety Training System')
    parser.add_argument('--mode', choices=['demo', 'research', 'custom', 'config'], 
                       default='demo', help='Training mode')
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--output-dir', type=str, default='./safety_output', 
                       help='Output directory')
    parser.add_argument('--model-name', type=str, default='distilbert-base-uncased',
                       help='Hugging Face model name')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        # Quick demo
        demo = SafetyTrainingDemo()
        demo.run_quick_demo()
        
    elif args.mode == 'research':
        # Research demo
        demo = SafetyTrainingDemo()
        demo.run_research_demo()
        
    elif args.mode == 'custom':
        # Custom configuration
        config = SafetyConfig(
            model_name=args.model_name,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir
        )
        
        trainer = ContrastiveSafetyTrainer(config)
        
        # Create data and train
        benign_data, harmful_data = create_sample_data()
        dataset = SafetyDataset(benign_data, harmful_data, trainer.tokenizer, config)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        
        trainer.train(dataloader)
        trainer.save(os.path.join(args.output_dir, 'custom_model.pt'))
        trainer.cleanup()
        
    elif args.mode == 'config':
        # Load from configuration file
        if not args.config:
            logger.error("Config file path required for config mode")
            return
        
        trainer = SafetyTrainerFactory.create_from_config(args.config)
        
        # Create data and train
        benign_data, harmful_data = create_sample_data()
        dataset = SafetyDataset(benign_data, harmful_data, trainer.tokenizer, trainer.config)
        dataloader = DataLoader(dataset, batch_size=trainer.config.batch_size, shuffle=True)
        
        trainer.train(dataloader)
        trainer.save(os.path.join(trainer.config.output_dir, 'config_model.pt'))
        trainer.cleanup()

# =============================================================================
# 9. USAGE EXAMPLES AND TESTING
# =============================================================================

def example_basic_usage():
    """Example of basic usage"""
    print("=== Basic Usage Example ===")
    
    # Create trainer
    trainer = SafetyTrainerFactory.create_lightweight()
    
    # Create data
    benign_data, harmful_data = create_sample_data()
    dataset = SafetyDataset(
        benign_data[:10], 
        harmful_data[:10], 
        trainer.tokenizer, 
        trainer.config
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Train
    trainer.train(dataloader)
    
    # Evaluate
    test_data = ["Hello world", "Harmful content example"]
    results = trainer.evaluate(test_data)
    print(f"Results: {results}")
    
    # Save
    trainer.save('./basic_model.pt')
    trainer.cleanup()

def example_config_usage():
    """Example of configuration-based usage"""
    print("=== Configuration-Based Usage Example ===")
    
    # Create custom config
    config = create_custom_config_example()
    
    # Create trainer from config
    trainer = ContrastiveSafetyTrainer(config)
    
    # Rest of training process...
    print(f"Created trainer with config: {config.model_name}")
    trainer.cleanup()

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Run main CLI
    main()