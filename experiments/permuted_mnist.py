"""
Minimal experiment to reproduce plasticity loss on Permuted MNIST.

This experiment:
1. Trains a small network on a sequence of permuted MNIST tasks
2. Tracks plasticity metrics (dead units, weight magnitude, stable rank)
3. Compares vanilla SGD, L2 regularization, continual backprop, and preventive reinitialization
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from src.metrics import PlasticityMetrics, ContributionUtility
from src.continual_backprop import ContinualLearner
from src.neuron_death_predictor import NeuronDeathPredictor, PreventiveReinitialization


class SimpleMLP(nn.Module):
    """Simple MLP for MNIST."""

    def __init__(self, input_size=784, hidden_sizes=[256, 256], output_size=10):
        super(SimpleMLP, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.network(x)


def create_permuted_mnist(original_data, seed=None):
    """
    Create a permuted version of MNIST.

    Args:
        original_data: Original MNIST data (images, labels)
        seed: Random seed for permutation

    Returns:
        Permuted dataset
    """
    if seed is not None:
        np.random.seed(seed)

    images, labels = original_data
    # Create random permutation
    perm = np.random.permutation(784)

    # Apply permutation to images
    images_flat = images.reshape(-1, 784)
    permuted_images = images_flat[:, perm]

    return permuted_images.reshape(images.shape), labels, perm


def load_mnist_data(num_samples=10000):
    """
    Load a subset of MNIST data.

    For this minimal experiment, we'll use a smaller subset.
    """
    try:
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST('../data', train=True, download=True,
                                      transform=transform)

        # Take subset
        indices = torch.randperm(len(train_dataset))[:num_samples]
        images = []
        labels = []

        for idx in indices:
            img, label = train_dataset[idx]
            images.append(img)
            labels.append(label)

        images = torch.stack(images)
        labels = torch.tensor(labels)

        return images, labels

    except Exception as e:
        print(f"Error loading MNIST: {e}")
        print("Creating synthetic data for testing...")

        # Create synthetic data
        images = torch.randn(num_samples, 1, 28, 28)
        labels = torch.randint(0, 10, (num_samples,))

        return images, labels


def train_one_task(model, optimizer, train_loader, device, criterion,
                  continual_learner=None, predictor=None, preventive_reinit=None):
    """
    Train on one task (one permutation).

    Args:
        model: Neural network model
        optimizer: Optimizer
        train_loader: DataLoader for training data
        device: Device to train on
        criterion: Loss function
        continual_learner: ContinualLearner instance (if using continual backprop)
        predictor: NeuronDeathPredictor instance
        preventive_reinit: PreventiveReinitialization instance

    Returns:
        Average loss, accuracy
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # For tracking activations and gradients
    activations = {}
    gradients = {}

    def save_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    def save_gradient(name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                gradients[name] = grad_output[0].detach()
        return hook

    # Register hooks if we're using predictor
    hooks = []
    if predictor is not None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(save_activation(name)))
                hooks.append(module.register_backward_hook(save_gradient(name)))

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        if continual_learner is not None:
            # Use continual learner
            loss = continual_learner.train_step(data, target, criterion)
        else:
            # Standard training
            optimizer.zero_grad()
            output = model(data)
            loss_val = criterion(output, target)
            loss_val.backward()
            optimizer.step()
            loss = loss_val.item()

        # Update predictor if available
        if predictor is not None and len(activations) > 0:
            # Get utilities if we have continual learner
            utilities = None
            if continual_learner and continual_learner.use_continual_backprop:
                utilities = continual_learner.continual_bp.utilities

            predictor.update(activations, gradients, utilities)

            # Apply preventive reinitialization if needed
            if preventive_reinit is not None and batch_idx % 10 == 0:
                preventive_reinit.apply_preventive_reinitialization()

        total_loss += loss
        with torch.no_grad():
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def evaluate(model, test_loader, device, criterion):
    """Evaluate model on test data."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def run_experiment(config):
    """
    Run the plasticity loss experiment.

    Args:
        config: Dictionary with experiment configuration
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading MNIST data...")
    images, labels = load_mnist_data(num_samples=config.get('num_samples', 10000))

    # Split into train/test
    split_idx = int(0.8 * len(images))
    train_images, test_images = images[:split_idx], images[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]

    print(f"Training samples: {len(train_images)}, Test samples: {len(test_images)}")

    # Results storage
    results = {
        'config': config,
        'tasks': []
    }

    # Run experiment for each method
    methods = config.get('methods', ['vanilla', 'l2', 'continual_backprop', 'preventive'])

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Running experiment with method: {method}")
        print(f"{'='*60}\n")

        # Create model
        model = SimpleMLP(
            hidden_sizes=config.get('hidden_sizes', [256, 256])
        ).to(device)

        # Create optimizer
        weight_decay = config.get('weight_decay', 0.0)
        if method == 'l2':
            weight_decay = config.get('l2_weight_decay', 5e-4)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.get('lr', 0.01),
            momentum=config.get('momentum', 0.9),
            weight_decay=weight_decay
        )

        criterion = nn.CrossEntropyLoss()

        # Create continual learner if needed
        continual_learner = None
        if method == 'continual_backprop':
            continual_learner = ContinualLearner(
                model, optimizer,
                use_continual_backprop=True,
                replacement_rate=config.get('replacement_rate', 1e-5),
                maturity_threshold=config.get('maturity_threshold', 100)
            )

        # Create predictor and preventive reinit if needed
        predictor = None
        preventive_reinit = None
        if method == 'preventive':
            predictor = NeuronDeathPredictor(model)
            preventive_reinit = PreventiveReinitialization(
                model, predictor,
                intervention_threshold=config.get('intervention_threshold', 0.8)
            )

        # Create metrics tracker
        metrics_tracker = PlasticityMetrics(model)

        # Task loop
        num_tasks = config.get('num_tasks', 20)
        task_results = []

        for task_idx in tqdm(range(num_tasks), desc=f"{method}"):
            # Create permuted task
            perm_train_images, perm_train_labels, perm = create_permuted_mnist(
                (train_images, train_labels),
                seed=task_idx
            )
            perm_test_images, perm_test_labels, _ = create_permuted_mnist(
                (test_images, test_labels),
                seed=task_idx
            )

            # Create dataloaders
            train_dataset = TensorDataset(
                torch.tensor(perm_train_images, dtype=torch.float32),
                torch.tensor(perm_train_labels, dtype=torch.long)
            )
            test_dataset = TensorDataset(
                torch.tensor(perm_test_images, dtype=torch.float32),
                torch.tensor(perm_test_labels, dtype=torch.long)
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=config.get('batch_size', 128),
                shuffle=True
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=config.get('batch_size', 128),
                shuffle=False
            )

            # Train on this task
            train_loss, train_acc = train_one_task(
                model, optimizer, train_loader, device, criterion,
                continual_learner=continual_learner,
                predictor=predictor,
                preventive_reinit=preventive_reinit
            )

            # Evaluate
            test_loss, test_acc = evaluate(model, test_loader, device, criterion)

            # Compute metrics
            dead_units = metrics_tracker.compute_dead_units(test_loader, device)
            weight_mag = metrics_tracker.compute_weight_magnitude()
            effective_rank = metrics_tracker.compute_effective_rank(test_loader, device)

            # Store results
            task_result = {
                'task_idx': task_idx,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'dead_units': dead_units,
                'weight_magnitude': weight_mag,
                'effective_rank': effective_rank
            }

            # Add predictor stats if available
            if predictor is not None:
                task_result['predictor_stats'] = predictor.get_statistics()

            if preventive_reinit is not None:
                task_result['intervention_stats'] = preventive_reinit.get_statistics()

            task_results.append(task_result)

            # Print progress
            if (task_idx + 1) % 5 == 0:
                print(f"\nTask {task_idx + 1}/{num_tasks}")
                print(f"  Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
                print(f"  Avg Dead Units: {np.mean(list(dead_units.values())):.2f}%")
                print(f"  Avg Weight Mag: {weight_mag['overall']:.4f}")

        results[method] = task_results

        # Cleanup
        if continual_learner is not None:
            continual_learner.cleanup()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"../results/permuted_mnist_{timestamp}.json"

    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, 'w') as f:
        # Convert tensors to lists for JSON serialization
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))

    print(f"\nResults saved to: {results_file}")

    return results


if __name__ == "__main__":
    # Configuration
    config = {
        'num_samples': 10000,  # Small dataset for quick experiments
        'num_tasks': 20,  # Number of permuted MNIST tasks
        'hidden_sizes': [256, 256],  # Network architecture
        'lr': 0.01,
        'momentum': 0.9,
        'batch_size': 128,
        'weight_decay': 0.0,
        'l2_weight_decay': 5e-4,
        'replacement_rate': 1e-5,
        'maturity_threshold': 100,
        'intervention_threshold': 0.7,
        'methods': ['vanilla', 'l2', 'continual_backprop', 'preventive']
    }

    print("="*60)
    print("Plasticity Loss Experiment: Permuted MNIST")
    print("="*60)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    results = run_experiment(config)

    print("\n" + "="*60)
    print("Experiment completed!")
    print("="*60)
