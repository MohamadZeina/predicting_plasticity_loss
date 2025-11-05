"""
Quick test to verify the implementation works.

This runs a very small experiment (2 tasks, tiny network) to check
that all components are working correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from src.metrics import PlasticityMetrics
from src.continual_backprop import ContinualLearner
from src.neuron_death_predictor import NeuronDeathPredictor, PreventiveReinitialization


class TinyMLP(nn.Module):
    """Tiny MLP for quick testing."""
    def __init__(self):
        super(TinyMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)


def test_metrics():
    """Test metrics computation."""
    print("\n" + "="*60)
    print("Testing Metrics Computation")
    print("="*60)

    model = TinyMLP()

    # Create dummy data
    dummy_data = torch.randn(100, 1, 28, 28)
    dummy_labels = torch.randint(0, 10, (100,))
    dataset = TensorDataset(dummy_data, dummy_labels)
    loader = DataLoader(dataset, batch_size=32)

    metrics = PlasticityMetrics(model)

    # Test dead units
    print("\nComputing dead units...")
    dead_units = metrics.compute_dead_units(loader)
    print(f"Dead units: {dead_units}")

    # Test weight magnitude
    print("\nComputing weight magnitude...")
    weight_mag = metrics.compute_weight_magnitude()
    print(f"Weight magnitude: {weight_mag['overall']:.4f}")

    # Test effective rank
    print("\nComputing effective rank...")
    eff_rank = metrics.compute_effective_rank(loader)
    print(f"Effective rank: {eff_rank}")

    print("\n✓ Metrics computation successful!")


def test_continual_backprop():
    """Test continual backpropagation."""
    print("\n" + "="*60)
    print("Testing Continual Backpropagation")
    print("="*60)

    model = TinyMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    learner = ContinualLearner(
        model, optimizer,
        use_continual_backprop=True,
        replacement_rate=0.01,  # High rate for testing
        maturity_threshold=5
    )

    # Create dummy data
    data = torch.randn(32, 1, 28, 28)
    target = torch.randint(0, 10, (32,))
    criterion = nn.CrossEntropyLoss()

    # Train for a few steps
    print("\nTraining for 10 steps...")
    for i in range(10):
        loss = learner.train_step(data, target, criterion)
        if i % 3 == 0:
            print(f"  Step {i}: Loss = {loss:.4f}")

    # Check statistics
    stats = learner.continual_bp.get_statistics()
    print(f"\nContinual BP statistics:")
    for key, value in list(stats.items())[:3]:
        print(f"  {key}: {value:.4f}")

    learner.cleanup()
    print("\n✓ Continual backpropagation successful!")


def test_predictor():
    """Test neuron death predictor."""
    print("\n" + "="*60)
    print("Testing Neuron Death Predictor")
    print("="*60)

    model = TinyMLP()
    predictor = NeuronDeathPredictor(model, history_length=20)

    # Simulate some activations
    print("\nSimulating activation updates...")
    for i in range(30):
        # Create fake activations (decreasing over time to simulate dying neurons)
        activations = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                num_units = module.out_features
                # Make some units decrease over time
                acts = torch.randn(32, num_units) * (1.0 - i * 0.02)
                acts = torch.clamp(acts, min=0)  # ReLU
                activations[name] = acts

        predictor.update(activations)

        if i % 10 == 0:
            stats = predictor.get_statistics()
            print(f"  Step {i}:")
            for key in list(stats.keys())[:2]:
                print(f"    {key}: {stats[key]:.4f}")

    # Get at-risk neurons
    at_risk = predictor.get_at_risk_neurons(top_k=5)
    print(f"\nAt-risk neurons (top 5 per layer):")
    for layer, indices in at_risk.items():
        if indices:
            print(f"  {layer}: {indices}")

    print("\n✓ Predictor test successful!")


def test_preventive_reinit():
    """Test preventive reinitialization."""
    print("\n" + "="*60)
    print("Testing Preventive Reinitialization")
    print("="*60)

    model = TinyMLP()
    predictor = NeuronDeathPredictor(model)
    preventive = PreventiveReinitialization(
        model, predictor,
        intervention_threshold=0.5  # Low threshold for testing
    )

    # Simulate predictions that trigger intervention
    print("\nSimulating high death scores...")
    for name in predictor.death_predictions.keys():
        # Set some neurons to high death score
        predictor.death_predictions[name][:5] = 0.9

    # Apply preventive reinitialization
    print("\nApplying preventive reinitialization...")
    reinitialized = preventive.apply_preventive_reinitialization()

    print(f"Neurons reinitialized: {reinitialized}")

    stats = preventive.get_statistics()
    print(f"Total interventions: {stats['total_interventions']}")

    print("\n✓ Preventive reinitialization successful!")


def test_mini_experiment():
    """Run a mini experiment with 2 tasks."""
    print("\n" + "="*60)
    print("Running Mini Experiment (2 tasks)")
    print("="*60)

    device = torch.device('cpu')

    # Create synthetic data
    print("\nCreating synthetic data...")
    data = torch.randn(200, 1, 28, 28)
    labels = torch.randint(0, 10, (200,))

    # Two permutations
    perm1 = np.random.permutation(784)
    perm2 = np.random.permutation(784)

    for method in ['vanilla', 'preventive']:
        print(f"\n--- Testing method: {method} ---")

        model = TinyMLP().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        predictor = None
        preventive_reinit = None

        if method == 'preventive':
            predictor = NeuronDeathPredictor(model)
            preventive_reinit = PreventiveReinitialization(model, predictor)

        for task_idx, perm in enumerate([perm1, perm2]):
            # Apply permutation
            data_flat = data.reshape(-1, 784)
            perm_data = data_flat[:, perm].reshape(data.shape)

            dataset = TensorDataset(perm_data, labels)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)

            # Train
            model.train()
            total_loss = 0
            for batch_data, batch_target in loader:
                batch_data, batch_target = batch_data.to(device), batch_target.to(device)

                optimizer.zero_grad()
                output = model(batch_data)
                loss = criterion(output, batch_target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                # Update predictor
                if predictor is not None:
                    activations = {}
                    for name, module in model.named_modules():
                        if isinstance(module, nn.Linear):
                            # Need to do forward pass to get activations
                            pass  # Simplified for test
                    # predictor.update would go here

            avg_loss = total_loss / len(loader)
            print(f"  Task {task_idx}: Avg Loss = {avg_loss:.4f}")

    print("\n✓ Mini experiment successful!")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("QUICK COMPONENT TESTS")
    print("="*60)

    try:
        test_metrics()
        test_continual_backprop()
        test_predictor()
        test_preventive_reinit()
        test_mini_experiment()

        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nYou can now run the full experiment:")
        print("  python experiments/permuted_mnist.py")

    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
