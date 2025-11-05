"""
Continual Backpropagation implementation.

This module implements the continual backpropagation algorithm described in the paper,
which selectively reinitializes low-utility units to maintain plasticity.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import math


class ContinualBackprop:
    """
    Continual Backpropagation: Selective reinitialization of low-utility units.

    This class wraps a PyTorch model and provides methods to selectively
    reinitialize units based on their contribution utility.
    """

    def __init__(self,
                 model: nn.Module,
                 replacement_rate: float = 1e-5,
                 maturity_threshold: int = 100,
                 decay_rate: float = 0.99,
                 init_method: str = 'kaiming'):
        """
        Initialize ContinualBackprop.

        Args:
            model: PyTorch model to wrap
            replacement_rate: Fraction of units to replace per update
            maturity_threshold: Minimum age before a unit can be replaced
            decay_rate: Decay rate for utility running average
            init_method: Initialization method ('kaiming', 'xavier', etc.)
        """
        self.model = model
        self.replacement_rate = replacement_rate
        self.maturity_threshold = maturity_threshold
        self.decay_rate = decay_rate
        self.init_method = init_method

        # Track utility and age for each layer
        self.utilities = {}
        self.ages = {}
        self.units_to_replace = {}

        # Initialize tracking for each Linear layer
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                num_units = module.out_features
                self.utilities[name] = torch.zeros(num_units)
                self.ages[name] = torch.zeros(num_units, dtype=torch.long)
                self.units_to_replace[name] = 0.0

    def update_utility(self, activations: Dict[str, torch.Tensor]):
        """
        Update contribution utility for each layer based on activations.

        Args:
            activations: Dictionary mapping layer names to activation tensors
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in activations:
                h = activations[name]  # (batch_size, num_units)
                w = module.weight.data  # (out_features, in_features)

                # Compute instantaneous contribution utility
                # |h_i| * sum_k |w_ik|
                h_mean = torch.abs(h).mean(dim=0)  # Average over batch

                # Sum absolute outgoing weights for each input unit
                # Note: w is (out_features, in_features), so we need to sum over out_features
                # But for hidden units, they are the in_features of the next layer
                # So we need to think about this differently

                # Actually, for a hidden layer, we want contribution to downstream
                # h is output of this layer, w is the next layer's weights
                # We'll compute it as just |h| for simplicity, weighted by outgoing connections

                # Get outgoing weights (this layer's weight matrix)
                outgoing_weights = torch.abs(w).sum(dim=0)  # Sum over output dimension

                instantaneous_utility = h_mean * outgoing_weights

                # Update running average
                device = h.device
                self.utilities[name] = (self.decay_rate * self.utilities[name].to(device) +
                                       (1 - self.decay_rate) * instantaneous_utility)

                # Increment ages
                self.ages[name] = self.ages[name] + 1

    def reinitialize_units(self):
        """
        Reinitialize low-utility units based on replacement rate.

        This implements Algorithm 1 from the paper.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in self.utilities:
                # Update count of units to replace
                num_units = module.out_features
                mature_mask = self.ages[name] >= self.maturity_threshold
                num_eligible = mature_mask.sum().item()

                self.units_to_replace[name] += num_eligible * self.replacement_rate

                # Reinitialize units if count >= 1
                while self.units_to_replace[name] >= 1.0:
                    # Find unit with lowest utility among mature units
                    utilities = self.utilities[name].clone()
                    utilities[~mature_mask] = float('inf')

                    if utilities.min() == float('inf'):
                        break  # No mature units to replace

                    unit_idx = utilities.argmin().item()

                    # Reinitialize this unit
                    self._reinit_unit(module, unit_idx, name)

                    # Reset utility and age
                    self.utilities[name][unit_idx] = 0
                    self.ages[name][unit_idx] = 0

                    # Update mature mask
                    mature_mask[unit_idx] = False

                    # Decrement count
                    self.units_to_replace[name] -= 1.0

    def _reinit_unit(self, module: nn.Linear, unit_idx: int, layer_name: str):
        """
        Reinitialize a specific unit in a layer.

        Input weights are reinitialized from the original distribution.
        Output weights are set to zero to avoid disrupting learned function.

        Args:
            module: The Linear module
            unit_idx: Index of the unit to reinitialize
            layer_name: Name of the layer (for tracking)
        """
        with torch.no_grad():
            # Reinitialize incoming weights
            in_features = module.in_features
            out_features = module.out_features

            if self.init_method == 'kaiming':
                # Kaiming initialization for ReLU
                std = math.sqrt(2.0 / in_features)
                module.weight.data[unit_idx] = torch.randn(in_features) * std
            elif self.init_method == 'xavier':
                std = math.sqrt(2.0 / (in_features + out_features))
                module.weight.data[unit_idx] = torch.randn(in_features) * std
            else:
                # Default: standard normal scaled by 1/sqrt(in_features)
                module.weight.data[unit_idx] = torch.randn(in_features) / math.sqrt(in_features)

            # Reset bias if it exists
            if module.bias is not None:
                module.bias.data[unit_idx] = 0

            # Note: Output weights are part of the NEXT layer
            # We need to find and zero them
            self._zero_outgoing_weights(layer_name, unit_idx)

    def _zero_outgoing_weights(self, layer_name: str, unit_idx: int):
        """
        Zero out the outgoing weights from a unit.

        Args:
            layer_name: Name of the layer containing the unit
            unit_idx: Index of the unit
        """
        # Find the next layer and zero the corresponding input weights
        layer_names = [name for name, _ in self.model.named_modules()
                      if isinstance(_, nn.Linear)]

        if layer_name in layer_names:
            layer_idx = layer_names.index(layer_name)
            if layer_idx + 1 < len(layer_names):
                next_layer_name = layer_names[layer_idx + 1]
                next_module = dict(self.model.named_modules())[next_layer_name]

                with torch.no_grad():
                    # Zero the weights connecting from unit_idx
                    next_module.weight.data[:, unit_idx] = 0

    def get_statistics(self) -> Dict:
        """
        Get statistics about the current state.

        Returns:
            Dictionary with statistics about utilities, ages, etc.
        """
        stats = {}

        for name in self.utilities.keys():
            mature_mask = self.ages[name] >= self.maturity_threshold
            stats[f'{name}_avg_utility'] = self.utilities[name].mean().item()
            stats[f'{name}_min_utility'] = self.utilities[name].min().item()
            stats[f'{name}_max_utility'] = self.utilities[name].max().item()
            stats[f'{name}_mature_units'] = mature_mask.sum().item()
            stats[f'{name}_avg_age'] = self.ages[name].float().mean().item()

        return stats


def create_activation_hook(activations_dict: Dict, name: str):
    """
    Create a forward hook to capture activations.

    Args:
        activations_dict: Dictionary to store activations
        name: Name/key for this layer

    Returns:
        Hook function
    """
    def hook(module, input, output):
        activations_dict[name] = output.detach()
    return hook


class ContinualLearner:
    """
    Wrapper that combines a model, optimizer, and continual backprop.
    """

    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 use_continual_backprop: bool = True,
                 replacement_rate: float = 1e-5,
                 maturity_threshold: int = 100):
        self.model = model
        self.optimizer = optimizer
        self.use_continual_backprop = use_continual_backprop

        if use_continual_backprop:
            self.continual_bp = ContinualBackprop(
                model,
                replacement_rate=replacement_rate,
                maturity_threshold=maturity_threshold
            )
        else:
            self.continual_bp = None

        # Hook to capture activations
        self.activations = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(
                    create_activation_hook(self.activations, name)
                )
                self.hooks.append(hook)

    def train_step(self, data, target, criterion):
        """
        Perform one training step with optional continual backprop.

        Args:
            data: Input batch
            target: Target labels
            criterion: Loss function

        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        output = self.model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Continual backprop step
        if self.use_continual_backprop:
            self.continual_bp.update_utility(self.activations)
            self.continual_bp.reinitialize_units()

        return loss.item()

    def cleanup(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()
