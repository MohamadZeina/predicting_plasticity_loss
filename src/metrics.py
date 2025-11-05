"""
Metrics for tracking plasticity loss in neural networks.

This module implements the key metrics described in the paper:
- Dead/dormant units (ReLU units with zero activation)
- Weight magnitude
- Stable rank of representations
- Contribution utility
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


class PlasticityMetrics:
    """Track metrics related to plasticity loss."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.activations = {}
        self.utility_history = {}

    def compute_dead_units(self, dataloader: torch.utils.data.DataLoader,
                          device: str = 'cpu',
                          threshold: float = 0.01) -> Dict[str, float]:
        """
        Compute percentage of dead units in each layer.

        A unit is considered dead if it outputs zero (or near-zero) for all examples.

        Args:
            dataloader: DataLoader with sample data
            device: Device to run computation on
            threshold: Threshold below which activation is considered zero

        Returns:
            Dictionary mapping layer names to percentage of dead units
        """
        self.model.eval()
        layer_activations = {}

        # Register hooks to capture activations
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                def hook_fn(mod, inp, out, name=name):
                    if name not in layer_activations:
                        layer_activations[name] = []
                    layer_activations[name].append(out.detach().cpu())
                hooks.append(module.register_forward_hook(hook_fn))

        # Forward pass through data
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(dataloader):
                if batch_idx >= 20:  # Sample only first 20 batches for efficiency
                    break
                data = data.to(device)
                _ = self.model(data)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Compute dead unit percentages
        dead_unit_percentages = {}
        for name, activations in layer_activations.items():
            # Concatenate all activations
            all_acts = torch.cat(activations, dim=0)

            # Reshape to (num_samples, num_units)
            if len(all_acts.shape) > 2:
                all_acts = all_acts.reshape(all_acts.shape[0], -1)

            # Check which units are always below threshold
            max_activation = all_acts.max(dim=0)[0]
            dead_units = (max_activation < threshold).float().mean().item()
            dead_unit_percentages[name] = dead_units * 100

        return dead_unit_percentages

    def compute_weight_magnitude(self) -> Dict[str, float]:
        """
        Compute average absolute weight magnitude for each layer.

        Returns:
            Dictionary mapping layer names to average weight magnitude
        """
        weight_magnitudes = {}

        for name, param in self.model.named_parameters():
            if 'weight' in name:
                avg_mag = param.abs().mean().item()
                weight_magnitudes[name] = avg_mag

        # Also compute overall average
        all_weights = []
        for param in self.model.parameters():
            all_weights.append(param.abs().flatten())
        weight_magnitudes['overall'] = torch.cat(all_weights).mean().item()

        return weight_magnitudes

    def compute_stable_rank(self, dataloader: torch.utils.data.DataLoader,
                           device: str = 'cpu') -> Dict[str, float]:
        """
        Compute stable rank of representations in each layer.

        Stable rank is defined as ||σ||₁² / ||σ||₂² where σ are singular values.
        A lower stable rank indicates less diversity in the representation.

        Args:
            dataloader: DataLoader with sample data
            device: Device to run computation on

        Returns:
            Dictionary mapping layer names to stable rank
        """
        self.model.eval()
        layer_representations = {}

        # Register hooks to capture pre-activation values
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                def hook_fn(mod, inp, out, name=name):
                    if name not in layer_representations:
                        layer_representations[name] = []
                    layer_representations[name].append(inp[0].detach().cpu())
                hooks.append(module.register_forward_hook(hook_fn))

        # Forward pass through data
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(dataloader):
                if batch_idx >= 20:  # Sample only first 20 batches
                    break
                data = data.to(device)
                _ = self.model(data)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Compute stable rank
        stable_ranks = {}
        for name, representations in layer_representations.items():
            # Concatenate all representations
            all_reps = torch.cat(representations, dim=0)

            # Reshape to (num_samples, num_features)
            if len(all_reps.shape) > 2:
                all_reps = all_reps.reshape(all_reps.shape[0], -1)

            # Compute SVD
            try:
                U, S, V = torch.svd(all_reps.float())

                # Stable rank = ||σ||₁² / ||σ||₂²
                if S.sum() > 0:
                    stable_rank = (S.sum() ** 2) / (S ** 2).sum()
                    stable_ranks[name] = stable_rank.item()
                else:
                    stable_ranks[name] = 0.0
            except:
                stable_ranks[name] = 0.0

        return stable_ranks

    def compute_effective_rank(self, dataloader: torch.utils.data.DataLoader,
                              device: str = 'cpu') -> Dict[str, float]:
        """
        Compute effective rank of representations (as in the paper).

        Effective rank = exp(H(p₁, p₂, ..., pₖ)) where pₖ = σₖ/||σ||₁
        and H is the Shannon entropy.

        Args:
            dataloader: DataLoader with sample data
            device: Device to run computation on

        Returns:
            Dictionary mapping layer names to effective rank
        """
        self.model.eval()
        layer_representations = {}

        # Register hooks to capture activations
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                def hook_fn(mod, inp, out, name=name):
                    if name not in layer_representations:
                        layer_representations[name] = []
                    layer_representations[name].append(out.detach().cpu())
                hooks.append(module.register_forward_hook(hook_fn))

        # Forward pass through data
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(dataloader):
                if batch_idx >= 20:
                    break
                data = data.to(device)
                _ = self.model(data)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Compute effective rank
        effective_ranks = {}
        for name, representations in layer_representations.items():
            # Concatenate all representations
            all_reps = torch.cat(representations, dim=0)

            # Reshape to (num_samples, num_features)
            if len(all_reps.shape) > 2:
                all_reps = all_reps.reshape(all_reps.shape[0], -1)

            # Compute SVD
            try:
                U, S, V = torch.svd(all_reps.float())

                # Normalize singular values to get probabilities
                S = S[S > 1e-10]  # Remove near-zero singular values
                if len(S) > 0:
                    p = S / S.sum()

                    # Compute Shannon entropy
                    entropy = -(p * torch.log(p + 1e-10)).sum()

                    # Effective rank
                    eff_rank = torch.exp(entropy).item()
                    effective_ranks[name] = eff_rank
                else:
                    effective_ranks[name] = 0.0
            except:
                effective_ranks[name] = 0.0

        return effective_ranks


class ContributionUtility:
    """
    Compute contribution utility as described in the paper (Equation 1).

    The contribution utility measures how much each unit contributes to
    its downstream consumers.
    """

    def __init__(self, model: torch.nn.Module, decay_rate: float = 0.99):
        self.model = model
        self.decay_rate = decay_rate
        self.utilities = {}
        self.ages = {}

        # Initialize utilities for each layer
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                num_units = module.out_features
                self.utilities[name] = torch.zeros(num_units)
                self.ages[name] = torch.zeros(num_units)

    def update(self, batch_data: torch.Tensor, device: str = 'cpu'):
        """
        Update contribution utility based on a batch of data.

        Args:
            batch_data: Input batch
            device: Device to run computation on
        """
        self.model.eval()
        activations = {}

        # Register hooks to capture activations
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                def hook_fn(mod, inp, out, name=name):
                    activations[name] = out.detach()
                hooks.append(module.register_forward_hook(hook_fn))

        # Forward pass
        with torch.no_grad():
            batch_data = batch_data.to(device)
            _ = self.model(batch_data)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Update utilities
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and name in activations:
                h = activations[name]  # (batch_size, num_units)
                w = module.weight.data  # (out_features, in_features)

                # Compute instantaneous contribution
                # For each hidden unit i, sum over consumers k: |h_i * w_ik|
                contribution = torch.abs(h).mean(dim=0)  # Average over batch

                # For output contribution, multiply by outgoing weights
                if w.shape[0] > 0:  # Has outgoing connections
                    # Sum of |weight| for each input unit
                    outgoing_weight_sum = torch.abs(w).sum(dim=0)
                    instantaneous_utility = contribution * outgoing_weight_sum
                else:
                    instantaneous_utility = contribution

                # Update running average
                self.utilities[name] = (self.decay_rate * self.utilities[name].to(device) +
                                       (1 - self.decay_rate) * instantaneous_utility)

                # Update ages
                self.ages[name] = self.ages[name] + 1

    def get_lowest_utility_units(self, layer_name: str,
                                 k: int,
                                 maturity_threshold: int = 100) -> List[int]:
        """
        Get indices of k units with lowest utility in a layer.

        Args:
            layer_name: Name of the layer
            k: Number of units to return
            maturity_threshold: Minimum age for a unit to be considered

        Returns:
            List of unit indices with lowest utility
        """
        if layer_name not in self.utilities:
            return []

        utilities = self.utilities[layer_name].cpu()
        ages = self.ages[layer_name].cpu()

        # Mask out immature units
        mature_mask = ages >= maturity_threshold
        masked_utilities = utilities.clone()
        masked_utilities[~mature_mask] = float('inf')

        # Get k lowest utility units
        _, indices = torch.topk(masked_utilities, k=min(k, len(masked_utilities)),
                                largest=False)

        return indices.tolist()
