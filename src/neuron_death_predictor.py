"""
Neuron Death Prediction System.

This module implements methods to predict which neurons will become inactive,
allowing us to take preventive action before they die.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque


class NeuronDeathPredictor:
    """
    Predict which neurons are likely to become dead in the near future.

    This uses multiple signals:
    1. Trend in activation magnitudes (decreasing → likely to die)
    2. Gradient flow (decreasing → likely to die)
    3. Contribution utility trend
    4. Distance from decision boundary
    """

    def __init__(self,
                 model: nn.Module,
                 history_length: int = 100,
                 prediction_threshold: float = 0.7):
        """
        Initialize the predictor.

        Args:
            model: PyTorch model to monitor
            history_length: How many timesteps of history to maintain
            prediction_threshold: Threshold for predicting death (0-1)
        """
        self.model = model
        self.history_length = history_length
        self.prediction_threshold = prediction_threshold

        # Track history for each layer
        self.activation_history = {}
        self.gradient_history = {}
        self.utility_history = {}
        self.death_predictions = {}

        # Initialize tracking for each Linear layer
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                num_units = module.out_features
                self.activation_history[name] = {
                    i: deque(maxlen=history_length) for i in range(num_units)
                }
                self.gradient_history[name] = {
                    i: deque(maxlen=history_length) for i in range(num_units)
                }
                self.utility_history[name] = {
                    i: deque(maxlen=history_length) for i in range(num_units)
                }
                self.death_predictions[name] = torch.zeros(num_units)

    def update(self,
               activations: Dict[str, torch.Tensor],
               gradients: Optional[Dict[str, torch.Tensor]] = None,
               utilities: Optional[Dict[str, torch.Tensor]] = None):
        """
        Update histories with new observations.

        Args:
            activations: Dictionary of layer activations
            gradients: Dictionary of layer gradients (optional)
            utilities: Dictionary of layer utilities (optional)
        """
        for name, activation in activations.items():
            if name not in self.activation_history:
                continue

            # Compute mean activation per unit (average over batch)
            mean_activation = torch.abs(activation).mean(dim=0).cpu()

            for unit_idx in range(len(mean_activation)):
                self.activation_history[name][unit_idx].append(
                    mean_activation[unit_idx].item()
                )

        if gradients is not None:
            for name, gradient in gradients.items():
                if name not in self.gradient_history:
                    continue

                mean_gradient = torch.abs(gradient).mean(dim=0).cpu()

                for unit_idx in range(len(mean_gradient)):
                    self.gradient_history[name][unit_idx].append(
                        mean_gradient[unit_idx].item()
                    )

        if utilities is not None:
            for name, utility in utilities.items():
                if name not in self.utility_history:
                    continue

                for unit_idx in range(len(utility)):
                    self.utility_history[name][unit_idx].append(
                        utility[unit_idx].item()
                    )

        # Update predictions
        self._update_predictions()

    def _update_predictions(self):
        """
        Update death predictions based on current histories.

        A neuron is predicted to die if:
        1. Its activation is trending downward
        2. Its gradient is very small or decreasing
        3. Its utility is decreasing
        4. Its current activation is already low
        """
        for name in self.activation_history.keys():
            num_units = len(self.activation_history[name])

            for unit_idx in range(num_units):
                score = self._compute_death_score(name, unit_idx)
                self.death_predictions[name][unit_idx] = score

    def _compute_death_score(self, layer_name: str, unit_idx: int) -> float:
        """
        Compute a score (0-1) indicating likelihood of death.

        Higher score = more likely to die soon.

        Args:
            layer_name: Name of the layer
            unit_idx: Index of the unit

        Returns:
            Death score between 0 and 1
        """
        scores = []

        # 1. Activation trend score
        act_history = list(self.activation_history[layer_name][unit_idx])
        if len(act_history) >= 10:
            # Compute trend (linear regression slope)
            trend_score = self._compute_trend_score(act_history)
            scores.append(trend_score)

            # Current activation magnitude
            current_act = act_history[-1] if act_history else 0
            magnitude_score = 1.0 / (1.0 + current_act)  # Lower activation → higher score
            scores.append(magnitude_score)

        # 2. Gradient trend score
        grad_history = list(self.gradient_history[layer_name][unit_idx])
        if len(grad_history) >= 10:
            grad_trend_score = self._compute_trend_score(grad_history)
            scores.append(grad_trend_score)

            # Current gradient magnitude
            current_grad = grad_history[-1] if grad_history else 0
            grad_magnitude_score = 1.0 / (1.0 + current_grad * 100)
            scores.append(grad_magnitude_score)

        # 3. Utility trend score
        util_history = list(self.utility_history[layer_name][unit_idx])
        if len(util_history) >= 10:
            util_trend_score = self._compute_trend_score(util_history)
            scores.append(util_trend_score)

        # Combine scores (average)
        if scores:
            return np.mean(scores)
        else:
            return 0.0

    def _compute_trend_score(self, history: List[float]) -> float:
        """
        Compute trend score from history.

        Negative trend (decreasing) → high score
        Positive trend (increasing) → low score

        Args:
            history: List of historical values

        Returns:
            Score between 0 and 1
        """
        if len(history) < 2:
            return 0.5

        # Simple linear regression
        x = np.arange(len(history))
        y = np.array(history)

        # Handle constant values
        if np.std(y) < 1e-10:
            return 0.5

        # Compute slope
        slope = np.polyfit(x, y, 1)[0]

        # Normalize slope to 0-1 score
        # Negative slope → high score
        # Use sigmoid-like function
        score = 1.0 / (1.0 + np.exp(slope * 10))

        return float(np.clip(score, 0, 1))

    def get_at_risk_neurons(self,
                           top_k: Optional[int] = None,
                           threshold: Optional[float] = None) -> Dict[str, List[int]]:
        """
        Get neurons that are at risk of dying.

        Args:
            top_k: Return top k at-risk neurons per layer (optional)
            threshold: Return neurons above this threshold (optional)

        Returns:
            Dictionary mapping layer names to lists of at-risk neuron indices
        """
        if threshold is None:
            threshold = self.prediction_threshold

        at_risk = {}

        for name, predictions in self.death_predictions.items():
            scores = predictions.numpy()

            if top_k is not None:
                # Get top-k highest scores
                top_indices = np.argsort(scores)[-top_k:][::-1]
                at_risk[name] = top_indices.tolist()
            else:
                # Get all above threshold
                at_risk_indices = np.where(scores > threshold)[0]
                at_risk[name] = at_risk_indices.tolist()

        return at_risk

    def get_statistics(self) -> Dict:
        """
        Get statistics about predictions.

        Returns:
            Dictionary with prediction statistics
        """
        stats = {}

        for name, predictions in self.death_predictions.items():
            stats[f'{name}_avg_death_score'] = predictions.mean().item()
            stats[f'{name}_max_death_score'] = predictions.max().item()
            stats[f'{name}_at_risk_count'] = (predictions > self.prediction_threshold).sum().item()

        return stats

    def visualize_neuron_health(self, layer_name: str, unit_idx: int) -> Dict:
        """
        Get detailed health information for a specific neuron.

        Args:
            layer_name: Name of the layer
            unit_idx: Index of the neuron

        Returns:
            Dictionary with health metrics
        """
        return {
            'activation_history': list(self.activation_history[layer_name][unit_idx]),
            'gradient_history': list(self.gradient_history[layer_name][unit_idx]),
            'utility_history': list(self.utility_history[layer_name][unit_idx]),
            'death_score': self.death_predictions[layer_name][unit_idx].item(),
            'current_activation': list(self.activation_history[layer_name][unit_idx])[-1]
            if self.activation_history[layer_name][unit_idx] else 0.0,
        }


class PreventiveReinitialization:
    """
    Preventive reinitialization strategy based on death predictions.

    Instead of waiting for neurons to die, we reinitialize them when
    we predict they're about to die, potentially maintaining more plasticity.
    """

    def __init__(self,
                 model: nn.Module,
                 predictor: NeuronDeathPredictor,
                 intervention_threshold: float = 0.8,
                 init_method: str = 'kaiming'):
        """
        Initialize preventive reinitialization.

        Args:
            model: PyTorch model
            predictor: NeuronDeathPredictor instance
            intervention_threshold: Death score threshold for intervention
            init_method: Initialization method
        """
        self.model = model
        self.predictor = predictor
        self.intervention_threshold = intervention_threshold
        self.init_method = init_method

        # Track interventions
        self.intervention_count = 0
        self.interventions_by_layer = {}

    def apply_preventive_reinitialization(self) -> Dict[str, int]:
        """
        Apply preventive reinitialization to at-risk neurons.

        Returns:
            Dictionary mapping layer names to number of neurons reinitialized
        """
        at_risk = self.predictor.get_at_risk_neurons(
            threshold=self.intervention_threshold
        )

        reinitialized = {}

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in at_risk:
                indices = at_risk[name]

                if len(indices) > 0:
                    self._reinitialize_neurons(module, indices, name)
                    reinitialized[name] = len(indices)
                    self.intervention_count += len(indices)

                    if name not in self.interventions_by_layer:
                        self.interventions_by_layer[name] = 0
                    self.interventions_by_layer[name] += len(indices)

        return reinitialized

    def _reinitialize_neurons(self,
                             module: nn.Linear,
                             indices: List[int],
                             layer_name: str):
        """
        Reinitialize specific neurons in a layer.

        Args:
            module: Linear module
            indices: List of neuron indices to reinitialize
            layer_name: Name of the layer
        """
        import math

        with torch.no_grad():
            in_features = module.in_features
            out_features = module.out_features

            for idx in indices:
                if self.init_method == 'kaiming':
                    std = math.sqrt(2.0 / in_features)
                    module.weight.data[idx] = torch.randn(in_features) * std
                elif self.init_method == 'xavier':
                    std = math.sqrt(2.0 / (in_features + out_features))
                    module.weight.data[idx] = torch.randn(in_features) * std

                if module.bias is not None:
                    module.bias.data[idx] = 0

                # Zero outgoing weights (to next layer)
                self._zero_outgoing_weights(layer_name, idx)

    def _zero_outgoing_weights(self, layer_name: str, unit_idx: int):
        """Zero outgoing weights from a neuron."""
        layer_names = [name for name, module in self.model.named_modules()
                      if isinstance(module, nn.Linear)]

        if layer_name in layer_names:
            layer_idx = layer_names.index(layer_name)
            if layer_idx + 1 < len(layer_names):
                next_layer_name = layer_names[layer_idx + 1]
                next_module = dict(self.model.named_modules())[next_layer_name]

                with torch.no_grad():
                    next_module.weight.data[:, unit_idx] = 0

    def get_statistics(self) -> Dict:
        """Get statistics about interventions."""
        return {
            'total_interventions': self.intervention_count,
            'interventions_by_layer': self.interventions_by_layer.copy()
        }
