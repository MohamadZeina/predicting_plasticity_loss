# Predicting and Preventing Plasticity Loss in Deep Learning

This repository contains experiments to reproduce and extend the findings from the Nature paper "Loss of plasticity in deep continual learning" by Dohare et al. (2024).

## 🎯 Key Innovation: Predictive Neuron Death Prevention

While the original paper introduces **Continual Backpropagation** (reactive reinitialization after neurons die), our work introduces **Preventive Reinitialization** - predicting which neurons will become inactive and intervening *before* they die.

### Why This Matters

Dead neurons are irreversibly lost in standard networks. By the time we detect them (activation = 0 for all inputs), they've already stopped contributing. Our approach:

1. **Predicts** neuron death before it happens using multiple signals
2. **Prevents** loss of plasticity by intervening early
3. **Maintains** network capacity more effectively

## 📊 Experiment Overview

### Minimal Reproduction

We implement a minimal version of the paper's experiments using Permuted MNIST:

- **Problem**: Sequence of 20 permuted MNIST tasks
- **Network**: Small MLP (256-256 hidden units)
- **Metrics**:
  - Test accuracy (plasticity indicator)
  - Dead units percentage
  - Weight magnitude
  - Effective rank of representations

### Methods Compared

1. **Vanilla SGD**: Standard backpropagation (baseline)
2. **L2 Regularization**: Weight decay to prevent magnitude growth
3. **Continual Backpropagation**: Reactive reinitialization (from paper)
4. **Preventive Reinitialization**: Our novel predictive approach

## 🔮 Neuron Death Prediction System

Our prediction system uses multiple signals to forecast neuron death:

### Predictive Signals

1. **Activation Trend**: Is activation magnitude decreasing over time?
   ```python
   # Linear regression slope of activation history
   # Negative slope → likely to die
   ```

2. **Gradient Flow**: Are gradients becoming vanishingly small?
   ```python
   # Track gradient magnitude trend
   # Decreasing gradients → neuron not learning
   ```

3. **Contribution Utility**: Is the neuron's contribution to downstream layers decreasing?
   ```python
   # From paper: u[i] = |h[i]| * Σ|w[i,k]|
   # Decreasing utility → low importance
   ```

4. **Current State**: Is current activation already very low?
   ```python
   # Current activation magnitude
   # Low magnitude → close to death
   ```

### Death Score Computation

Each neuron gets a death score (0-1) combining all signals:

```python
death_score = average([
    activation_trend_score,    # Trend analysis
    activation_magnitude_score, # Current state
    gradient_trend_score,       # Learning capacity
    gradient_magnitude_score,   # Current gradients
    utility_trend_score         # Importance trend
])
```

### Intervention Strategy

When `death_score > threshold` (default 0.7):
1. Reinitialize incoming weights from original distribution
2. Zero outgoing weights (preserve learned function)
3. Reset tracking statistics

## 🏗️ Project Structure

```
predicting_plasticity_loss/
├── src/
│   ├── metrics.py                    # Plasticity metrics tracking
│   ├── continual_backprop.py         # Continual backpropagation implementation
│   ├── neuron_death_predictor.py     # Novel prediction system
│   └── visualize.py                  # Visualization utilities
├── experiments/
│   └── permuted_mnist.py             # Main experiment script
├── results/                          # Experiment outputs
├── notebooks/                        # Analysis notebooks
└── data/                            # Dataset cache

```

## 🚀 Running Experiments

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run minimal experiment (20 tasks, small network)
cd experiments
python permuted_mnist.py
```

### Configuration

Edit `permuted_mnist.py` to customize:

```python
config = {
    'num_samples': 10000,              # Dataset size
    'num_tasks': 20,                   # Number of permuted tasks
    'hidden_sizes': [256, 256],        # Network architecture
    'lr': 0.01,                        # Learning rate
    'replacement_rate': 1e-5,          # Continual BP rate
    'intervention_threshold': 0.7,     # Prediction threshold
    'methods': [                       # Methods to compare
        'vanilla',
        'l2',
        'continual_backprop',
        'preventive'
    ]
}
```

### Analyze Results

```bash
# Visualize results
python src/visualize.py results/permuted_mnist_<timestamp>.json
```

## 📈 Expected Results

Based on the paper and our extensions:

### Vanilla SGD (Baseline)
- ❌ Accuracy drops significantly (e.g., 90% → 70%)
- ❌ Dead units increase to 20-40%
- ❌ Weight magnitude grows continuously
- ❌ Effective rank decreases

### L2 Regularization
- ⚠️ Reduced accuracy drop (90% → 80%)
- ⚠️ Controlled weight magnitude
- ⚠️ Still has dead units (15-25%)

### Continual Backpropagation (Paper)
- ✅ Stable accuracy (~90%)
- ✅ Few dead units (<5%)
- ✅ Stable weight magnitude
- ✅ High effective rank maintained

### Preventive Reinitialization (Ours)
- ✅✅ Potentially better accuracy
- ✅✅ Preemptive neuron preservation
- ✅✅ Earlier intervention = less damage
- 📊 Hypothesis: Should outperform reactive methods

## 🔬 Key Hypotheses to Test

1. **Early Detection**: Can we predict neuron death 10-50 steps before it occurs?

2. **Better Preservation**: Does preventive intervention maintain more plasticity than reactive?

3. **Optimal Threshold**: What death score threshold minimizes interventions while maximizing plasticity?

4. **Signal Importance**: Which predictive signal (activation, gradient, utility) is most informative?

## 📝 Implementation Details

### Contribution Utility (Equation 1 from paper)

```python
def update_utility(h, w, u_prev, eta=0.99):
    """
    h: hidden unit activations (batch_size, num_units)
    w: outgoing weights (out_features, in_features)
    u_prev: previous utility
    eta: decay rate
    """
    # Instantaneous contribution
    contribution = |h| * Σ|w_ik|

    # Running average
    u_new = eta * u_prev + (1 - eta) * contribution

    return u_new
```

### Preventive Reinitialization

```python
def preventive_reinit(neuron, death_score, threshold=0.7):
    """
    Reinitialize neuron if predicted to die soon.
    """
    if death_score > threshold:
        # Reinitialize incoming weights
        neuron.weight = sample_from_init_distribution()

        # Zero outgoing weights (preserve function)
        next_layer.weight[:, neuron_idx] = 0

        # Reset tracking
        neuron.age = 0
        neuron.utility = 0
```

## 🎓 Key Insights from Paper

1. **Standard deep learning loses plasticity**: Networks gradually lose the ability to learn new tasks

2. **Three main correlates**:
   - Increasing dead units
   - Growing weight magnitudes
   - Decreasing effective rank

3. **Root cause**: Networks move away from the beneficial properties of random initialization

4. **Solution requires**:
   - Small weights (L2 regularization)
   - Continued diversity (randomness injection)
   - Selective intervention (don't disrupt good neurons)

## 🔮 Future Directions

1. **Adaptive Thresholds**: Learn optimal intervention threshold per layer

2. **Multi-task Prediction**: Predict neuron usefulness for future tasks

3. **Causality Analysis**: Determine causal relationships between metrics

4. **Scaling Studies**: Test on larger networks and more complex tasks

5. **Neuroscience Connections**: Compare with biological neuron turnover

## 📚 References

```bibtex
@article{dohare2024loss,
  title={Loss of plasticity in deep continual learning},
  author={Dohare, Shibhansh and Hernandez-Garcia, J Fernando and Lan, Qingfeng and Rahman, Parash and Mahmood, A Rupam and Sutton, Richard S},
  journal={Nature},
  volume={632},
  pages={768--774},
  year={2024},
  publisher={Nature Publishing Group}
}
```

## 🤝 Contributing

This is an experimental research repository. Feel free to:
- Open issues for bugs or questions
- Submit PRs with improvements
- Share your results and findings

## 📧 Contact

For questions about the experiments or neuron death prediction, please open an issue.

---

**Note**: This is a minimal reproduction for fast iteration. For full-scale experiments matching the paper, significantly more computation is required.
