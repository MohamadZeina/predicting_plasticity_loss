# Predicting and Preventing Plasticity Loss in Deep Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

This repository implements experiments to reproduce and extend findings from the Nature paper ["Loss of plasticity in deep continual learning"](https://doi.org/10.1038/s41586-024-07711-7) (Dohare et al., 2024).

**Key Innovation**: While the paper introduces reactive reinitialization (Continual Backpropagation), we introduce **Predictive Neuron Death Prevention** - forecasting which neurons will become inactive and intervening *before* they die.

## 🔬 The Problem: Plasticity Loss

Neural networks gradually lose their ability to learn when training continues across multiple tasks, even when task difficulty remains constant. This manifests as:

- **Dead neurons** (ReLU units stuck at zero)
- **Growing weight magnitudes** (ill-conditioned optimization)
- **Decreasing effective rank** (redundant representations)

## 💡 Our Solution: Predictive Intervention

Instead of waiting for neurons to die, we:

1. **Predict** neuron death using multiple signals (activation trends, gradients, utility)
2. **Intervene** early, before neurons become inactive
3. **Preserve** network capacity more effectively

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run quick test (verify implementation)
cd experiments
python quick_test.py

# Run minimal experiment (20 tasks, small network)
python permuted_mnist.py

# Visualize results
cd ..
python src/visualize.py results/permuted_mnist_<timestamp>.json
```

## 📊 What Gets Compared

- **Vanilla SGD**: Baseline (loses plasticity)
- **L2 Regularization**: Weight decay (helps but not enough)
- **Continual Backpropagation**: Reactive reinitialization (from paper)
- **Preventive Reinitialization**: Predictive intervention (ours)

## 📁 Repository Structure

```
predicting_plasticity_loss/
├── src/
│   ├── metrics.py                    # Track dead units, weight magnitude, effective rank
│   ├── continual_backprop.py         # Continual backpropagation (from paper)
│   ├── neuron_death_predictor.py     # Novel prediction system
│   └── visualize.py                  # Plotting and analysis
├── experiments/
│   ├── quick_test.py                 # Component tests
│   └── permuted_mnist.py             # Main experiment
├── notebooks/
│   └── understanding_plasticity_loss.md  # Detailed explanation
├── results/                          # Experiment outputs
└── EXPERIMENT_README.md              # Full documentation

```

## 🔮 How Prediction Works

Each neuron gets a **death score** (0-1) based on:

```python
death_score = average([
    activation_trend_score,     # Is activation decreasing?
    activation_magnitude_score, # Is current activation low?
    gradient_trend_score,       # Are gradients vanishing?
    gradient_magnitude_score,   # Is gradient currently tiny?
    utility_trend_score         # Is importance decreasing?
])

if death_score > 0.7:  # Predicted to die
    reinitialize_neuron()  # Intervene early!
```

## 📈 Expected Results

| Method | Accuracy Drop | Dead Units | Weight Growth |
|--------|---------------|------------|---------------|
| Vanilla | ❌ Large (90→70%) | ❌ High (40%) | ❌ Continuous |
| L2 Reg | ⚠️ Moderate (90→80%) | ⚠️ Medium (25%) | ✅ Controlled |
| Continual BP | ✅ Minimal (90→88%) | ✅ Low (5%) | ✅ Stable |
| **Preventive** | ✅✅ Minimal (target) | ✅✅ Very Low | ✅✅ Stable |

## 📚 Key Hypotheses

1. Can we predict neuron death 10-50 steps before it occurs?
2. Does preventive intervention maintain better plasticity than reactive?
3. Which predictive signal (activation, gradient, utility) is most informative?
4. What is the optimal intervention threshold?

## 🛠️ Implementation Details

See [`EXPERIMENT_README.md`](EXPERIMENT_README.md) for:
- Detailed architecture
- Mathematical formulations
- Contribution utility computation
- Death score calculation
- Preventive reinitialization strategy

See [`notebooks/understanding_plasticity_loss.md`](notebooks/understanding_plasticity_loss.md) for:
- Visual explanations
- Intuitive examples
- Neuron life stories
- Why prediction helps

## 📖 Citation

If you use this code or find the approach useful:

```bibtex
@article{dohare2024loss,
  title={Loss of plasticity in deep continual learning},
  author={Dohare, Shibhansh and Hernandez-Garcia, J Fernando and Lan, Qingfeng and Rahman, Parash and Mahmood, A Rupam and Sutton, Richard S},
  journal={Nature},
  volume={632},
  pages={768--774},
  year={2024}
}
```

## 🤝 Contributing

This is an experimental research repository. Contributions welcome:
- Bug reports and fixes
- New predictive signals
- Scaling experiments
- Alternative intervention strategies

## 📧 Questions?

Open an issue or check the documentation:
- [`EXPERIMENT_README.md`](EXPERIMENT_README.md) - Technical details
- [`notebooks/understanding_plasticity_loss.md`](notebooks/understanding_plasticity_loss.md) - Conceptual guide

---

**Note**: This is a minimal reproduction for fast iteration. Full-scale experiments require significantly more computation.