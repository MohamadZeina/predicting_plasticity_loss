# Understanding Plasticity Loss and Prediction

## What is Plasticity Loss?

In continual learning, **plasticity** refers to a neural network's ability to learn new things. **Plasticity loss** occurs when a network gradually loses this ability over time, even though the task difficulty remains constant.

### The Problem

```
Task 1: Network learns well (90% accuracy)
Task 2: Network learns well (89% accuracy)
Task 3: Network learns okay (85% accuracy)
...
Task 20: Network barely learns (65% accuracy)
```

The network hasn't forgotten Task 1-19 (that's a different problem). It just can't learn Task 20 as well anymore.

## Why Does This Happen?

The paper identifies three main correlates:

### 1. Dead Neurons

```python
# A neuron is "dead" if its output is always zero
for all_inputs in dataset:
    if neuron_output == 0:  # ReLU units stuck at zero
        # This neuron contributes NOTHING
        # Its weights can't be updated (gradient = 0)
```

**Why it matters**: Each dead neuron is permanently lost capacity.

### 2. Growing Weight Magnitudes

```python
# Over time, weights grow larger
Task 1: avg_weight = 0.1
Task 10: avg_weight = 1.5
Task 20: avg_weight = 3.8
```

**Why it matters**: Large weights lead to ill-conditioned optimization (slower learning).

### 3. Decreasing Effective Rank

```python
# Effective rank measures how diverse the hidden representations are
# Low rank = many neurons are redundant

Task 1: effective_rank = 200 / 256 (diverse!)
Task 20: effective_rank = 50 / 256  (redundant!)
```

**Why it matters**: Redundant neurons don't add new representational power.

## The Key Insight

All three problems stem from **moving away from initialization**:

```python
# At initialization:
- Weights are small random numbers ✓
- All neurons are active ✓
- Representations are diverse ✓

# After training on many tasks:
- Weights are large, optimized numbers ✗
- Many neurons are dead ✗
- Representations are low-rank ✗
```

## Solution 1: Continual Backpropagation (From Paper)

**Idea**: Selectively reinitialize low-utility neurons during training.

```python
for each training step:
    # 1. Compute contribution utility for each neuron
    utility[i] = |activation[i]| × Σ|outgoing_weights[i,k]|

    # 2. Find lowest-utility mature neurons
    if neuron.age > maturity_threshold:
        lowest_utility_neurons = find_bottom_k(utility)

    # 3. Reinitialize them
    for neuron in lowest_utility_neurons:
        neuron.incoming_weights = sample_from_init_distribution()
        neuron.outgoing_weights = 0  # Don't disrupt learned function!
```

**Result**: Maintains plasticity by continually refreshing the network with random neurons.

## Solution 2: Preventive Reinitialization (Our Contribution)

**Problem with reactive approach**: By the time utility is low, the neuron has already stopped contributing.

**Our idea**: Predict which neurons will die and intervene early.

### Prediction Signals

#### Signal 1: Activation Trend

```python
# Track activation magnitude over time
activation_history = [0.5, 0.4, 0.35, 0.28, 0.20, ...]

# Compute trend (linear regression)
slope = fit_line(activation_history)

if slope < 0:  # Decreasing
    death_risk += 1
```

**Interpretation**: A neuron whose activations are steadily decreasing is on a path to zero.

#### Signal 2: Gradient Magnitude

```python
# Track gradients flowing through the neuron
gradient_history = [0.01, 0.008, 0.005, 0.003, ...]

if gradient_magnitude < threshold:
    death_risk += 1  # Not learning anymore
```

**Interpretation**: Small gradients mean the neuron isn't being updated.

#### Signal 3: Contribution Utility Trend

```python
# Track utility over time
utility_history = [0.8, 0.7, 0.55, 0.40, ...]

if utility_trend < 0:  # Decreasing
    death_risk += 1
```

**Interpretation**: Decreasing utility means becoming less important.

#### Signal 4: Current State

```python
if current_activation < 0.1:
    death_risk += 1  # Already very low
```

**Interpretation**: If already nearly dead, high risk.

### Death Score Computation

```python
def compute_death_score(neuron):
    scores = []

    # Trend scores (0 = improving, 1 = dying)
    scores.append(trend_score(neuron.activation_history))
    scores.append(trend_score(neuron.gradient_history))
    scores.append(trend_score(neuron.utility_history))

    # Magnitude scores
    scores.append(magnitude_score(neuron.current_activation))
    scores.append(magnitude_score(neuron.current_gradient))

    # Average all signals
    death_score = mean(scores)  # Range: 0 (healthy) to 1 (dying)

    return death_score
```

### Intervention Strategy

```python
for each neuron:
    death_score = compute_death_score(neuron)

    if death_score > 0.7:  # Predicted to die soon
        # INTERVENE BEFORE IT DIES
        reinitialize(neuron)
        print(f"Saved neuron {neuron.id} at death_score={death_score}")
```

## Why Prediction Might Be Better

### Reactive (Continual Backpropagation)

```
Neuron utility: 1.0 → 0.8 → 0.5 → 0.2 → 0.05 → REINITIALIZE
                                              ↑
                                         Already mostly dead
```

### Preventive (Our Approach)

```
Neuron activity: 1.0 → 0.8 → 0.6 → 0.45 → PREDICT DEATH → REINITIALIZE
                                          ↑
                                     Still partially active
                                     Can salvage learning
```

## Hypothesis to Test

1. **Early detection**: Can we reliably predict death 10-50 steps before utility reaches zero?

2. **Better preservation**: Does early intervention maintain more plasticity?

3. **Efficiency**: Do we need fewer total reinitializations?

4. **Accuracy**: Does it lead to better task performance?

## Example: A Neuron's Life Story

### Without Intervention (Dies)

```
Step 0:   activation=0.8, gradient=0.02, utility=1.2  [Healthy]
Step 50:  activation=0.6, gradient=0.015, utility=0.9 [Still good]
Step 100: activation=0.4, gradient=0.008, utility=0.5 [Declining]
Step 150: activation=0.2, gradient=0.002, utility=0.2 [Struggling]
Step 200: activation=0.0, gradient=0.000, utility=0.0 [DEAD - Lost forever]
```

### With Continual Backpropagation (Reactive)

```
Step 0:   activation=0.8, gradient=0.02, utility=1.2  [Healthy]
Step 50:  activation=0.6, gradient=0.015, utility=0.9 [Still good]
Step 100: activation=0.4, gradient=0.008, utility=0.5 [Declining]
Step 150: activation=0.2, gradient=0.002, utility=0.2 [Struggling]
Step 180: utility=0.05 < threshold → REINITIALIZE     [Saved at last moment]
Step 200: activation=0.7, gradient=0.018, utility=1.0 [Reborn!]
```

### With Preventive Reinitialization (Proactive)

```
Step 0:   activation=0.8, gradient=0.02, utility=1.2, death_score=0.1 [Healthy]
Step 50:  activation=0.6, gradient=0.015, utility=0.9, death_score=0.3 [Still good]
Step 100: activation=0.4, gradient=0.008, utility=0.5, death_score=0.6 [Declining]
Step 120: death_score=0.75 > 0.7 → PREDICT DEATH → REINITIALIZE [Early save!]
Step 140: activation=0.8, gradient=0.02, utility=1.1, death_score=0.2 [Healthy again!]
```

**Key difference**: Intervention at step 120 vs 180 - 60 steps earlier!

## Mathematical Formulation

### Contribution Utility (From Paper)

For neuron $i$ in layer $l$:

$$u_l[i]_t = \eta \cdot u_l[i]_{t-1} + (1-\eta) \cdot \left(\sum_{k=1}^{n_{l+1}} |h_{l,i,t}| \cdot |w_{l,i,k,t}|\right)$$

Where:
- $h_{l,i,t}$ = activation of neuron $i$ at time $t$
- $w_{l,i,k,t}$ = weight from neuron $i$ to neuron $k$ in next layer
- $\eta$ = decay rate (typically 0.99)

### Death Score (Our Contribution)

$$\text{death\_score}_i = \frac{1}{5}\sum_{j=1}^{5} s_j$$

Where:
- $s_1 = \sigma(-\beta_{\text{act}})$ (activation trend, negative slope → high score)
- $s_2 = \frac{1}{1 + h_i}$ (current activation magnitude)
- $s_3 = \sigma(-\beta_{\text{grad}})$ (gradient trend)
- $s_4 = \frac{1}{1 + |\nabla_i|}$ (current gradient magnitude)
- $s_5 = \sigma(-\beta_{\text{util}})$ (utility trend)

Where $\sigma(x) = \frac{1}{1+e^{-x}}$ is the sigmoid function and $\beta$ represents slopes.

## Implementation Checklist

- [x] Track activation history per neuron
- [x] Track gradient history per neuron
- [x] Track utility history per neuron
- [x] Compute trend scores (linear regression)
- [x] Compute magnitude scores
- [x] Combine into death score
- [x] Threshold-based intervention
- [x] Reinitialize predicted-to-die neurons
- [ ] Compare with reactive approach
- [ ] Analyze which signal is most predictive
- [ ] Optimize intervention threshold

## Expected Outcomes

### Metrics to Track

1. **Test Accuracy Over Tasks**: Does plasticity loss occur?
2. **Dead Unit Percentage**: Do units die over time?
3. **Weight Magnitude**: Do weights grow uncontrollably?
4. **Effective Rank**: Does diversity decrease?
5. **Prediction Accuracy**: Do we predict correctly?
6. **Intervention Timing**: How early do we intervene?

### Success Criteria

Preventive approach is successful if:
- Maintains higher test accuracy than reactive
- Intervenes earlier (lower death scores at intervention)
- Uses similar or fewer total reinitializations
- Maintains higher effective rank
- Has fewer truly dead neurons

## Next Steps

1. ✅ Implement prediction system
2. ✅ Implement preventive reinitialization
3. ⏳ Run experiments comparing all methods
4. ⏳ Analyze which predictive signals are most important
5. ⏳ Tune intervention threshold
6. ⏳ Scale to larger networks and harder tasks
