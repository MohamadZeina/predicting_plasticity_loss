# Research Plan: Predicting Plasticity Loss

## Research Question

**Can we predict which neurons will become inactive (dead) in deep neural networks during continual learning, and does early intervention preserve plasticity better than reactive reinitialization?**

## Background

The Nature paper by Dohare et al. (2024) demonstrates that:
1. Deep networks lose plasticity over time in continual learning
2. This is caused by dead neurons, growing weights, and decreasing rank
3. Continual Backpropagation (reactive reinitialization) helps

Our hypothesis: **Predictive intervention should outperform reactive intervention**

## Experimental Design

### Phase 1: Minimal Reproduction ✅

**Goal**: Verify plasticity loss occurs in our setup

**Setup**:
- Dataset: Permuted MNIST (20 tasks)
- Network: Small MLP (256-256 hidden units)
- Baseline: Vanilla SGD

**Expected outcome**:
- Accuracy drops across tasks
- Dead units increase
- Weight magnitude grows
- Effective rank decreases

**Status**: Implementation complete, ready to run

### Phase 2: Prediction System Validation ⏳

**Goal**: Verify we can predict neuron death

**Metrics**:
1. **Prediction accuracy**: Of neurons we predict will die, what % actually die?
2. **False positive rate**: What % of predicted deaths don't occur?
3. **Lead time**: How many steps before death do we predict correctly?
4. **Coverage**: What % of actual deaths did we predict?

**Analysis**:
```python
# For each neuron that dies (activation → 0):
death_step = find_first_zero_activation(neuron)

# Look back in history
for step in range(death_step - 50, death_step):
    if prediction[step] == "will die":
        lead_time = death_step - step
        record_successful_prediction(lead_time)
```

**Success criteria**:
- Predict ≥70% of neuron deaths
- Lead time ≥10 steps
- False positive rate <30%

### Phase 3: Method Comparison ⏳

**Goal**: Compare all four methods

**Methods**:
1. Vanilla SGD (baseline)
2. L2 Regularization
3. Continual Backpropagation (reactive)
4. Preventive Reinitialization (predictive)

**Primary metrics**:
- Test accuracy over tasks (plasticity indicator)
- Dead unit percentage
- Weight magnitude
- Effective rank

**Secondary metrics** (for preventive vs reactive):
- Timing of interventions
- Total number of interventions
- Death scores at intervention time
- Neurons saved vs neurons lost

**Expected results**:
```
Vanilla:    Accuracy: 90% → 70%  Dead: 0% → 40%
L2:         Accuracy: 90% → 80%  Dead: 0% → 25%
Reactive:   Accuracy: 90% → 88%  Dead: 0% → 5%
Preventive: Accuracy: 90% → 90%  Dead: 0% → 2%  (hypothesis)
```

### Phase 4: Signal Analysis ⏳

**Goal**: Understand which signals are most predictive

**Approach**: Train a logistic regression to predict neuron death:
```python
# Features
features = [
    activation_trend,
    activation_magnitude,
    gradient_trend,
    gradient_magnitude,
    utility_trend,
    utility_magnitude,
    weight_norm,
    age
]

# Target
target = will_die_in_next_k_steps

# Analyze feature importance
model = LogisticRegression()
model.fit(features, target)
print(model.coef_)  # Which features matter most?
```

**Questions**:
1. Is activation trend or magnitude more predictive?
2. Do gradients add information beyond activations?
3. Is utility trend sufficient, or do we need all signals?
4. Can we simplify to 1-2 signals without losing accuracy?

### Phase 5: Threshold Optimization ⏳

**Goal**: Find optimal intervention threshold

**Approach**: Grid search over thresholds
```python
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

for threshold in thresholds:
    results = run_experiment(intervention_threshold=threshold)
    track_metrics(results)

# Find threshold that maximizes plasticity with minimal interventions
optimal = argmax(accuracy / interventions)
```

**Trade-off**:
- Low threshold (0.5): Many interventions, may disrupt learning
- High threshold (0.9): Few interventions, may miss neurons

## Hypotheses

### H1: Predictability
**We can predict neuron death with ≥70% accuracy, 10+ steps in advance**

Rationale: Neurons don't die suddenly - there's a gradual decline visible in activation/gradient trends

### H2: Performance
**Preventive intervention maintains higher test accuracy than reactive**

Rationale: Early intervention preserves partial neuron functionality, while reactive waits until neuron is nearly dead

### H3: Efficiency
**Preventive uses similar or fewer total reinitializations**

Rationale: By saving neurons earlier, we prevent cascading failures where one dead neuron leads to others dying

### H4: Timing
**Preventive intervenes at higher death scores (earlier) than reactive**

Measurement:
```python
# Reactive: intervenes when utility ≈ 0 (death score ≈ 1.0)
# Preventive: intervenes when death score ≈ 0.7

average_death_score_at_intervention[reactive] ≈ 0.95
average_death_score_at_intervention[preventive] ≈ 0.75
```

### H5: Signal Importance (ordered by predicted importance)
1. **Activation trend** - Most direct signal of dying
2. **Utility trend** - Captures importance decline
3. **Gradient magnitude** - Shows learning capacity
4. **Activation magnitude** - Current state
5. **Gradient trend** - Secondary indicator

## Implementation Validation

### Unit Tests
- [x] Metrics computation (dead units, weight mag, rank)
- [x] Contribution utility calculation
- [x] Death score computation
- [x] Trend analysis (linear regression)
- [x] Reinitalization logic
- [ ] End-to-end integration test

### Sanity Checks
```python
# 1. Dead units increase over time (vanilla)
assert dead_units[task_20] > dead_units[task_1]

# 2. Intervention reduces dead units
assert dead_units[with_intervention] < dead_units[without_intervention]

# 3. Death scores correlate with actual death
correlation(death_scores, actually_died) > 0.5

# 4. Reinitalized neurons become active
assert activation[after_reinit] > activation[before_reinit]
```

## Expected Challenges

### Challenge 1: Noisy Signals
**Problem**: Short-term fluctuations in activation/gradients

**Solution**: Use running averages and longer history windows

### Challenge 2: False Positives
**Problem**: Predicting death for neurons that recover

**Solution**: Require consistent downward trend, not just low values

### Challenge 3: Hyperparameter Sensitivity
**Problem**: Performance depends heavily on threshold choice

**Solution**: Grid search and validation set tuning

### Challenge 4: Computational Cost
**Problem**: Tracking history for every neuron is expensive

**Solution**:
- Subsample neurons for detailed tracking
- Use efficient circular buffers
- Track only recent history (last 100 steps)

## Success Metrics

### Minimum Viable Success
- ✅ Reproduce plasticity loss (accuracy drop >10%)
- ✅ Predict neuron death (accuracy >60%)
- ✅ Preventive outperforms vanilla (accuracy +5%)

### Strong Success
- Predict neuron death (accuracy >70%, lead time >10)
- Preventive matches or beats reactive
- Identify 1-2 key predictive signals
- Published results with clear insights

### Ideal Success
- Preventive significantly outperforms reactive (+5% accuracy)
- Generalize to other datasets/architectures
- Simple predictive model (1-2 signals)
- Clear understanding of why prediction helps

## Timeline

### Week 1: Setup & Validation ⏳
- [x] Implement all components
- [x] Write documentation
- [ ] Run quick tests
- [ ] Verify metrics are correct
- [ ] Debug any issues

### Week 2: Experiments
- [ ] Run Phase 1 (baseline reproduction)
- [ ] Run Phase 2 (prediction validation)
- [ ] Run Phase 3 (method comparison)
- [ ] Initial analysis

### Week 3: Analysis & Iteration
- [ ] Phase 4 (signal analysis)
- [ ] Phase 5 (threshold optimization)
- [ ] Write up results
- [ ] Create visualizations

## Deliverables

1. **Code**: Clean, documented, reproducible
2. **Results**: JSON files with all metrics
3. **Visualizations**: Plots comparing methods
4. **Analysis**: Which signals matter, optimal thresholds
5. **Documentation**: How to run, interpret results

## Open Questions

1. **Generalization**: Does this work on ImageNet/CIFAR?
2. **Architecture**: Does it work with ResNets, Transformers?
3. **Task similarity**: Does it matter how similar tasks are?
4. **Scale**: Does it work with larger networks (1000+ units per layer)?
5. **Online learning**: Can we update predictions efficiently online?

## Future Directions

### Short-term
1. Implement adaptive thresholds (per-layer or per-neuron)
2. Test on other datasets (CIFAR-100, ImageNet subset)
3. Compare with other plasticity methods (ReDo, L2, Dropout)

### Medium-term
1. Combine with forgetting prevention
2. Learn optimal reinitialization strategies
3. Meta-learning for task-specific predictors

### Long-term
1. Theoretical analysis of why prediction helps
2. Biological plausibility (compare with neuroscience)
3. Apply to large language models
4. Continual learning without task boundaries

## Key Insights to Extract

From successful experiments, we want to understand:

1. **When do neurons die?** Early tasks, late tasks, or continuously?

2. **Why do neurons die?** Low gradients, low activation, or both?

3. **Can death be prevented?** Or just delayed?

4. **What's the cost?** Do interventions disrupt learning?

5. **Which signal matters most?** Can we simplify prediction?

## Evaluation Criteria

### Scientific Merit
- Does it address an important problem? ✅
- Is the approach novel? ✅
- Are results reproducible? ⏳
- Do we gain new insights? ⏳

### Technical Quality
- Is implementation correct? ⏳
- Are comparisons fair? ⏳
- Are metrics appropriate? ✅
- Is analysis rigorous? ⏳

### Impact Potential
- Could this scale to real applications? ⏳
- Could others build on this? ✅
- Does it open new research directions? ✅

---

**Status**: Ready to run experiments
**Next step**: Execute quick_test.py, then permuted_mnist.py
**Timeline**: 1-2 weeks for complete analysis
