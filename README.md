
# 🧠 Computational Psychiatry with Reinforcement Learning

This repository implements reinforcement learning models to simulate and analyze behavioral patterns in PTSD vs. control groups during a probabilistic gain/loss bandit task. Inspired by Brown et al. (2018), the models capture cognitive differences using Q-learning and associability-modulated learning frameworks.

## 🔍 Problem Statement

How does PTSD affect reinforcement learning from gains and losses? Can computational models capture individual differences in learning behavior and decision-making under uncertainty?

## 🧰 Features

- ✅ Temporal Difference (Q-learning) model
- ✅ Associability-modulated RL model
- ✅ Model simulation and participant-level parameter fitting
- ✅ Negative log-likelihood evaluation and optimization
- ✅ AIC/BIC-based model selection and group comparison
- ✅ Parameter recovery and model recovery diagnostics
- ✅ Group-level statistical analysis (t-tests, Pearson correlation)
- ✅ Visualizations of learning dynamics and fitted parameters

## 📊 Key Equations

### Q-learning

\[
Q_C(t+1) = Q_C(t) + \alpha \cdot (R(t) - Q_C(t))
\]

### Associability-Modulated Learning

\[
\kappa_C(t+1) = (1 - \eta) \cdot \kappa_C(t) + \eta \cdot |\delta(t)| \\
Q_C(t+1) = Q_C(t) + \alpha \cdot \kappa_C(t) \cdot \delta(t)
\]


## 📈 Results Summary

| Model           | AIC (↓) | BIC (↓) | Captures Group Difference? |
|----------------|---------|---------|-----------------------------|
| Q-learning     | ~7100   | ~7600   | Partial                    |
| Associability  | ~6800   | ~7600   | Better loss asymmetry     |

- Associability model better fits PTSD participants based on AIC.
- Learning rate (\alpha) for losses tends to be higher in PTSD group.

## 📚 References

- Brown VM et al. (2018). *Associability-modulated loss learning in PTSD*, eLife.
- Sutton & Barto (2018). *Reinforcement Learning: An Introduction*.
- Boukezzi et al. (2020). *Altered reward processing in PTSD*, NeuroImage: Clinical.

## 🚀 Future Directions

- Integrate real behavioral datasets (e.g., clinical trial data)
- Extend to latent-state or model-based RL architectures
- Include uncertainty-aware Bayesian RL variants

---

**Author**: Trisha Prasad  
**University of Edinburgh – March 2025**
