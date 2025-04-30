# 🧠 Explainers

One goal of **EvalxNLP** is to enable users to generate diverse explanations using multiple explainability methods. It integrates eight widely recognized **post-hoc feature attribution (Ph-FA)** explainability methods from the XAI literature.

These methods assign importance scores to individual features based on their contribution to the model's prediction. We focus on Ph-FA methods for their interpretability, relevance to end users, and generally efficient computational cost.

---

## 🧩 Categories of Explainers

Feature attribution methods in EvalxNLP are categorized into:

- **Perturbation-based methods**
- **Gradient-based methods**

Below, we provide an overview of the methods included in each category.

---

## 🔀 Perturbation-Based Methods

Perturbation-based methods explain model predictions by altering or masking input features and observing the resulting changes in the output.

### LIME

> A model-agnostic method that explains individual predictions by training a local surrogate model.

LIME generates explanations by perturbing the input data and measuring how the model's predictions change. Perturbed samples are weighted by their similarity to the original input. A simple, interpretable model—typically linear—is then trained on this weighted data to approximate the local behavior of the original model. LIME aims to balance fidelity (how closely the surrogate approximates the model) and interpretability (how understandable the explanation is to humans).

### SHAP (Partition SHAP)

> A game-theoretic approach to explain feature contributions using Shapley values.

SHAP computes the contribution of each input feature by evaluating all possible subsets of features, as proposed in cooperative game theory. Partition SHAP, a variant, improves efficiency by leveraging feature independence, making it scalable for high-dimensional data while preserving theoretical guarantees.

### SHAP-I

> An advanced extension of SHAP that captures higher-order feature interactions.

SHAP-I extends the original SHAP method to support any-order interactions between features. It incorporates efficient algorithms to compute interaction-aware Shapley values, enabling explanation of complex dependencies in high-dimensional input data.

---

## 🔁 Gradient-Based Methods

Gradient-based methods derive feature attributions by analyzing the gradients of the model's output with respect to its inputs.

### Saliency

> A baseline technique that visualizes input feature importance using raw gradients.

Saliency methods compute the gradient of the output with respect to each input feature. The absolute values of these gradients highlight which parts of the input most influence the model's prediction.

### GxI (Gradient × Input)

> Enhances gradient information by incorporating input values.

GxI multiplies the gradient of the output with respect to each input feature by the corresponding input value. This combination provides a more intuitive measure of each feature's contribution, especially for models where feature magnitudes play a critical role.

### IG (Integrated Gradients)

> A principled method that attributes importance by integrating gradients along a path from baseline to input.

Integrated Gradients explain a model’s prediction by accumulating gradients as the input transitions from a baseline (e.g., a zero vector) to the actual input. This method satisfies desirable properties such as completeness and provides a mathematically grounded attribution of feature importance.

### DL (DeepLIFT)

> Assigns attributions by comparing neuron activations to a reference.

DeepLIFT computes the difference between the activations of neurons for an input and a reference baseline. It assigns credit to each input feature for the change in output, using the Rescale Rule. DeepLIFT is often more stable than raw gradients and is computationally more efficient than IG, especially for large datasets and deep architectures.

### GBP (Guided Backpropagation)

> Enhances interpretability by filtering gradients.

Guided Backpropagation modifies the standard backpropagation algorithm to ignore negative gradients during the backward pass. This approach emphasizes input features that positively influence the output, resulting in sharper and more focused explanations.

---1~# 🧠 Explainers

One goal of **EvalxNLP** is to enable users to generate diverse explanations using multiple explainability methods. It integrates eight widely recognized **post-hoc feature attribution (Ph-FA)** explainability methods from the XAI literature.

These methods assign importance scores to individual features based on their contribution to the model's prediction. We focus on Ph-FA methods for their interpretability, relevance to end users, and generally efficient computational cost.

---

## 🧩 Categories of Explainers

Feature attribution methods in EvalxNLP are categorized into:

- **Perturbation-based methods**
- **Gradient-based methods**

Below, we provide an overview of the methods included in each category.

---

## 🔀 Perturbation-Based Methods

Perturbation-based methods explain model predictions by altering or masking input features and observing the resulting changes in the output.

### LIME

> A model-agnostic method that explains individual predictions by training a local surrogate model.

LIME generates explanations by perturbing the input data and measuring how the model's predictions change. Perturbed samples are weighted by their similarity to the original input. A simple, interpretable model—typically linear—is then trained on this weighted data to approximate the local behavior of the original model. LIME aims to balance fidelity (how closely the surrogate approximates the model) and interpretability (how understandable the explanation is to humans).

### SHAP (Partition SHAP)

> A game-theoretic approach to explain feature contributions using Shapley values.

SHAP computes the contribution of each input feature by evaluating all possible subsets of features, as proposed in cooperative game theory. Partition SHAP, a variant, improves efficiency by leveraging feature independence, making it scalable for high-dimensional data while preserving theoretical guarantees.

### SHAP-I

> An advanced extension of SHAP that captures higher-order feature interactions.

SHAP-I extends the original SHAP method to support any-order interactions between features. It incorporates efficient algorithms to compute interaction-aware Shapley values, enabling explanation of complex dependencies in high-dimensional input data.

---

## 🔁 Gradient-Based Methods

Gradient-based methods derive feature attributions by analyzing the gradients of the model's output with respect to its inputs.

### Saliency

> A baseline technique that visualizes input feature importance using raw gradients.

Saliency methods compute the gradient of the output with respect to each input feature. The absolute values of these gradients highlight which parts of the input most influence the model's prediction.

### GxI (Gradient × Input)

> Enhances gradient information by incorporating input values.

GxI multiplies the gradient of the output with respect to each input feature by the corresponding input value. This combination provides a more intuitive measure of each feature's contribution, especially for models where feature magnitudes play a critical role.

### IG (Integrated Gradients)

> A principled method that attributes importance by integrating gradients along a path from baseline to input.

Integrated Gradients explain a model’s prediction by accumulating gradients as the input transitions from a baseline (e.g., a zero vector) to the actual input. This method satisfies desirable properties such as completeness and provides a mathematically grounded attribution of feature importance.

### DL (DeepLIFT)

> Assigns attributions by comparing neuron activations to a reference.

DeepLIFT computes the difference between the activations of neurons for an input and a reference baseline. It assigns credit to each input feature for the change in output, using the Rescale Rule. DeepLIFT is often more stable than raw gradients and is computationally more efficient than IG, especially for large datasets and deep architectures.

### GBP (Guided Backpropagation)

> Enhances interpretability by filtering gradients.

Guided Backpropagation modifies the standard backpropagation algorithm to ignore negative gradients during the backward pass. This approach emphasizes input features that positively influence the output, resulting in sharper and more focused explanations.

---
