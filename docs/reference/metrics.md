# 📊 Evaluation Metrics

EvalxNLP incorporates a diverse and well-recognized set of properties and metrics from prior research to evaluate post-hoc explanation methods. These metrics are designed to assess explanation quality across three major properties:

- **Faithfulness**
- **Plausibility**
- **Complexity**

Each metric is **custom-implemented**, following its original research paper to ensure correctness and theoretical fidelity.

> _(↓)/(↑) indicates whether lower or higher values are better for that metric._

---

## ✅ Faithfulness

> Measures how well the explanations align with the model's actual reasoning.

### Soft Sufficiency ↓

Measures how well the most important tokens (based on their importance scores) can retain the model's prediction when other tokens are softly perturbed. It assumes that retaining more elements of important tokens should preserve the model's output, while dropping less important tokens should have minimal impact.

A **Bernoulli mask** is generated, where each token is dropped with a probability proportional to its normalized importance score:

$$
\mathrm{mask} \sim \mathrm{Bernoulli}\left(\mathrm{normalized\_importance\_scores}\right)
$$

The **Suff score** is calculated as:

$$
\mathrm{Suff} = 1 - \max\left(0, P_{\mathrm{full}}(y) - P_{\mathrm{reduced}}(y)\right)
$$

Where:

- $P_{\mathrm{full}}(y)$ is the model's predicted probability for the original input
- $P_{\mathrm{reduced}}(y)$ is the model's predicted probability for the perturbed input

The Suff score is normalized to the range $[0, 1]$ using a baseline Suff score (computed by masking all tokens):

$$
\mathrm{normalized\_Suff} = \frac{\max\left(0, \mathrm{Suff} - \mathrm{baseline\_Suff}\right)}{1 - \mathrm{baseline\_Suff}}
$$

The final score is the average across all instances, with higher values indicating that the model's predictions are less affected by the perturbation of important tokens.

---

### Soft Comprehensiveness ↑

It evaluates how much the model's prediction changes when important tokens are softly perturbed using Bernoulli mask. It assumes that heavily perturbing important tokens should significantly affect the model's output, indicating their importance to the prediction.

For each instance, the importance scores of tokens are normalized to the range $[0, 1]$:

$$
\mathrm{normalized\_importance\_scores} = \frac{\mathrm{importance\_scores} - \min(\mathrm{importance\_scores})}{\max(\mathrm{importance\_scores}) - \min(\mathrm{importance\_scores})}
$$

A Bernoulli mask is then generated, where each token is dropped with a probability proportional to $1 - \mathrm{normalized\_importance\_scores}$:

$$
\mathrm{mask} \sim \mathrm{Bernoulli}\left(1 - \mathrm{normalized\_importance\_scores}\right)
$$

This mask is applied to the token embeddings, creating a perturbed input. The $\mathrm{Comp}$ score is calculated as the difference between the model's confidence in the original prediction and its confidence after perturbation:

$$
\mathrm{Comp} = \max\left(0, P_{\mathrm{original}}(y) - P_{\mathrm{perturbed}}(y)\right)
$$

where:

- $P_{\mathrm{original}}(y)$ is the model's predicted probability for the original input
- $P_{\mathrm{perturbed}}(y)$ is the model's predicted probability for the perturbed input

The final $\mathrm{Comp}$ score is the average across all instances, with higher values indicating that the model relies more heavily on the important tokens.

---

### FAD curve N-AUC ↓

Measures the impact of dropping the most salient tokens on model performance, with the steepness of the curve indicating the method's faithfulness. The N-AUC quantifies this steepness, where a higher score reflects better alignment of the attribution method with the model's true feature importance.

For each instance, tokens are progressively dropped based on their saliency scores and the model's accuracy is evaluated at different drop percentages (e.g., 0%, 10%, ..., 40%). The saliency scores determine which tokens to replace with a baseline token (`[MASK]`). The N-AUC is computed over a specified percentage range (0% to 20%) using the trapezoidal rule:

$$
\mathrm{N\text{-}AUC} = \frac{\mathrm{AUC}}{\mathrm{max\_AUC}}
$$

where:

- $\mathrm{AUC}$ is the area under the accuracy curve, computed as:

  $$
  \mathrm{AUC} = \int_{x_{\min}}^{x_{\max}} y(x) \, dx
  $$

  where $x$ represents the percentage of tokens dropped, and $y(x)$ represents the model's accuracy at that percentage.

- $\mathrm{max\_AUC}$ is the maximum possible area, calculated as:

  $$
  \mathrm{max\_AUC} = (x_{\max} - x_{\min}) \times \max(y)
  $$

  where $x_{\min}$ and $x_{\max}$ are the lower and upper bounds of the percentage range, and $\max(y)$ is the highest accuracy value in the range.

This metric quantifies how much model performance degrades when salient tokens are removed, with lower N-AUC indicating greater reliance on the dropped tokens.

---

### AUTPC ↓

Area Under the Token Perturbation Curve evaluates the faithfulness of saliency explanations by progressively masking the most important tokens (based on their saliency scores) and measuring the drop in the model's performance. Masking involves replacing tokens with a baseline value (typically `[MASK]` or a zero vector).

Starting with 0% masking and incrementally increasing to 100%, the model’s accuracy
is computed at each threshold. The resulting performance curve, plotting accuracy
against the percentage of tokens masked, is used to calculate the area under the curve
(AUC).

This AUTPC value, normalized to the range [0, 1], provides a single metric
summarizing how significantly the model relies on the highlighted tokens, with higher
values indicating more faithful explanations. A smaller AUC indicates that removing
critical features degrades performance faster, demonstrating more faithful explanations.
Results are standardized by the number of features for comparability.

---

## 💡 Plausibility

> Measures how well explanations align with human-annotated rationales.

### IOU-F1 Score ↑

It evaluates the alignment between predicted rationales and ground truth rationales at the span level using the Intersection over Union (IOU) metric. For the i-th instance with predicted rationale $S_i^p$ and ground truth rationale $S_i^g$, the IOU is calculated as:

$$
S_i = \frac{|S_i^p \cap S_i^g|}{|S_i^p \cup S_i^g|}
$$

A rationale is considered a match when $S_i \geq 0.5$. The IOU-F1 score aggregates these matches across all $N$ instances:

$$
\mathrm{IOU\text{-}F1} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(S_i \geq 0.5)
$$

where $\mathbb{I}$ is the indicator function:
$$
\mathbb{I}(S_i \geq 0.5) = 
\begin{cases} 
1 & \text{if } S_i \geq 0.5 \\
0 & \text{otherwise}
\end{cases}
$$


This metric ranges from 0 to 1, with higher values indicating better alignment between predicted and ground truth rationales. The 0.5 threshold ensures that only substantially overlapping spans are counted as matches.

---

### Token-level F1 Score ↑

The Token F1-score measures the alignment between predicted rationales and ground truth rationales by computing the F1-score, which balances precision and recall. For each instance, the predicted rationale $S_i^p$ is compared to the ground truth rationale $S_i^g$, and the F1-score is calculated based on the overlap of tokens.

The F1-score for the i-th instance is defined as:

$$
F1_i = \frac{2 \times P_i \times R_i}{P_i + R_i}
$$

where:
- **Precision** $P_i$:
  $$
  P_i = \frac{|S_i^p \cap S_i^g|}{|S_i^p|}
  $$
- **Recall** $R_i$:
  $$
  R_i = \frac{|S_i^p \cap S_i^g|}{|S_i^g|}
  $$

The final Token F1-score is the average F1-score across all $N$ instances:

$$
\mathrm{Token\text{-}F1} = \frac{1}{N} \sum_{i=1}^N F1_i
$$

For both the Token-F1 score and IOU-F1 score, the top $K$ tokens with positive influence are selected, where $K$ is the average length of the human rationale for the dataset.

---

### AUPRC ↑

For each instance, the saliency scores are compared to the ground truth rationale
mask. The precision-recall curve is computed, and then the area under this curve is
calculated. The final AUPRC score is the average of these values across all instances,
providing a single metric that quantifies the alignment between predicted saliency and
human-annotated rationales, with higher scores indicating better performance.

---

## 🔀 Complexity

> Measures how interpretable the explanation is by checking sparsity and token focus.

### Complexity ↓

It measures the complexity of token-level attributions using Shannon entropy, which quantifies how evenly the importance scores are distributed across tokens. For each instance, the fractional contribution of each token is computed as:

$$
f_j = \frac{|a_j|}{\sum_{k=1}^n |a_k|}
$$

where:
- $a_j$ is the saliency score of the $j$-th token
- $n$ is the total number of tokens

The complexity score for the instance is then calculated as the entropy of the fractional contributions:

$$
\mathrm{Complexity} = -\frac{1}{n}\sum_{j=1}^n f_j \cdot \log(f_j + \epsilon)
$$

where $\epsilon$ is a small constant (e.g., $10^{-8}$) to avoid numerical instability. The final complexity score is the average across all instances, with higher values indicating more evenly distributed attributions (higher complexity) and lower values indicating more concentrated attributions (lower complexity).

---

### Sparseness ↑

It measures the sparsity of model attributions using the Gini index, which quantifies how concentrated the importance scores are across features. For each instance, the absolute values of the attributions are sorted in ascending order, and the Gini index is computed as:

$$
\mathrm{Sparseness} = 1 - 2 \cdot \frac{\sum_{j=1}^n (n - j + 0.5) \cdot |a_j|}{\left(\sum_{j=1}^n |a_j|\right) \cdot n}
$$

where:
- $a_j$ is the $j$-th sorted attribution value
- $n$ is the total number of features (tokens)
- $|a_j|$ is the absolute value of the $j$-th attribution

The sparseness score ranges from 0 to 1, where:
- **0** indicates dense attributions (importance is evenly distributed across features)
- **1** indicates sparse attributions (importance is concentrated on a few features)

The final sparseness score is the average across all instances, providing a single metric to evaluate how focused the model is on specific features.

---
