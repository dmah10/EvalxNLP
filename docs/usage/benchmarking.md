EvalxNLP allows users to **evaluate the quality and reliability** of generated explanations and compare multiple explainers across plausibility and faithfulness metrics.

---

### 🧪 Evaluating a Single Sentence

#### Step 1: Evaluate Explanations

Use the `evaluate_single_sentence()` method from the `XAIFramework` class to evaluate explanations for a single sentence. You may also provide a **human rationale** to enable plausibility metric evaluation.

#### Human Rationale

- Human rationales are **binary lists** indicating which tokens are important for the target label.
- A value of `1` marks a token as **important**, and `0` as **not important**.
- If no rationale is provided, plausibility metrics will be **skipped**.

```python
example = "Worst experience I've ever had!!"
label = "negative"
human_rationale = [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0]  # Binary rationale

metrics = xai_framework.evaluate_single_sentence(
    sentence=example,
    target_label=label,
    human_rationale=human_rationale
)
```

#### Step 2: View Results
Visualize the evaluation metrics in a tabular format using the `create_pivot_table()` function:

```python
xai_framework.create_pivot_table(metrics)
```

- Metrics are color-coded by property.
- Darker hues indicate stronger performance.
- The best-performing score for each metric is bolded for easier comparison.

### 🧪 Evaluating a Dataset
#### Step 1: Compute Evaluation Metrics
Use the `compute_evaluation_metrics()` method to calculate explanation quality across a dataset. It takes a list of explanation objects as input (e.g., from `get_feature_importance_for_dataset()`).

```python
metrics = xai_framework.compute_evaluation_metrics(exp_scores)
```

#### Step 2: Visualize Results
Visualize the dataset-level evaluation using the same pivot table function:

```python
xai_framework.create_pivot_table(metrics)
```
