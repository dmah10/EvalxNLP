## 🧠 Explaining Single Sentence

EvalxNLP integrates seamlessly with Hugging Face Transformer-based models. You can use any publicly available text classification model or a locally fine-tuned one. This guide walks you through the process of generating explanations for a single sentence.

---

### Step 1: Initialize the Model and Tokenizer

Begin by loading your pre-trained model and tokenizer. You can choose any model from the [Hugging Face Model Hub](https://huggingface.co/models).

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```
### Step 2: Initialize the XAIFramework Class

Instantiate the framework with the desired explainers and evaluators. EvalxNLP supports multiple plug-and-play XAI modules.

```python
from XAIbenchmark import XAIFramework
from explainers import InputXGradientExplainer, IntegratedGradientsExplainer
from evaluators import AUCTPEvaluator, SoftComprehensivenessEvaluator
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize explainers
ig = IntegratedGradientsExplainer(model, tokenizer)
ixg = InputXGradientExplainer(model, tokenizer, multiply_by_inputs=True)

# Initialize evaluators
sc = SoftComprehensivenessEvaluator(model, tokenizer)
auctp = AUCTPEvaluator(model, tokenizer)

# Instantiate the framework
xai_framework = XAIFramework(
    model,
    tokenizer,
    explainers=[ig, ixg],
    evaluators=[sc, auctp],
    device=device
)
```

### Step 3: Generate Explanations

Provide an input sentence and the corresponding target label to generate post-hoc explanations.

```python
example = "Worst experience I've ever had!"
label = "negative"

exps = xai_framework.explain(input_data=example, target_label=label)
```

### Step 4: Visualize Explanations

You can visualize the output explanations directly using the built-in visualization tool.

```python
xai_framework.visualize(exps)
```

## 📂 Explaining Datasets

EvalxNLP allows users to explain and benchmark datasets from both the Hugging Face Hub and local files (e.g., CSV or Excel) using the `DatasetLoader` class. It supports text classification tasks such as **Sentiment Analysis**, **Hate Speech Detection**, and **Natural Language Inference (NLI)**.

---

### Step 1: Define Dataset Loading Arguments

To begin, define the dataset loading configuration using the `LoadDatasetArgs` data class. The following fields are required:

- **`dataset_name`**: Name of the dataset (e.g., Hugging Face dataset ID or `"csv"`/`"excel"` for local files).
- **`text_field`**: Column name containing the input text.
- **`label_field`**: Column name containing target labels.

Optional fields:
- **`rationale_field`**: For plausibility-based evaluation using human rationales.
- **`text_field_2`**: For NLI tasks, used to distinguish premise and hypothesis.

---

### Step 2: Load the Dataset

Once configured, pass the arguments to `load_fields_from_dataset()` to retrieve the content:

```python
from dataset_loaders.dataset_loader import LoadDatasetArgs, load_fields_from_dataset

dataset_args = LoadDatasetArgs(
    dataset_name="Hate-speech-CNERG/hatexplain",
    text_field="post_tokens",
    label_field="annotators",
    rationale_field="rationales",
    dataset_split="test"
)

input_texts, labels, rationales = load_fields_from_dataset(dataset_args)
```

This returns:

- input_texts: List of input sentences.

- labels: List of class labels.

- rationales: (Optional) Token-level binary importance annotations.

### Step 3: Ensure Correct Format
To use the data with EvalxNLP, ensure each field is in the expected format:

Input Texts: List of strings.

Labels: List of integers or strings.

Rationales: List of binary lists (e.g., [0, 1, 0, 1, 0]).

#### Data Post-processing Example

EvalxNLP provides helper classes to process specific datasets. Here's how to prepare data from the HateXplain dataset:

```python
import numpy as np
from dataset_loaders.hatexplain import HateSpeechProcessor

class HateSpeechProcessor:
    def __init__(self):
        pass

    def process_input_texts(self, input_texts):
        return [" ".join(text_array) for text_array in input_texts]

    def process_labels(self, labels):
        return [int(np.bincount(label["label"]).argmax()) for label in labels]

    def process_rationales(self, rationales):
        return rationales

    def process_dataset(self, input_texts, labels, rationales):
        processed_input_texts = self.process_input_texts(input_texts)
        processed_labels = self.process_labels(labels)
        processed_rationales = self.process_rationales(rationales)
        return processed_input_texts, processed_labels, processed_rationales

# Process the dataset
hs = HateSpeechProcessor()
processed_input_texts, processed_labels, processed_rationales = hs.process_dataset(input_texts, labels, rationales)
```
### (Optional) Subset Selection

You can easily select a subset of samples to speed up evaluation:

```python
input_texts_sample = processed_input_texts[:10]
labels_sample = processed_labels[:10]
rationale_sample = processed_rationales[:10]
```

### Step 4: Generate Explanations

Finally, pass the data into the get_feature_importance_for_dataset() function to generate and save explanations:

```python
exp_scores = xai_framework.get_feature_importance_for_dataset(
    input_texts_sample,
    labels_sample,
    rationale_sample,
    output_file="../results/scores/movie_xx.json"
)
```
The results include explanation scores per token and can be stored for further analysis or visualization.