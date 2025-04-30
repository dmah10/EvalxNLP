EvalxNLP supports **Hate Speech Detection** for both **single sentence evaluation** and **dataset-level benchmarking**. This section demonstrates how to use the framework for both cases.

---

### 🧪 Single Sentence

The following example shows how to generate and evaluate model explanations for a single sentence classified as offensive:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from XAIbenchmark import XAIFramework

# Load a hate speech detection model
model_name = "Hate-speech-CNERG/bert-base-uncased-hatexplain"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Initialize the explanation framework
xai_framework = XAIFramework(model, tokenizer)

# Define input and target label
sentence = "I hate people from that community"
label = "offensive"

# Generate explanation and visualize
exps = xai_framework.explain(input_data=sentence, target_label=label)
xai_framework.visualize(exps)

# Evaluate explanation without a human rationale (optional)
xai_framework.evaluate_single_sentence(sentence, target_label="offensive")
```

### 📚 Dataset (HateXplain)

EvalxNLP also supports dataset-level evaluation using the [HateXplain](https://huggingface.co/datasets/Hate-speech-CNERG/hatexplain) dataset using the built-in class `HateSpeechProcessor`. The following example demonstrates the full pipeline: generating importance scores, visualizing them, interpreting them with LLM-generated textual explanations, computing evaluation metrics, visualizing these metrics, and finally interpreting them using LLM-based explanations.

```python
import sys
sys.path.append('..')
from XAIbenchmark import XAIFramework
import warnings
import openpyxl
import torch
warnings.filterwarnings("ignore")
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dataset_loaders.dataset_loader import LoadDatasetArgs,load_fields_from_dataset
from dataset_loaders.movie_rationales import MovieRationalesProcessor
from LLMExplanationGenerator import LLMExplanationGenerator
from EvaluationMetricsExplainer import EvaluationMetricsExplainer

device = "cuda" if torch.cuda.is_available() else "cpu"

api_key = "YOUR_API_KEY"

model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

xai_framework = XAIFramework(model, tokenizer, device=device)

dataset_args_ = LoadDatasetArgs(
    dataset_name="Hate-speech-CNERG/hatexplain",
    text_field="post_tokens",
    label_field="annotators",
    rationale_field="rationales",
    dataset_split="test",
)
# Load the dataset fields
results_hate = load_fields_from_dataset(dataset_args_)

texts_hate= results_hate['text']
labels_hate= results_hate['labels']
rationales_hate= results_hate['rationales']

hs= HateSpeechProcessor(tokenizer)
processed_input_texts_hate, processed_labels_hate, processed_rationales_hate= hs.process_dataset(input_texts=texts_hate, labels=labels_hate, rationales=rationales_hate)

#Select a sub-sample if you want
input_texts_sample=processed_input_texts_hate[:20]
labels_sample=processed_labels_hate[:20]
rationale_sample= processed_rationales_hate[:20]

exp_scores= xai_framework.get_feature_importance_for_dataset(input_texts_sample,labels_sample,rationale_sample,output_file="../results/scores/hatespeech_scores.json")

scores_explainer = LLMExplanationGenerator(api_key=api_key)

# Generate and save explanations (returns both explanations and file paths)
explanations, saved_files = scores_explainer.generate_and_save_explanations(
    exps=exp_scores,
    output_format="both"  # or "json"/"html"
)

scores_explainer.display_explanations(explanations)

metrics= xai_framework.compute_evaluation_metrics(exp_scores)
xai_framework.create_pivot_table(metrics,save_path="../results/metrics/hatespeech.xlsx")

metrics_explainer = EvaluationMetricsExplainer(api_key=api_key)

results = metrics_explainer.explain_results(metrics)
json_path, html_path = metrics_explainer.save_results(results)

metrics_explainer.display_results(results)
```
