<!-- ## 🧪 Usage Examples -->

You can use EvalxNLP for **Sentiment Analysis** usecase, on a **single sentence**, or **dataset**.

---

### 💬 Single Sentence

The following example demonstrates how to use EvalxNLP to explain and evaluate a model's prediction for a single input sentence:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from XAIbenchmark import XAIFramework

# Load a pre-trained sentiment classification model
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Initialize EvalxNLP explanation framework
xai_framework = XAIFramework(model, tokenizer)

# Define input and label
sentence = "Worst experience I've ever had!"
label = "negative"

# Generate explanation
exps = xai_framework.explain(input_data=sentence, target_label=label)

# Visualize the explanation
xai_framework.visualize(exps)

# Evaluate explanation against a human rationale
xai_framework.evaluate_single_sentence(
    sentence,
    target_label="negative",
    human_rationale=[1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0]
)
```

### 🎬 Dataset (Movie Reviews)

You can also apply EvalxNLP to a dataset such as [Movie Reviews](https://huggingface.co/datasets/eraser-benchmark/movie_rationales) using the built-in class `MovieRationalesProcessor`. The following example demonstrates the full pipeline: generating importance scores, visualizing them, interpreting them with LLM-generated textual explanations, computing evaluation metrics, visualizing these metrics, and finally interpreting them using LLM-based explanations.

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
from explainers import IntegratedGradientsExplainer, GuidedBackpropExplainer
from evaluators import AUCTPEvaluator, SoftComprehensivenessEvaluator,ComplexityEvaluator, IOUF1Evaluator
from dataset_loaders.dataset_loader import LoadDatasetArgs,load_fields_from_dataset
from dataset_loaders.movie_rationales import MovieRationalesProcessor
from LLMExplanationGenerator import LLMExplanationGenerator
from EvaluationMetricsExplainer import EvaluationMetricsExplainer

device = "cuda" if torch.cuda.is_available() else "cpu"

api_key = "YOUR_API_KEY"

model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

ig= IntegratedGradientsExplainer(model,tokenizer,device=device)
gb= GuidedBackpropExplainer(model,tokenizer,device=device)

sc= SoftComprehensivenessEvaluator(model,tokenizer,device=device)
cx= ComplexityEvaluator(model,tokenizer,device=device)
iou_f1= IOUF1Evaluator(model,tokenizer,device=device)

xai_framework = XAIFramework(model, tokenizer,explainers=[ig,gb], evaluators=[sc,cx,iou_f1], device=device)

dataset_args_ = LoadDatasetArgs(
    dataset_name="eraser-benchmark/movie_rationales",
    text_field="review",
    label_field="label",
    rationale_field="evidences",
    dataset_split="test",
)

# Load the dataset fields
results = load_fields_from_dataset(dataset_args_)
input_texts= results['text']
labels= results['labels']
rationales= results['rationales']

mv= MovieRationalesProcessor(tokenizer)
processed_rationales= mv.process_dataset(input_texts, labels, rationales)

#Select a sub-sample if you want
input_texts_sample=input_texts[:2]
labels_sample=labels[:2]
rationale_sample= processed_rationales[:2]

exp_scores= xai_framework.get_feature_importance_for_dataset(input_texts_sample,labels_sample,rationale_sample,output_file="../results/scores/moviereviews_score.json")

scores_explainer = LLMExplanationGenerator(api_key=api_key)

# Generate and save explanations (returns both explanations and file paths)
explanations, saved_files = scores_explainer.generate_and_save_explanations(
    exps=exp_scores,
    output_format="both"  # or "json"/"html"
)

scores_explainer.display_explanations(explanations)

metrics= xai_framework.compute_evaluation_metrics(exp_scores)
xai_framework.create_pivot_table(metrics,save_path="../results/metrics/rationales.xlsx")

metrics_explainer = EvaluationMetricsExplainer(api_key=api_key)

results = metrics_explainer.explain_results(metrics)
json_path, html_path = metrics_explainer.save_results(results)

metrics_explainer.display_results(results)
```
