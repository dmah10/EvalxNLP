import sys
sys.path.append('..')
from XAIbenchmark import XAIFramework
import warnings
import openpyxl
import torch
warnings.filterwarnings("ignore")

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from explainers import InputXGradientExplainer, IntegratedGradientsExplainer, DeepLiftExplainer, LimeExplainer, SHAPExplainer, SHAPIQExplainer, GuidedBackpropExplainer
from evaluators import AUCTPEvaluator, SoftComprehensivenessEvaluator, SoftSufficiencyEvaluator, FADEvaluator, SparsenessEvaluator, ComplexityEvaluator, IOUF1Evaluator,TokenF1Evaluator, AUPRCEvaluator
from dataset_loaders.dataset_loader import LoadDatasetArgs,load_fields_from_dataset
from dataset_loaders.movie_rationales import MovieRationalesProcessor
from dataset_loaders.hatexplain import HateSpeechProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "Hate-speech-CNERG/bert-base-uncased-hatexplain"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

ig= IntegratedGradientsExplainer(model,tokenizer,device=device)
gb= GuidedBackpropExplainer(model,tokenizer,device=device)
dl= DeepLiftExplainer(model,tokenizer,device=device)
ixg= InputXGradientExplainer(model,tokenizer,multiply_by_inputs=True,device=device)
g= InputXGradientExplainer(model,tokenizer,multiply_by_inputs=False,device=device)
lime= LimeExplainer(model,tokenizer)
shap= SHAPExplainer(model,tokenizer)
shapiq= SHAPIQExplainer(model,tokenizer)

sc= SoftComprehensivenessEvaluator(model,tokenizer,device=device)
ss= SoftSufficiencyEvaluator(model,tokenizer,device=device)
fad= FADEvaluator(model,tokenizer,device=device)
sp= SparsenessEvaluator(model,tokenizer,device=device)
cx= ComplexityEvaluator(model,tokenizer,device=device)
auctp= AUCTPEvaluator(model,tokenizer,device=device)
iou_f1= IOUF1Evaluator(model,tokenizer,device=device)
token_f1= TokenF1Evaluator(model,tokenizer,device=device)
auprc= AUPRCEvaluator(model,tokenizer,device=device)

xai_framework = XAIFramework(model, tokenizer,explainers=[gb,dl,ixg,g,ig,lime,shap,shapiq], evaluators=[sc,ss,fad,sp,cx,auctp,iou_f1,token_f1,auprc], device=device)

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

xai_framework.benchmark_dataset(input_texts_sample, labels_sample, rationale_sample, save_path="../results/hate_xplain.xlsx")