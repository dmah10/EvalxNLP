import pandas as pd
import shelve
import hashlib
from typing import Dict, List, Optional, Tuple,Union
from pathlib import Path
import json
import pickle

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from evaluators import SoftSufficiencyEvaluator, FADEvaluator, AUPRCEvaluator, SparsenessEvaluator, IOUF1Evaluator,TokenF1Evaluator, SoftComprehensivenessEvaluator, ComplexityEvaluator, AUCTPEvaluator
from explainers import InputXGradientExplainer, IntegratedGradientsExplainer,DeepLiftExplainer, LimeExplainer, SHAPExplainer, SHAPIQExplainer, GuidedBackpropExplainer
from explanation import Explanation

from utils.text_classifier_utils import TextClassifierEngine
from utils.saliency_utils import lp_normalize
from utils.generate_heatmap import generate_explanation_heatmap

class XAIFramework:
    def __init__(self, model,tokenizer,device,explainers: Optional[List] = None, evaluators: Optional[List] = None, cache_filename="xai_cache.db"):
        self.cache_filename = cache_filename
        self._device = device
        self.model= model
        self.tokenizer = tokenizer
        self.explainers = explainers or self._initialize_default_explainers()
        self.evaluators = evaluators or self._initialize_default_evaluators()
        self.text_classifier_engine = TextClassifierEngine(self.model, self.tokenizer)

    def _initialize_model_and_tokenizer(self) -> Tuple[Optional[AutoModelForSequenceClassification], Optional[AutoTokenizer]]:
        """
        Initialize the model and tokenizer from the pretrained model name.
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            print(f"Successfully loaded model from: {self.model_name}")
            return model, tokenizer
        except Exception as e:
            print(f"Error loading model from {self.model_name}: {e}")
            return None, None

    def _initialize_default_explainers(self) -> List[object]:
        return [
            SHAPExplainer(self.model, self.tokenizer),
            LimeExplainer(self.model, self.tokenizer),
            InputXGradientExplainer(self.model, self.tokenizer,device=self._device,multiply_by_inputs=False),
            InputXGradientExplainer(self.model, self.tokenizer, device=self._device, multiply_by_inputs=True),
            IntegratedGradientsExplainer(self.model, self.tokenizer, device=self._device),
            DeepLiftExplainer(self.model, self.tokenizer,device=self._device),
            SHAPIQExplainer(self.model, self.tokenizer),
            GuidedBackpropExplainer(self.model, self.tokenizer,device=self._device)
        ]

    def _initialize_default_evaluators(self) -> List[object]:
        return [
            SoftComprehensivenessEvaluator(self.model, self.tokenizer,self._device),
            SoftSufficiencyEvaluator(self.model, self.tokenizer,self._device),
            FADEvaluator(self.model, self.tokenizer, self._device),
            AUCTPEvaluator(self.model, self.tokenizer,self._device),
            ComplexityEvaluator(self.model, self.tokenizer, self._device),
            SparsenessEvaluator(self.model, self.tokenizer,self._device),
            AUPRCEvaluator(self.model, self.tokenizer,self._device),
            IOUF1Evaluator(self.model, self.tokenizer,self._device),
            TokenF1Evaluator(self.model, self.tokenizer,self._device)
        ]

    def _generate_cache_key(self, model: str, explainer: str, input_text: str, extra_params: Optional[Dict] = None) -> str:
        """
        Generate a unique cache key based on model, explainer, dataset, input text, and extra parameters.
        """ 
        text_hash = hashlib.md5(input_text.encode('utf-8')).hexdigest()

        input_str = f"{model}_{explainer}_{text_hash}"
        if extra_params:
            input_str += f"_{str(extra_params)}"
        cache_key = hashlib.md5(input_str.encode('utf-8')).hexdigest()
        return cache_key
   
    
    def create_pivot_table(self, input_data: Union[Dict, str], save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Create a styled pivot table with:
        - Metrics grouped by category (faithfulness → plausibility → complexity)
        - Lighter pastel color scheme (red, blue, purple)
        - No symbols in column names
        - Proper column display
        """
        
        try:
            # Load data from either dict or Excel
            if isinstance(input_data, dict):
                results = []
                for explainer_name, explainer_metrics in input_data.items():
                    for evaluator_name, metric_value in explainer_metrics.items():
                        results.append({
                            "Explainer": explainer_name,
                            "Evaluator": evaluator_name,
                            "Metric Value": metric_value
                        })
                results_df = pd.DataFrame(results)
            elif isinstance(input_data, str) and input_data.endswith('.xlsx'):
                results_df = pd.read_excel(input_data)
            else:
                raise ValueError("Input must be dictionary or Excel path")

            # Define metric categories and directions (in desired display order)
            METRIC_CONFIG = {
                # Faithfulness (pastel red)
                'Soft Comprehensiveness ↑': {'category': 'faithfulness', 'direction': 1},
                'Soft Sufficiency ↓': {'category': 'faithfulness', 'direction': -1},
                'AUTPC ↓': {'category': 'faithfulness', 'direction': -1},
                'FAD ↓': {'category': 'faithfulness', 'direction': -1},
                
                # Complexity (pastel purple)
                'Complexity ↓': {'category': 'complexity', 'direction': -1},
                'Sparseness ↑': {'category': 'complexity', 'direction': 1},

                 # Plausibility (pastel blue)
                'IOU F1 score ↑': {'category': 'plausibility', 'direction': 1},
                'Token F1 score ↑': {'category': 'plausibility', 'direction': 1},
                'AUPRC ↑': {'category': 'plausibility', 'direction': 1}
            }
            
            # Create pivot table with columns in specified order
            ordered_metrics = [m for m in METRIC_CONFIG.keys() if m in results_df['Evaluator'].unique()]
            pivot_table = results_df.pivot(index="Explainer", columns="Evaluator", values="Metric Value")[ordered_metrics]
            # Get min/max values per metric for consistent coloring
            metric_ranges = {
                metric: (results_df.loc[results_df['Evaluator'] == metric, 'Metric Value'].min(),
                        results_df.loc[results_df['Evaluator'] == metric, 'Metric Value'].max())
                for metric in ordered_metrics
            }

            def highlight_best(series):
                config = METRIC_CONFIG.get(series.name)
                if not config:
                    return [''] * len(series)
                
                is_best = series == series.max() if config['direction'] == 1 else series == series.min()
                return ['font-weight: bold' if v else '' for v in is_best]

            def color_by_metric(val, metric_name):
                if pd.isna(val):
                    return 'background-color: white; color: #555555'
                
                config = METRIC_CONFIG.get(metric_name)
                if not config:
                    return ''
                
                # Normalize value (0=worst, 1=best)
                vmin, vmax = metric_ranges[metric_name]
                val_norm = 0.5 if vmax == vmin else (val - vmin) / (vmax - vmin)
                if config['direction'] == -1:
                    val_norm = 1 - val_norm

                # Wider intensity range for better contrast (120-255)
                intensity = 130 + int(135 * val_norm)  # 120-255 range
                
                # Dynamic text color (white for dark backgrounds)
                text_color = 'black'

                if config['category'] == 'faithfulness':
                    # Distinct red gradient (no pink shift)
                    return f'background-color: rgb({min(255, intensity + 50)}, {max(100, 255 - intensity)}, {max(100, 255 - intensity)}); color: {text_color}'
                
                elif config['category'] == 'plausibility':
                    # Crisp blue gradient (no yellow/green)
                    return f'background-color: rgb({max(120, 255 - intensity)}, {max(120, 255 - intensity)}, {intensity}); color: {text_color}'
                
                elif config['category'] == 'complexity':
                    # Clear purple gradient (no gray/green)
                    return f'background-color: rgb({intensity}, {max(100, 255 - intensity)}, {intensity}); color: {text_color}'

            # Then modify the styling application:
            styled_table = pivot_table.style
            for metric in pivot_table.columns:
                styled_table = styled_table.applymap(
                    lambda x, m=metric: color_by_metric(x, m),
                    subset=[metric]
                ).apply(
                    highlight_best, 
                    subset=[metric]
                )
           
            # Save results
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                with pd.ExcelWriter(save_path) as writer:
                    results_df.to_excel(writer, sheet_name='Raw Data', index=False)
                    pivot_table.to_excel(writer, sheet_name='Pivot Table')
                    styled_table.to_excel(writer, sheet_name='Styled Results')
                print(f"Results saved to {save_path}")
            
            return styled_table
            
        except Exception as e:
            print(f"Error generating pivot table: {e}")
            return pd.DataFrame()

    def benchmark_dataset(self, input_texts: List[str], labels: List[str], rationales: Optional[List[str]] = None, extra_params: Optional[Dict] = None, split_type: str = "test",save_metrics_path: Optional[str] = None,save_scores_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate a table with results for all explainers and metrics for the entire dataset.
        """
        try:
            dataset_metrics = self.get_evaluation_metrics(input_texts, labels, rationales, save_scores_path= save_scores_path)
            # print(self.create_pivot_table(dataset_metrics,save_metrics_path))
            return dataset_metrics
        except Exception as e:
            print(f"Error generating explainer table for dataset: {e}")

    def get_evaluation_metrics(self, input_texts: List[str], labels: List[str], rationales: Optional[List[str]], extra_params: Optional[Dict] = None,save_scores_path: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Retrieve evaluation metrics for a model, explainer, and dataset, either from cache or by computation.
        Now supports a list of explainer names.
        """
        try:           
            explainer_results = self.get_feature_importance_for_dataset(input_texts, labels, rationales, output_file=save_scores_path)
            metrics = self.compute_evaluation_metrics(explainer_results)
            return metrics
        except Exception as e:
            print(f"Error retrieving evaluation metrics: {e}")
            return {}

    def compute_evaluation_metrics(self, explainer_results: Dict[str, List[Explanation]]) -> Dict[str, Dict[str, float]]:
        """
        Compute evaluation metrics for the entire dataset batch.
        """
        try:
            evaluator_results = {}
            for explainer_name, explanations in explainer_results.items():
                print(f"\nExplainer: {explainer_name}")
                evaluator_results[explainer_name] = {}
                evaluators = self.evaluators
                for evaluator in evaluators:
                    print(f"Computing value for {evaluator.NAME}")
                    evaluator_name = evaluator.NAME
                    try:
                        metric_result = evaluator.compute(explanations)
                    except Exception as e:
                        print(f"Error computing metric for {evaluator_name}: {e}")
                        continue
                    evaluator_results[explainer_name][evaluator_name] = metric_result
            return evaluator_results
        except Exception as e:
            print(f"Error computing evaluation metrics: {e}")
            return {}

    def process_text_list(self,text_list):
        """
        Processes a list of text inputs. If the input is a list of tuples, combines each tuple into a single string using the separator token.
        If the input is a list of strings, leaves it as is.

        Args:
            text_list (List[Union[str, Tuple[str, str]]): List of text inputs, which can be strings or tuples of two strings.
            sep_token (str): Separator token to use when combining tuples. Default is "[SEP]".

        Returns:
            List[str]: Processed list of strings.
        """
        sep_token="[SEP]"
        processed_texts = []

        for text in text_list:
            if isinstance(text, str):
                # If it's a string, add it directly to the list
                processed_texts.append(text)
            elif isinstance(text, tuple) and len(text) == 2:
                # If it's a tuple of two strings, combine them with the separator token
                combined_text = f"{text[0]} {sep_token} {text[1]}"
                processed_texts.append(combined_text)
            else:
                # Handle unexpected types (optional: raise an error or skip)
                raise ValueError(f"Unsupported input type: {type(text)}. Expected str or tuple of two strings.")

        return processed_texts

    def get_feature_importance_for_dataset(self, input_texts: List[str], labels: List[str], rationales: Optional[List[str]]=None, extra_params: Optional[Dict] = None, split_type: str = "test",output_file: Optional[str] = None) -> Dict[str, List[Explanation]]:
        """
        Compute feature importance scores for the entire dataset, for all explainer methods.
        Now using the output from xai_framework.load_fields_from_dataset.

        Args:
            input_texts (List[str]): List of input texts.
            labels (List[str]): List of labels.
            rationales (Optional[List[str]]): List of rationales.
            explainer_names (List[str]): List of explainer names to use.
            extra_params (Optional[Dict]): Additional parameters for explainers.
            split_type (str): Type of dataset split (e.g., "train", "test").

        Returns:
            Dict[str, List[Explanation]]: Dictionary of explanations for each explainer.
        """
        
        processed_texts = self.process_text_list(input_texts)
        all_explanations = {}
        try:
            for instance_id, (text, target_label) in enumerate(zip(processed_texts, labels)):
                print(f"Processing instance {instance_id}")
                rationale = rationales[instance_id] if rationales else None
                explainer_results = self.get_feature_importance_for_instance(
                    text, target_label, rationale, instance_id
                )
                
                for explainer_name, explanation in explainer_results.items():
                    if explainer_name not in all_explanations:
                        all_explanations[explainer_name] = []
                    all_explanations[explainer_name].append(explanation) 

            if output_file:
                self._save_explanations(all_explanations, output_file)
                
        except Exception as e:
            print(f"Error computing feature importance: {e}")
            return {}
        
        return all_explanations

    def _save_explanations(self, explanations, file_path: str):
        # Convert explanations to serializable format
        serializable_explanations = {}
        for explainer_name, explanation_list in explanations.items():
            serializable_explanations[explainer_name] = [
                {
                    'text': exp.text,
                    'tokens': [str(token) for token in exp.tokens],
                    'scores': exp.scores.tolist() if hasattr(exp.scores, 'tolist') else exp.scores,
                    'explainer': str(exp.explainer),
                    'target': str(exp.target)
                }
                for exp in explanation_list
            ]
    
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
             # Use encoding='utf-8' explicitly
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_explanations, f, indent=2, ensure_ascii=False)
            print(f"Saved explanations to {file_path}")
        except Exception as e:
            print(f"Error saving explanations: {e}")

    def get_feature_importance_for_instance(self, text: str, target_label: int, rationale: Optional[str], extra_params: Optional[Dict] = None) -> Dict[str, Explanation]:
        """
        Retrieve or compute feature importance scores for a given model, explainer, and dataset instance.
        """
        try:
            all_results = {}
            explainers = self.explainers
           
            for explainer in explainers:
                cache_key = self._generate_cache_key(self.model, explainer.NAME, text, extra_params)
                with shelve.open(self.cache_filename) as cache:
                    if cache_key in cache:
                        print(f"Found cached explanation for {explainer.NAME}")
                        all_results[explainer.NAME] = cache[cache_key]
                    else:
                        print(f"Computing explanation for {explainer.NAME}")
                        try:
                            explanation = explainer.compute_feature_importance(text, target=target_label)
                            explanation = lp_normalize([explanation])
                            explanation.rationale = rationale
                            cache[cache_key] = explanation
                            all_results[explainer.NAME] = explanation
                        except Exception as e:
                            print(e)
                            continue

            return all_results
        except Exception as e:
            print(f"Error computing feature importance for instance: {e}")
            return {}

    def explain(self, input_data: str, target_label: int = 1) -> pd.DataFrame:
        """
        Generate and visualize explanations for single or multiple input sentences.

        Args:
            input_data (str): Single input sentence or a list of input sentences.
            target_label (int): Target label for explanation.
            explainer_names (Optional[List[str]]): List of explainer names to use.

        Returns:
            pd.DataFrame: Combined heatmap data for all inputs.
        """
        try:
            explainers = self.explainers

            validated_input = self.text_classifier_engine.validate_input(input_data)
            explanation_outputs = []

            for explainer in explainers:
                result = explainer.compute_feature_importance(validated_input, target_label)
                explanation_outputs.append(result)

            explanation_outputs = lp_normalize(explanation_outputs)
            return explanation_outputs
        except Exception as e:
            print(f"Error generating explanations: {e}")
            return pd.DataFrame()

    def visualize(self, explanations: List[Explanation], skip_boundary_tokens: bool = True, visualization_style: Optional[str] = "cool_warm") -> pd.DataFrame:
        """
        Visualize explanations as heatmaps.

        Args:
            explanations (List[Explanation]): List of explanations to visualize.
            skip_boundary_tokens (bool): Whether to skip [CLS] and [SEP] tokens.
            visualization_style (Optional[str]): Heatmap visualization style.

        Returns:
            pd.DataFrame: Heatmap data.
        """
        try:
            generate_explanation_heatmap(explanations, skip_boundary_tokens, visualization_style)
            return
        except Exception as e:
            print(f"Error visualizing explanations: {e}")
            return pd.DataFrame()

    def classify_text(self, input_data: str) -> Dict[str, float]:
        """
        Classify input text and return probabilities for each class.

        Args:
            input_data (str): Input text to classify.

        Returns:
            Dict[str, float]: Dictionary of class probabilities.
        """
        try:
            _, prediction_logits = self.text_classifier_engine.run_inference(input_data)
            probabilities = prediction_logits[0].softmax(-1)
            return {self.model.config.id2label[idx]: prob.item() for idx, prob in enumerate(probabilities)}
        except Exception as e:
            print(f"Error classifying text: {e}")
            return {}

    def evaluate_single_sentence(self, input_data: str, target_label: int = 1, human_rationale: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Evaluate all explainers and evaluators with the option to use human rationale for plausibility metrics.

        Args:
            input_data (str): Input text or data for explanation.
            target_label (int): The target label for the explanation.
            explainer_names (Optional[List[str]]): List of explainer names to evaluate. Defaults to all registered explainers.
            evaluator_names (Optional[List[str]]): List of evaluator names to use. Defaults to all registered evaluators.
            human_rationale (Optional[List[int]]): Human-provided rationale for plausibility metrics. Defaults to None.

        Returns:
            pd.DataFrame: A pivot table summarizing evaluation results.
        """
        try:
            explainers= self.explainers
            evaluators= self.evaluators
            results = []

            for explainer in explainers:
                explanation = explainer.compute_feature_importance(input_data, target_label)
                explanation = lp_normalize([explanation])
                explanation.rationale = human_rationale
                for evaluator in evaluators:
                    try:
                        if evaluator.requires_human_rationale and human_rationale is None:
                            metric_result = None
                            print(f"Skipping {evaluator.NAME} as human rationale is not provided.")
                            continue
                        else:
                            metric_result = evaluator.compute([explanation])
                        results.append({
                            "Explainer": explainer.NAME,
                            "Evaluator": evaluator.NAME,
                            "Value": metric_result
                        })
                    except:
                        print(f"Skipping {evaluator.NAME} due to an error.")
                        continue

            results_df = pd.DataFrame(results)
            # Convert DataFrame to dictionary in the required format
            results_dict = {
                explainer: {
                    evaluator: metric_value 
                    for evaluator, metric_value in group.set_index('Evaluator')['Value'].items()
                }
                for explainer, group in results_df.groupby('Explainer')
            }
            
            return results_dict
        except Exception as e:
            print(f"Error evaluating single sentence: {e}")
            return pd.DataFrame()



