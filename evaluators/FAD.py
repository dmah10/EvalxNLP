import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy.special import softmax
import torch
from numpy import trapz
from explanation import Explanation
from tqdm import tqdm
from .baseevaluator import BaseEvaluator

class FADEvaluator(BaseEvaluator):
    """
    Class to evaluate the impact of feature (token) dropping on text data
    based on saliency scores (e.g., Gradient x Input).
    """
    NAME = "FAD ↓"

    def __init__(self, model, tokenizer,device="cpu", batch_size=32):
        """
        Initialize the evaluator.

        Args:
            model: The trained text classification model.
            tokenizer: Tokenizer used for the model.
            batch_size: Number of samples to process in a batch.
            device: Device to run computations on (e.g., "cuda:0" or "cpu").
        """
        self.model = model
        self.tokenizer = tokenizer
        self.requires_human_rationale = False
        self.batch_size = batch_size
        self._device = device

    def _replace_tokens_with_baseline(self, tokens, saliency_scores, percent_to_drop):
        """
        Replace the top `percent_to_drop` tokens with a baseline token (e.g., [MASK]).

        Args:
            tokens: List of tokens.
            saliency_scores: Saliency scores for each token.
            percent_to_drop: Percentage of tokens to drop.

        Returns:
            List of modified tokens.
        """
        tokens_modified = tokens.copy()
        num_tokens_to_drop = int(len(tokens) * percent_to_drop / 100)
        saliency_sorted_indices = np.argsort(-np.abs(saliency_scores))
        tokens_to_replace = saliency_sorted_indices[:num_tokens_to_drop]

        baseline_token = self.tokenizer.mask_token
        for idx in tokens_to_replace:
            tokens_modified[idx] = baseline_token
        return tokens_modified

    def evaluate(self, explanations, percent_dropped_features):
        """
        Evaluate the impact of dropping tokens on model accuracy.

        Args:
            explanations: List of explanation objects.
            percent_dropped_features: List of percentages of tokens to drop.

        Returns:
            pd.DataFrame: Results containing accuracy for each percentage of tokens dropped.
        """
        results = []
        for percent_to_drop in percent_dropped_features:
            predictions, labels = [], []

            for i in range(0, len(explanations), self.batch_size):
                batch_explanations = explanations[i:i + self.batch_size]
                batch_texts = []
                batch_labels = []

                for exp in batch_explanations:
                    tokens = exp.tokens
                    saliency_scores = np.array(exp.scores)
                    label = exp.target_pos_idx

                    modified_tokens = self._replace_tokens_with_baseline(tokens, saliency_scores, percent_to_drop)
                    modified_text = self.tokenizer.convert_tokens_to_string(modified_tokens)
                    batch_texts.append(modified_text)
                    batch_labels.append(label)

                # Tokenize and move inputs to the GPU
                encoded_inputs = self.tokenizer(
                    batch_texts, return_tensors="pt", truncation=True, max_length=512, padding=True
                )
                encoded_inputs = {key: val.to(self._device) for key, val in encoded_inputs.items()}

                with torch.no_grad():
                    logits = self.model(**encoded_inputs).logits
                batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()

                predictions.extend(batch_predictions)
                labels.extend(batch_labels)

            accuracy = accuracy_score(labels, predictions)
            results.append({"Percent Dropped": percent_to_drop, "Accuracy": accuracy})

        return pd.DataFrame(results)

    def calculate_n_auc(self, results_df, percent_range=(0, 20)):
        """
        Calculate the normalized AUC for the given percentage range.

        Args:
            results_df: DataFrame containing the results.
            percent_range: Range of percentages to consider for AUC calculation.

        Returns:
            float: Normalized AUC.
        """
        filtered_results = results_df[(results_df["Percent Dropped"] >= percent_range[0]) &
                                      (results_df["Percent Dropped"] <= percent_range[1])]
        x = filtered_results["Percent Dropped"].values
        y = filtered_results["Accuracy"].values
        auc = trapz(y, x)
        max_auc = (x[-1] - x[0]) * max(y)
        return auc / max_auc if max_auc > 0 else 0.0

    def plot_results(self, results_df):
        """
        Plot the results (accuracy vs. percentage of tokens dropped).

        Args:
            results_df: DataFrame containing the results.
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(results_df["Percent Dropped"], results_df["Accuracy"], marker="o", label="Accuracy")
        plt.xlabel("Percent of Tokens Dropped")
        plt.ylabel("Accuracy")
        plt.title("Impact of Dropping Top Tokens on Accuracy")
        plt.grid()
        plt.legend()
        plt.show()

    def compute(self, explanations, percent_dropped_features=None, percent_range=(0, 20)):
        """
        Compute the normalized AUC for the given explanations.

        Args:
            explanations: List of explanation objects.
            percent_dropped_features: List of percentages of tokens to drop.
            percent_range: Range of percentages to consider for AUC calculation.

        Returns:
            float: Normalized AUC.
        """
        explanations = explanations if isinstance(explanations, list) else [explanations]
        if percent_dropped_features is None:
            percent_dropped_features = list(range(0, 41, 10))

        results_df = self.evaluate(explanations, percent_dropped_features)
        final_n_auc = self.calculate_n_auc(results_df, percent_range)
        # self.plot_results(results_df)  # Uncomment to plot results
        return final_n_auc