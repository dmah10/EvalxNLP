import numpy as np
import torch
from sklearn.metrics import auc, accuracy_score
import matplotlib.pyplot as plt
from .baseevaluator import BaseEvaluator


class AUCTPEvaluator((BaseEvaluator)):
    NAME = "AUTPC ↓"

    def __init__(self, model, tokenizer,device="cpu", batch_size=32):
        """
        Initializes the FaithfulnessEvaluator.

        Args:
            model: The pre-trained model (e.g., BERT) to evaluate.
            tokenizer: Tokenizer corresponding to the model.
            batch_size (int): Batch size for processing.
            device (str): Device to use for computations ("cuda" or "cpu").
        """
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self._device = device
        self.requires_human_rationale = False

    @staticmethod
    def mask_tokens(tokens, importance_scores, threshold):
        """
        Masks the top `threshold%` tokens based on importance scores.

        Args:
            tokens (list): List of tokens.
            importance_scores (list): List of importance scores for each token.
            threshold (float): Percentage of tokens to mask.

        Returns:
            list: Masked tokens.
        """
        n_tokens_to_mask = int(len(tokens) * (threshold / 100))
        indices_to_mask = np.argsort(-np.array(importance_scores))[:n_tokens_to_mask]
        return [
            "[MASK]" if i in indices_to_mask else token
            for i, token in enumerate(tokens)
        ]

    def evaluate_performance(self, explanations, thresholds):
        """
        Evaluates model performance after masking tokens at various thresholds.

        Args:
            explanations (list): List of explanations containing tokens, scores, and labels.
            thresholds (list): List of thresholds (percentages of tokens to mask).

        Returns:
            list: Performance scores (accuracy) at each threshold.
        """
        performance_scores = []

        for threshold in thresholds:
            masked_texts = []
            labels = []

            for data in explanations:
                tokens = data.tokens
                importance_scores = data.scores
                label = data.target_pos_idx

                # Mask tokens based on saliency scores and threshold
                masked_tokens = self.mask_tokens(tokens, importance_scores, threshold)
                masked_text = self.tokenizer.convert_tokens_to_string(masked_tokens)

                masked_texts.append(masked_text)

                if isinstance(label, str):
                    label = self.model.config.label2id[label]

                labels.append(label)

            batch_predictions = []

            for i in range(0, len(masked_texts), self.batch_size):
                batch_texts = masked_texts[i:i + self.batch_size]

                # Ensure the batch is not empty before proceeding
                if not batch_texts:
                    print(f"Skipping empty batch at index {i}")
                    continue

                # Tokenize and predict
                try:
                    encoded_inputs = self.tokenizer(
                        batch_texts, return_tensors="pt", truncation=True, max_length=512, padding=True
                    ).to(self._device)
                    with torch.no_grad():
                        logits = self.model(**encoded_inputs).logits
                        predictions = torch.argmax(logits, dim=1).cpu().numpy()

                    batch_predictions.extend(predictions)

                except Exception as e:
                    print(f"Tokenizer error at batch index {i}: {e}")
                    continue  # Skip this batch

            # Ensure batch_predictions is not empty before calculating accuracy
            if not batch_predictions:
                print("No valid predictions generated.")
                performance_scores.append(0)  # Fallback to 0 accuracy for this threshold
            else:
                accuracy = accuracy_score(labels, batch_predictions)
                performance_scores.append(accuracy)

        return performance_scores

    def compute_auc_tp(self, thresholds, performance_scores):
        """
        Computes the Area Under the Threshold-Performance Curve (AUC-TP).

        Args:
            thresholds (list): List of thresholds (percentages of tokens masked).
            performance_scores (list): Performance scores at each threshold.

        Returns:
            float: AUC-TP value.
        """
        auc_tp = auc(thresholds, performance_scores)
        normalized_auc_tp = auc_tp / 100  # Normalize to [0, 1]
        return normalized_auc_tp

    def plot_performance(self, thresholds, performance_scores):
        """
        Plots the Threshold-Performance Curve.

        Args:
            thresholds (list): List of thresholds (percentages of tokens masked).
            performance_scores (list): Performance scores at each threshold.
        """
        plt.plot(thresholds, performance_scores, marker="o")
        plt.xlabel("Threshold (% of tokens masked)")
        plt.ylabel("Performance (Accuracy)")
        plt.title("Threshold-Performance Curve")
        plt.show()

    def compute(self, explanations):
        """
        Computes the faithfulness AUC for the given model, tokenizer, and precomputed scores.

        Args:
            explanations (list): List of explanations containing tokens, scores, and labels.

        Returns:
            float: AUC-TP value.
        """
        explanations = explanations if isinstance(explanations, list) else [explanations]
        thresholds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        # Evaluate performance across thresholds
        performance_scores = self.evaluate_performance(explanations, thresholds)

        # Compute AUC-TP
        auc_tp = self.compute_auc_tp(thresholds, performance_scores)
        # self.plot_performance(thresholds, performance_scores)  # Uncomment to plot the curve

        return auc_tp