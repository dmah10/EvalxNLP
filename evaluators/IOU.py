
import numpy as np
from utils.saliency_utils import top_k_selection
from .baseevaluator import BaseEvaluator


def calculate_avg_rationale_length(explanations):
    """
    Calculate the average length of human rationales across the explanations.

    Args:
        explanations: List of Explanation objects.

    Returns:
        int: Average length of rationales.
    """
    rationale_lengths = [
        len([r for r in exp.rationale if r == 1]) for exp in explanations
    ]
    return int(np.mean(rationale_lengths)) if rationale_lengths else 0


def evaluate_instance(explanation, human_rationale, k, metric_func):
    """
    Evaluate metrics for a single instance.

    Args:
        explanation: Explanation object for the instance.
        human_rationale: Human-provided rationale for the instance.
        k: Number of top tokens to select.
        metric_func: Function to compute the metric (e.g., IOU or F1).

    Returns:
        float: Computed metric value.
    """
    saliency_scores = explanation.scores[1:-1]  # Exclude special tokens ([CLS], [SEP])
    ground_truth_rationale = human_rationale[:len(saliency_scores)]
    predicted_rationale = top_k_selection(np.array(saliency_scores), k)
    return metric_func(predicted_rationale, np.array(ground_truth_rationale))


def calculate_iou(predicted, ground_truth):
    """
    Calculate Intersection over Union (IOU).

    Args:
        predicted: Predicted binary rationale array.
        ground_truth: Ground truth binary rationale array.

    Returns:
        float: IOU value.
    """
    predicted_set = set(np.where(predicted == 1)[0])
    ground_truth_set = set(np.where(ground_truth == 1)[0])

    intersection = len(predicted_set & ground_truth_set)
    union = len(predicted_set | ground_truth_set)

    return intersection / union if union > 0 else 0.0


def calculate_f1(predicted, ground_truth):
    """
    Calculate F1-score.

    Args:
        predicted: Predicted binary rationale array.
        ground_truth: Ground truth binary rationale array.

    Returns:
        float: F1-score value.
    """
    predicted_set = set(np.where(predicted == 1)[0])
    ground_truth_set = set(np.where(ground_truth == 1)[0])

    intersection = len(predicted_set & ground_truth_set)
    precision = intersection / len(predicted_set) if predicted_set else 0.0
    recall = intersection / len(ground_truth_set) if ground_truth_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return f1


class IOUF1Evaluator(BaseEvaluator):
    NAME = "IOU F1 score ↑"
    """
    Class to evaluate faithfulness of saliency explanations using IOU F1-score.
    """

    def __init__(self, model, tokenizer, device="cpu"):
        """
        Initialize the evaluator.

        Args:
            model: The model used for predictions.
            tokenizer: Tokenizer used to process text.
            device (str): Device to use for computations ("cuda" or "cpu").
        """
        self.model = model
        self.tokenizer = tokenizer
        self._device = device
        self.requires_human_rationale = True
        self.avg_rationale_length = None

    def evaluate(self, explanations):
        """
        Evaluate metrics for the entire dataset or a single instance.

        Args:
            explanations: List of Explanation objects or a single Explanation.

        Returns:
            float: IOU F1-score.
        """
        if not isinstance(explanations, list):
            explanations = [explanations]  # Convert to list if it's a single Explanation

        # Dataset evaluation
        metrics_list = []
        matches = 0  # Count of instances where IOU >= 0.5
        for exp in explanations:
            if exp.rationale is None:
                print("No rationale provided for the instance. Skipping evaluation.")
                continue
            iou = evaluate_instance(exp, exp.rationale, self.avg_rationale_length, calculate_iou)
            metrics_list.append(iou)
            # if iou >= 0.5:  # Check if IOU >= 0.5
            #     matches += 1

        # IOU_F1 = matches / len(metrics_list) if metrics_list else 0.0  # IOU F1-score
        IOU_F1=np.mean(metrics_list) if metrics_list else 0.0
        return IOU_F1

    def compute(self, explanations):
        """
        Compute metrics for the entire dataset or a single instance.

        Args:
            explanations: List of Explanation objects or a single Explanation.

        Returns:
            float: IOU F1-score.
        """
        explanations = explanations if isinstance(explanations, list) else [explanations]
        self.avg_rationale_length = calculate_avg_rationale_length(explanations)
        return self.evaluate(explanations)


class TokenF1Evaluator(BaseEvaluator):
    NAME = "Token F1 score ↑"
    """
    Class to evaluate faithfulness of saliency explanations using Token F1-score.
    """

    def __init__(self, model, tokenizer, device="cpu"):
        """
        Initialize the evaluator.

        Args:
            model: The model used for predictions.
            tokenizer: Tokenizer used to process text.
            device (str): Device to use for computations ("cuda" or "cpu").
        """
        self.model = model
        self.tokenizer = tokenizer
        self._device = device
        self.requires_human_rationale = True
        self.avg_rationale_length = None

    def evaluate(self, explanations):
        """
        Evaluate metrics for the entire dataset or a single instance.

        Args:
            explanations: List of Explanation objects or a single Explanation.

        Returns:
            float: Token F1-score.
        """
        if not isinstance(explanations, list):
            explanations = [explanations]  # Convert to list if it's a single Explanation

        # Dataset evaluation
        metrics_list = []
        for exp in explanations:
            if exp.rationale is None:
                print("No rationale provided for the instance. Skipping evaluation.")
                continue
            f1 = evaluate_instance(exp, exp.rationale, self.avg_rationale_length, calculate_f1)
            metrics_list.append(f1)

        Token_F1 = np.mean(metrics_list) if metrics_list else 0.0  # Token F1-score
        return Token_F1

    def compute(self, explanations):
        """
        Compute metrics for the entire dataset or a single instance.

        Args:
            explanations: List of Explanation objects or a single Explanation.

        Returns:
            float: Token F1-score.
        """
        explanations = explanations if isinstance(explanations, list) else [explanations]
        self.avg_rationale_length = calculate_avg_rationale_length(explanations)
        return self.evaluate(explanations)
