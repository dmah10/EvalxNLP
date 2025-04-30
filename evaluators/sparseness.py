from typing import Callable
import torch
from .baseevaluator import BaseEvaluator

class SparsenessEvaluator(BaseEvaluator):
    """Implementation of the sparseness metric.

    Computes the sparseness of the model based on the Gini index.

    Attributes:
        model (callable): model to explain.
    """
    NAME = "Sparseness ↑"

    def __init__(self, model, tokenizer, device="cpu"):
        """Initializes Sparseness with model and task."""
        self.model = model
        self.tokenizer = tokenizer
        self._device = device
        self.requires_human_rationale = False

    def get_sparsity(self, attribution: torch.Tensor) -> torch.Tensor:
        """Computes the sparseness of the model's attributions.

        Args:
            attribution (torch.Tensor): attributions for each instance.

        Returns:
            torch.Tensor: average sparseness score.
        """
        n_instances, n_features = attribution.shape

        # Compute the absolute value of attributions
        abs_attrib = torch.abs(attribution)

        # Calculate the total sum of absolute attributions
        attrib_sum = abs_attrib.sum(dim=1)

        # Compute the weighted sum for each instance
        weighted_sum = torch.zeros(n_instances).to(self._device)
        for j in range(n_features):
            weighted_sum += (n_features - j + 0.5) * abs_attrib[:, j]

        # Normalize to obtain sparseness score
        sparsity = 1 - 2 * weighted_sum / (attrib_sum * n_features + 1e-8)
        
        # Return the mean sparseness score across all instances
        return sparsity.mean()

    def compute(self, explanations) -> dict:
        """Compute the sparseness metric for the main model without random model comparison.

        Args:
            explanations: List of explanation objects containing attributions.

        Returns:
            dict: dict of sparseness metrics.
        """
        explanations = explanations if isinstance(explanations, list) else [explanations]

        # Extract attributions (importance scores) from each Explanation object
        attributions = [torch.tensor(explanation.scores).to(self._device) for explanation in explanations]
        
        # Find the maximum length of the attributions
        max_len = max(len(attr) for attr in attributions)
        
        # Pad the attributions to the same length
        padded_attributions = [torch.nn.functional.pad(attr, (0, max_len - len(attr))) for attr in attributions]
        
        # Stack padded attributions into a single tensor
        attributions = torch.stack(padded_attributions)

        # Sort attributions by absolute value and compute sparseness
        sorted_attrib = torch.sort(torch.abs(attributions), dim=1)[0]
        spars_score = self.get_sparsity(sorted_attrib)
        
        # Store original sparseness score in the metrics dictionary
        result = spars_score.item()
        return result