
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from .baseevaluator import BaseEvaluator

class SoftSufficiencyEvaluator(BaseEvaluator):
    NAME = "Soft Sufficiency ↓"

    def __init__(self, model, tokenizer, device="cpu", max_len=512):
        """
        Initializes the Soft Normalized Sufficiency computation.
        
        Args:
            model (nn.Module): The pre-trained model (e.g., BERT).
            tokenizer (transformers.Tokenizer): Tokenizer used to tokenize text inputs.
            max_len (int): Maximum token length to which the input should be padded/truncated.
            device (str): Device to run computations on (e.g., "cuda:0" or "cpu").
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self._device = device
        self.requires_human_rationale = False

    def soft_perturb(self, embeddings, importance_scores, attention_mask):
        """
        Applies soft perturbation to the token embeddings based on the importance scores.
        
        Args:
            embeddings (torch.Tensor): The token embeddings (batch_size, seq_len, embedding_dim).
            importance_scores (torch.Tensor): Importance scores for each token in the sequence (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask to indicate the padding positions (batch_size, seq_len).
        
        Returns:
            torch.Tensor: Perturbed token embeddings.
        """
        # Move tensors to the correct device
        importance_scores = importance_scores.unsqueeze(-1).to(self._device)
        attention_mask = attention_mask.unsqueeze(-1).float().to(self._device)

        # Normalize importance scores
        normalized_importance_scores = (importance_scores - importance_scores.min()) / (importance_scores.max() - importance_scores.min())
        
        # Create a Bernoulli mask based on importance scores
        mask = torch.bernoulli(normalized_importance_scores)
        mask = mask * attention_mask  # Apply the attention mask
        
        # Apply the mask to the embeddings
        perturbed_embeddings = embeddings * mask
        return perturbed_embeddings

    def sufficiency_(self, full_text_probs: np.array, reduced_probs: np.array) -> np.array:
        """
        Computes the sufficiency score.
        
        Args:
            full_text_probs (np.array): Model predictions on the full text.
            reduced_probs (np.array): Model predictions on the perturbed text.
        
        Returns:
            np.array: Sufficiency scores.
        """
        sufficiency = 1 - np.maximum(0, full_text_probs - reduced_probs)
        return sufficiency

    def compute_sufficiency(self, original_input, perturbed_input, calculate_baseline=False):
        """
        Computes the soft sufficiency score based on the change in model predictions.
        
        Args:
            original_input (dict): The input dictionary for the model with original tokens.
            perturbed_input (dict): The input dictionary for the model with perturbed tokens.
            calculate_baseline (bool): If True, computes baseline sufficiency where all tokens are zeroed/masked.
        
        Returns:
            tuple: The computed sufficiency score and baseline sufficiency score (if requested).
        """
        # Get model prediction on original input
        original_output = self.model(**original_input)
        original_prediction = F.softmax(original_output.logits, dim=-1).detach().cpu().numpy()
        full_text_probs = original_prediction.max(-1)
        full_text_class = original_prediction.argmax(-1)
        rows = np.arange(original_input["input_ids"].size(0))

        if calculate_baseline:
            baseline_input = original_input.copy()
            baseline_input['input_ids'] = torch.zeros_like(original_input['input_ids']).long().to(self._device)
            batch = baseline_input
        else:
            batch = perturbed_input

        # Get model prediction on perturbed or baseline input
        yhat = self.model(**batch)
        yhat_probs = F.softmax(yhat.logits, dim=-1).detach().cpu().numpy()
        reduced_probs = yhat_probs[rows, full_text_class]

        # Compute sufficiency for perturbed input
        sufficiency = self.sufficiency_(full_text_probs, reduced_probs)
        return sufficiency

    def normalize_sufficiency(self, sufficiency, baseline_sufficiency):
        """
        Normalizes the sufficiency score to the range [0, 1].
        
        Args:
            sufficiency (float): The raw sufficiency score.
            baseline_sufficiency (float): The baseline sufficiency score (when no perturbation).
        
        Returns:
            float: The normalized sufficiency score.
        """
        baseline_sufficiency -= 1e-4  # Avoid division by zero
        normalized_suff = np.maximum(0, (sufficiency - baseline_sufficiency) / (1 - baseline_sufficiency))
        normalized_suff = np.clip(normalized_suff, 0, 1)  # Ensure it is between 0 and 1
        return normalized_suff

    def compute_single_instance(self, explanation):
        """
        Computes Soft Normalized Sufficiency for a single instance.
        
        Args:
            explanation: An explanation object containing the text and saliency scores.
        
        Returns:
            list: The normalized sufficiency score for the instance.
        """
        original_sentences = explanation.text

        importance_scores = torch.tensor(explanation.scores).to(self._device)

        # Tokenize the sentences
        original_input = self.tokenizer(
            original_sentences, padding=True, truncation=True, 
            max_length=self.max_len, return_tensors="pt"
        )
        original_input = {key: val.to(self._device) for key, val in original_input.items()}

        # Get embeddings (using the model's outputs)
        with torch.no_grad():
            outputs = self.model(**original_input, output_hidden_states=True)
            original_embeddings = outputs.hidden_states[-1]

        # Apply soft perturbation based on importance scores
        perturbed_embeddings = self.soft_perturb(original_embeddings, importance_scores, original_input['attention_mask'])
        perturbed_input = original_input.copy()
        perturbed_input['input_ids'] = perturbed_embeddings.argmax(dim=-1)

        # Compute the sufficiency score based on the perturbed input
        baseline_sufficiency = self.compute_sufficiency(original_input, perturbed_input, calculate_baseline=True)
        sufficiency = self.compute_sufficiency(original_input, perturbed_input, calculate_baseline=False)

        # Normalize the sufficiency score
        normalized_sufficiency = self.normalize_sufficiency(sufficiency, baseline_sufficiency)
        return [normalized_sufficiency]

    def compute(self, explanations):
        """
        Computes Soft Normalized Sufficiency for all samples in the dataset.
        
        Args:
            explanations: List of explanation objects.
        
        Returns:
            float: The cumulative sufficiency score (average of all normalized sufficiency scores).
        """
        explanations = explanations if isinstance(explanations, list) else [explanations]

        all_normalized_sufficiency = []
        for explanation in explanations:
            normalized_sufficiency = self.compute_single_instance(explanation)
            all_normalized_sufficiency.extend(normalized_sufficiency)

        # Calculate the cumulative value (average) of sufficiency scores
        cumulative_sufficiency = np.mean(all_normalized_sufficiency)
        return cumulative_sufficiency