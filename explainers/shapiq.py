import numpy as np
import shapiq
from typing import List, Optional, Union
from .baseexplainer import BaseExplainer
from .explanation import Explanation


class SHAPIQExplainer(BaseExplainer):
    NAME = "SHAP-I"

    def __init__(
        self,
        model,
        tokenizer,
        max_order: int = 2,
        **kwargs,
    ):
        super().__init__(model, tokenizer, **kwargs)
        self.max_order = max_order

    def model_call(self, input_texts: List[str], target_pos_idx: int) -> np.ndarray:
        """
        Calls the sentiment classification model with a list of texts.

        Args:
            input_texts (List[str]): A list of input texts.
            target_pos_idx (int): The index of the target class.

        Returns:
            np.ndarray: A vector of sentiment scores for the target class.
        """
        outputs = self.text_classifier_engine.run_inference(input_texts, include_hidden_states=False)[1]
        scores = outputs.softmax(dim=-1).detach().cpu().numpy()
        return scores[:, target_pos_idx]  # Dynamically return scores for the target class

    def game_function(
        self, coalitions: np.ndarray[bool], tokenized_input: np.ndarray[int], mask_token_id: int, target_pos_idx: int
    ) -> np.ndarray:
        """
        Computes the value of the coalitions.

        Args:
            coalitions (np.ndarray[bool]): A matrix of coalitions.
            tokenized_input (np.ndarray[int]): The tokenized input sentence.
            mask_token_id (int): The token ID used for masking.
            target_pos_idx (int): The index of the target class.

        Returns:
            np.ndarray: A vector of coalition values for the target class.
        """
        texts = []
        for coalition in coalitions:
            tokenized_coalition = tokenized_input.copy()
            tokenized_coalition[~coalition] = mask_token_id
            coalition_text = self.text_classifier_engine.tokenizer.decode(tokenized_coalition)
            texts.append(coalition_text)

        sentiments = self.model_call(texts, target_pos_idx)
        return sentiments

    def compute_feature_importance(
        self,
        text: str,
        target: Union[int, str] = 1,
        budget: int = 128,
        **kwargs,
    ) -> Explanation:
        """
        Compute feature importance using SHAPIQ.

        Args:
            text (str): The input text.
            target (Union[int, str]): The target class index or label.
            budget (int): The computational budget for the SHAPIQ approximator.

        Returns:
            Explanation: An explanation object containing feature importances.
        """
        if budget is None:
            budget = 512
        target_pos_idx = self.text_classifier_engine.validate_target(target)
        text = self.text_classifier_engine.validate_input(text)
        item = self._tokenize(text, return_special_tokens_mask=True)
        tokenized_input = np.asarray(item["input_ids"][0].tolist())
        n_players = len(tokenized_input)

        mask_token_id = self.text_classifier_engine.tokenizer.mask_token_id

        # Define the game function
        def game_function(coalitions):
            return self.game_function(coalitions, tokenized_input, mask_token_id, target_pos_idx)

        # Use SHAPIQ approximator to compute interactions
        approximator = shapiq.SHAPIQ(n=n_players, max_order=self.max_order, index="k-SII")
        sii_values = approximator.approximate(budget=budget, game=game_function)

        # Extract individual feature attributions
        token_scores = np.zeros_like(tokenized_input, dtype=float)
        for idx, value in sii_values.dict_values.items():
            if len(idx) == 1:  # Only consider single-feature contributions
                token_scores[idx[0]] = value

        # Create Explanation object
        output = Explanation(
            text=text,
            tokens=self.get_tokens(text),
            scores=token_scores,
            explainer=self.NAME,
            target_pos_idx=target_pos_idx,
            target=self.text_classifier_engine.model.config.id2label[target_pos_idx],
        )
        return output
