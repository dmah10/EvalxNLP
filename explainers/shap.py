import pdb
from typing import Dict, Optional, Text, Union
import logging

import numpy as np
import shap
from shap.maskers import Text as TextMasker

from .baseexplainer import BaseExplainer
from .explanation import Explanation


class SHAPExplainer(BaseExplainer):
    NAME = "SHAP"

    def __init__(
        self,
        model,
        tokenizer,
        silent: bool = True,
        algorithm: str = "partition",
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(model, tokenizer, **kwargs)
        # Initializing SHAP-specific arguments
        self.init_args["silent"] = silent
        self.init_args["algorithm"] = algorithm
        self.init_args["seed"] = seed

    def compute_feature_importance(
        self,
        text,
        target: Union[int, Text] = 1,
        target_token: Optional[Union[int, Text]] = None,
        **kwargs,
    ):
        # sanity checks
        target_pos_idx = self.text_classifier_engine.validate_target(target)
        text = self.text_classifier_engine.validate_input(text)

        # Removing 'target_option' if passed as it's not relevant here
        if 'target_option' in kwargs:
            logging.warning("The 'target_option' argument is not used in SHAPExplainer and will be removed.")
            kwargs.pop('target_option')

        # Function to compute logits for SHAP explainer
        def func(texts: np.array):
            _, logits = self.text_classifier_engine.run_inference(texts.tolist())
            # Adjust logits based on the target token position
            return logits.softmax(-1).detach().cpu().numpy()

        masker = TextMasker(self.tokenizer)
        explainer_partition = shap.Explainer(model=func, masker=masker, **self.init_args)
        shap_values = explainer_partition(text, **kwargs)
        attr = shap_values.values[0][:, target_pos_idx]
        # Tokenize the text for token-level explanation
        item = self._tokenize(text, return_special_tokens_mask=True)
        token_ids = item['input_ids'][0].tolist()
        token_scores = np.zeros_like(token_ids, dtype=float)
        # Assigning SHAP values to tokens, ignoring special tokens
        for i, (shap_value, is_special_token) in enumerate(zip(attr, item['special_tokens_mask'][0])):
            if not is_special_token:
                token_scores[i] = shap_value

        output = Explanation(
            text=text,
            tokens=self.get_tokens(text),
            scores=token_scores,
            explainer=self.NAME,
            target_pos_idx=target_pos_idx,
            target=self.text_classifier_engine.model.config.id2label[target_pos_idx],
            target_token=None
        )
        return output
