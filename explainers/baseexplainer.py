"""Explainers API"""

import warnings
from abc import ABC, abstractmethod
from utils.text_classifier_utils import TextClassifierEngine
from typing import Optional, Tuple, Union, Dict, List
from utils.generate_heatmap import generate_explanation_heatmap
from utils.saliency_utils import lp_normalize
class BaseExplainer(ABC):
    @property
    @abstractmethod
    def NAME(self):
        pass

    def __init__(
        self, model, tokenizer, **kwargs
    ):
        if model is None or tokenizer is None:
            raise ValueError("Please specify a model and a tokenizer.")

        self.init_args = kwargs

        self.text_classifier_engine= TextClassifierEngine(model, tokenizer)

    @property
    def device(self):
        return self.text_classifier_engine.model.device

    @property
    def model(self):
        return self.text_classifier_engine.model

    @property
    def tokenizer(self):
        return self.text_classifier_engine.tokenizer

    def _tokenize(self, text, **tok_kwargs):
        return self.text_classifier_engine._tokenize(text, **tok_kwargs)

    def get_tokens(self, text):
        return self.text_classifier_engine.split_input_tokens(text)

    def get_input_embeds(self, text):
        return self.text_classifier_engine.extract_input_embeddings(text)

    @abstractmethod
    def compute_feature_importance(
        self, text: str, target: int, target_token: Optional[str], **explainer_args
    ):
        pass

    def __call__(
        self,
        text: str,
        target: Union[str,int],
        target_token: Optional[str] = None,
        **explainer_args
    ):
        return self.compute_feature_importance(
            text, target, target_token, **explainer_args
        )
