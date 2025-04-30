from abc import ABC, abstractmethod
import torch
import pandas as pd
from typing import List, Union, Optional

class BaseEvaluator(ABC):
    """
    Abstract base class for explainers. All custom explainers must inherit from this class
    and implement the `compute` method.
    """
    
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

    @abstractmethod
    def compute(self, *args, **kwargs):
        """
        Compute the explanation. Must be implemented by subclasses.

        Returns:
            Any: The computed explanation.
        """
        pass
