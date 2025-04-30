import math
from typing import List, Optional, Tuple, Union
import logging

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers.tokenization_utils_base import BatchEncoding
from explanation import Explanation

class TextClassifierEngine():
    """
    Simplified helper class for text classification tasks, managing tokenization, embedding extraction, and inference.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @property
    def labels(self) -> List[int]:
        return self.model.config.id2label

    def tokenize_input(self, text: str, return_dict: bool = False) -> Union[List[str], dict]:
        """
        Tokenizes input text and optionally returns a dictionary of token IDs and tokens.
        """
        encoded = self._tokenize(text)
        tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        if return_dict:
            return {idx: (id_, tok) for idx, (id_, tok) in enumerate(zip(encoded["input_ids"][0], tokens))}
        return tokens

    def extract_input_embeddings(self, text: str) -> torch.Tensor:
        """
        Retrieves input embeddings for the provided text.
        """
        encoded = self._tokenize(text)
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        embeddings = self._get_embeddings_from_ids(encoded["input_ids"][0])
        embeddings= embeddings.unsqueeze(0)
        return embeddings

    def _tokenize(self, text: str, **kwargs) -> BatchEncoding:
        return self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, **kwargs)

    def _get_embeddings_from_ids(self, ids) -> torch.Tensor:
        return self.model.get_input_embeddings()(ids)

    def split_input_tokens(self, text: str, **kwargs) -> List[str]:
        encoded = self._tokenize(text)
        input_length = encoded["attention_mask"].sum()
        ids = encoded["input_ids"][0][:input_length]
        return self.tokenizer.convert_ids_to_tokens(ids, **kwargs)
    
    def _forward_with_input_embeds(
        self,
        input_embeds,
        attention_mask,
        batch_size=8,
        show_progress=False,
        output_hidden_states=False,
    ):
        input_len = input_embeds.shape[0]
        n_batches = math.ceil(input_len / batch_size)
        input_batches = torch.tensor_split(input_embeds, n_batches)
        mask_batches = torch.tensor_split(attention_mask, n_batches)

        outputs = list()
        for emb, mask in tqdm(
            zip(input_batches, mask_batches),
            total=n_batches,
            desc="Batch",
            leave=False,
            disable=not show_progress,
        ):
            out = self.model(
                inputs_embeds=emb,
                attention_mask=mask,
                output_hidden_states=output_hidden_states,
            )
            outputs.append(out)

        logits = torch.cat([o.logits for o in outputs])
        return outputs, logits
    
    def run_inference(
        self,
        text: Union[str, List[str], Tuple[str, str]],
        batch_size: int = 8,
        show_progress: bool = False,
        use_embeddings: bool = False,
        include_hidden_states: bool = True,
        **kwargs,
    ):
        """
        Runs inference on input text and returns model outputs and logits.
        """
        if not isinstance(text, list):
            text = [text]

        num_batches = math.ceil(len(text) / batch_size)
        batches = np.array_split(text, num_batches)
        results = []
       
        for batch in tqdm(batches, total=num_batches, desc="Running Inference", disable=not show_progress):
            encoded = self._tokenize(batch.tolist(), padding="longest", **kwargs)
            encoded = {k: v.to(self.model.device) for k, v in encoded.items()}

            if use_embeddings:
                ids = encoded.pop("input_ids")
                emb = self._get_embeddings_from_ids(ids)
                output = self.model(inputs_embeds=emb, **encoded, output_hidden_states=include_hidden_states)
            else:
                output = self.model(**encoded, output_hidden_states=include_hidden_states)
            results.append(output)

        logits = torch.cat([res.logits for res in results], dim=0)
        return results, logits

    def predict_class(self, text: str, return_as_dict: bool = True):
        """
        Predicts class probabilities for the provided input text.
        """
        _, logits = self.run_inference(text, include_hidden_states=False)
        scores = logits[0].softmax(-1)

        if return_as_dict:
            return {self.model.config.id2label[idx]: score.item() for idx, score in enumerate(scores)}
        return scores

    def validate_target(self, target):
        if isinstance(target, str) and target not in self.model.config.label2id:
            raise ValueError(f"Invalid target '{target}'. Valid labels: {list(self.model.config.label2id.keys())}")
        if isinstance(target, int) and target not in self.model.config.id2label:
            raise ValueError(f"Invalid target '{target}'. Valid indices: {list(self.model.config.id2label.keys())}")
        return self.model.config.label2id.get(target, target)

    def validate_input(self, text):
        if not isinstance(text, (str, tuple, list)):
            raise ValueError("Input must be a string, tuple, or list.")
        sep_token = getattr(self.tokenizer, 'sep_token', '[SEP]')
        if isinstance(text, str):
            return [text]
        elif isinstance(text, tuple) and len(text) == 2:
            return [f"{text[0]} {sep_token} {text[1]}"]
        return text
