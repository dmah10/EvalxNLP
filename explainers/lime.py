import pdb
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from lime.lime_text import LimeTextExplainer

from .baseexplainer import BaseExplainer
from .explanation import Explanation


class LimeExplainer(BaseExplainer):
    NAME = "LIME"

    def compute_feature_importance(
        self,
        text,
        target=1,
        target_token: Optional[Union[int, str]] = None,
        token_masking_strategy="mask",
        batch_size=4,
        show_progress=True,
        num_samples=100,
        max_samples=5000,
        **kwargs
    ):
        target_pos_idx = self.text_classifier_engine.validate_target(target)
        text = self.text_classifier_engine.validate_input(text)
        target_token_pos_idx = None

        def fn_prediction_token_ids(token_ids_sentences: List[str]):
            """Run inference on a list of strings made of token ids.

            Masked token ids are represented with 'UNKWORDZ'.
            Note that with transformers language models, results differ if tokens are masked or removed before inference.
            We let the user choose with the parameter 'token_masking_strategy'

            :param token_ids_sentences: list of strings made of token ids.
            """
            if token_masking_strategy == "mask":
                unk_substitute = str(self.text_classifier_engine.tokenizer.mask_token_id)
            elif token_masking_strategy == "remove":
                raise NotImplementedError()
            else:
                raise NotImplementedError()

            # 1. replace or remove UNKWORDZ
            token_ids_sentences = [
                s.replace("UNKWORDZ", unk_substitute) for s in token_ids_sentences
            ]
            # 2. turn tokens into input_ids
            token_ids = [
                [int(i) for i in s.split(" ") if i != ""] for s in token_ids_sentences
            ]
            masked_texts = self.text_classifier_engine.tokenizer.batch_decode(token_ids)
            # 4. forward pass on the batch
            _, logits = self.text_classifier_engine.run_inference(
                masked_texts,
                include_hidden_states=False,
                add_special_tokens=False,
                show_progress=show_progress,
                batch_size=batch_size,
            )

            return logits.softmax(-1).detach().cpu().numpy()

        def run_lime_explainer(token_ids, target_pos_idx, num_samples, lime_args):
            """
            Runs the LIME explainer on a given set of token IDs to obtain feature importance scores.

            Args:
                token_ids (List[int]): A list of token IDs representing the text to be explained.
                target_pos_idx (int): The index of the target class for which explanations are being generated.
                num_samples (int): The number of samples to use in the LIME explanation process.
                lime_args (Dict): Additional arguments to pass to the LimeTextExplainer.

            Returns:
                LimeTextExplainer.Explanation: The explanation object from LIME with feature importance scores.
            """
            explainer_args = {k: v for k, v in self.init_args.items() if k != 'task_type'}

            lime_explainer = LimeTextExplainer(bow=False, **explainer_args)

            lime_args["num_samples"] = num_samples
            return lime_explainer.explain_instance(
                " ".join([str(i) for i in token_ids]),
                fn_prediction_token_ids,
                labels=[target_pos_idx],
                num_features=len(token_ids),
                **lime_args,
            )

        
        lime_args = kwargs.get('call_args', {})

        item = self._tokenize(text, return_special_tokens_mask=True)
        token_ids = item["input_ids"][0].tolist()

        if num_samples is None:
            num_samples = min(len(token_ids) ** 2, max_samples)  # powerset size
        
        expl = run_lime_explainer(token_ids, target_pos_idx, num_samples, lime_args)

        token_scores = np.array(
        [list(dict(sorted(expl.local_exp[target_pos_idx])).values())]
        )
        token_scores[item["special_tokens_mask"].bool().cpu().numpy()] = 0.0
        token_scores = token_scores[0]

        output = Explanation(
            text=text,
            tokens=self.get_tokens(text),
            scores=token_scores,
            explainer=self.NAME,
            target_pos_idx=target_pos_idx,
            target_token_pos_idx=target_token_pos_idx,
            target=self.text_classifier_engine.model.config.id2label[target_pos_idx],
            target_token= None
        )
        return output
