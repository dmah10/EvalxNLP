import torch
from captum.attr import InputXGradient, Saliency
from typing import Optional, Tuple, Union
from .baseexplainer import BaseExplainer
from .explanation import Explanation
from utils.text_classifier_utils import TextClassifierEngine

class InputXGradientExplainer(BaseExplainer):
    NAME = "G"

    def __init__(
        self,
        model,
        tokenizer,
        device="cpu",
        multiply_by_inputs: bool = True,
        **kwargs,
    ):
        super().__init__(model, tokenizer, **kwargs)

        self.multiply_by_inputs = multiply_by_inputs
        self._device = device

        if self.multiply_by_inputs:
            self.NAME += "xI"
        else:
            self.NAME = 'Saliency'

    def compute_feature_importance(
        self,
        text: Union[str, Tuple[str, str]],
        target: Union[int, str] = 1,
        target_token: Optional[Union[int, str]] = None,
        **kwargs,
    ):
        def func(input_embeds):
            outputs = self.text_classifier_engine.model(
                inputs_embeds=input_embeds, attention_mask=item["attention_mask"]
            )
            logits = outputs.logits
            return logits

        target_pos_idx = self.text_classifier_engine.validate_target(target)
        target_token_pos_idx = None
        text = self.text_classifier_engine.validate_input(text)

        item = self._tokenize(text)
        item = {k: v.to(self._device) for k, v in item.items()}
        input_len = item["attention_mask"].sum().item()
        dl = (
            InputXGradient(func, **self.init_args)
            if self.multiply_by_inputs
            else Saliency(func, **self.init_args)
        )

        inputs = self.get_input_embeds(text)

        attr = dl.attribute(inputs, target=target_pos_idx, **kwargs)
        attr = attr[0, :input_len, :].detach()

        # pool over hidden size
        attr = attr.sum(-1).cpu().numpy()
       
        output = Explanation(
            text=text,
            tokens=self.get_tokens(text),
            scores=attr,
            explainer=self.NAME,
            target_pos_idx=target_pos_idx,
            target_token_pos_idx=target_token_pos_idx,
            target=self.text_classifier_engine.model.config.id2label[target_pos_idx],
            target_token= None
        )
        return output
