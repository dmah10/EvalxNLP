import numpy as np

class MovieRationalesProcessor:
    NONE_RATIONALE = []
    classes = [0, 1]

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _get_offset_rationale(self, text, text_rationales):
        rationale_offsets = []
        for text_rationale in text_rationales:
            start_i = text.index(text_rationale)
            end_i = start_i + len(text_rationale)
            rationale_encoded_text = self.tokenizer.encode_plus(
                text[start_i:end_i],
                return_offsets_mapping=True,
                return_attention_mask=False,
            )
            rationale_token_offset = [
                (s + start_i, e + start_i)
                for (s, e) in rationale_encoded_text["offset_mapping"]
                if (s == 0 and e == 0) == False
            ]
            rationale_offsets.append(rationale_token_offset)
        return rationale_offsets

    def _get_rationale_one_hot_encoding(self, offsets, rationale_offsets, len_tokens):
        rationale = np.zeros(len_tokens)
        for rationale_offset in rationale_offsets:
            if rationale_offset in offsets:
                rationale[offsets.index(rationale_offset)] = 1
        return rationale

    def _get_rationale(self, text, label, rationales_array, split_type='test', rationale_union=True):
        encoded_text = self.tokenizer.encode_plus(
            text, return_offsets_mapping=True, return_attention_mask=False
        )
        tokens = self.tokenizer.convert_ids_to_tokens(encoded_text["input_ids"])
        offsets = encoded_text["offset_mapping"]

        rationale_label = label
        rationale_by_label = [self.NONE_RATIONALE for c in self.classes]

        text_rationales = rationales_array
        rationale_offsets = self._get_offset_rationale(text, text_rationales)
        if len(text_rationales) > 0 and isinstance(text_rationales, list):
            if rationale_union:
                rationale_offsets = [t1 for t in rationale_offsets for t1 in t]
                rationale_by_label[rationale_label] = self._get_rationale_one_hot_encoding(
                    offsets, rationale_offsets, len(tokens)
                ).astype(int)
            else:
                return rationale_by_label
        else:
            rationale_by_label[rationale_label] = self._get_rationale_one_hot_encoding(
                offsets, rationale_offsets, len(tokens)
            ).astype(int)

        if rationale_union:
            non_empty_rationale_by_label = [r for r in rationale_by_label if len(r) > 0]
            if non_empty_rationale_by_label:
                final_rationale = [int(any(each)) for each in zip(*non_empty_rationale_by_label)]
            else:
                final_rationale = []
            return final_rationale
        else:
            return rationale_by_label

    def process_dataset(self, input_texts, labels, rationales):
        texts = [text.replace("\n", " ") for text in input_texts]
        rationale_arr = [rationale.tolist() for rationale in rationales]
        processed_rationales = [
            self._get_rationale(text, label, rationale)
            for text, label, rationale in zip(texts, labels, rationale_arr)
        ]
        return processed_rationales
