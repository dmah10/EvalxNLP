
    # datasets/movie_reviews.py

from datasets import load_dataset
from transformers import PreTrainedTokenizer
from typing import List, Dict, Tuple, Any
import numpy as np

NONE_RATIONALE = []  

class MovieReviews:
    NAME = "MovieReviews"
    def __init__(self, tokenizer: PreTrainedTokenizer):
        """
        Initialize the dataset and tokenizer.
        """
        self.dataset = load_dataset("movie_rationales")
        self.tokenizer = tokenizer
        self.classes = [0, 1]
        
        # Set up train, validation, and test splits
        self.train_dataset = self.dataset["train"]
        self.validation_dataset = self.dataset["validation"]
        self.test_dataset = self.dataset["test"]
    def __len__(self):
        # We use the TEST_SET as default
        return self.len()

    def len(self, split_type: str = 'test'):
        if split_type == 'train':
            return len(self.train_dataset)
        elif split_type == 'validation':
            return len(self.validation_dataset)
        elif split_type == 'test':
            return len(self.test_dataset)
        else:
            raise ValueError(
                f"{split_type} not supported as split_type. Specify one among:  train, validation or test."
            )

    def _get_item(self, idx: int, split_type: str = "test") -> Dict:
        """
        Retrieve an item by index from the specified dataset split.
        """
        if isinstance(idx, int):
            if split_type == 'train':
                item_idx = self.train_dataset[idx]
            elif split_type == 'validation':
                item_idx = self.validation_dataset[idx]
            elif split_type == 'test':
                item_idx = self.test_dataset[idx]
            else:
                raise ValueError(
                    f"{split_type} not supported as split_type. Specify one among: TRAIN_SET, VALIDATION_SET or TEST_SET."
                )
            return item_idx
        elif isinstance(idx, dict):
            return idx
        else:
            raise ValueError()
    
    def _get_offset_rationale(self, text, text_rationales):
        tokenizer = self.tokenizer
        rationale_offsets = []
        for text_rationale in text_rationales:
            print(f"text------------ {text}")
            print(f"text rationale------------ {text_rationale}")
            start_i = text.index(text_rationale)
            end_i = start_i + len(text_rationale)
            rationale_encoded_text = tokenizer.encode_plus(
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

    
    def _get_text(self, idx, split_type: str = 'test'):
        item_idx = self._get_item(idx, split_type)
        text = item_idx["review"]
        text = text.replace("\n", " ")
        return text
    
    def _get_ground_truth(self, idx, split_type: str = 'test'):
        item_idx = self._get_item(idx, split_type)
        label = item_idx["label"]
        return label
    
    def _get_rationale_one_hot_encoding(self, offsets, rationale_offsets, len_tokens):
        rationale = np.zeros(len_tokens)

        for rationale_offset in rationale_offsets:
            if rationale_offset in offsets:
                rationale[offsets.index(rationale_offset)] = 1
                # Test what this function is returning
        return rationale

    def _get_rationale(self, idx, split_type: str = 'test', rationale_union=True):
        
        item_idx = self._get_item(idx, split_type)
        text = self._get_text(item_idx)

        tokenizer = self.tokenizer
        encoded_text = tokenizer.encode_plus(
            text, return_offsets_mapping=True, return_attention_mask=False
        )
        tokens = tokenizer.convert_ids_to_tokens(encoded_text["input_ids"])
        offsets = encoded_text["offset_mapping"]

        rationale_field_name = "evidences"

        # Movie rationales are defined for the ground truth label
        rationale_label = self._get_ground_truth(idx, split_type)

        rationale_by_label = [NONE_RATIONALE for c in self.classes]

        if rationale_field_name in item_idx:
            text_rationales = item_idx[rationale_field_name]
            print(f"printing rationsee {text_rationales}")
            rationale_offsets = self._get_offset_rationale(text, text_rationales)
            if len(text_rationales) > 0 and isinstance(text_rationales, list):
                # It is a list of lists
                if rationale_union:
                    # We get the union of the rationales.
                    rationale_offsets = [t1 for t in rationale_offsets for t1 in t]
                    rationale_by_label[
                        rationale_label
                    ] = self._get_rationale_one_hot_encoding(
                        offsets, rationale_offsets, len(tokens)
                    ).astype(
                        int
                    )

                else:
                    return rationale_by_label
            else:

                rationale_by_label[
                    rationale_label
                ] = self._get_rationale_one_hot_encoding(
                    offsets, rationale_offsets, len(tokens)
                ).astype(
                    int
                )
         # Here we ensure the output is a single 1D array
        if rationale_union:
            # If rationale_union is True, return the single unified rationale as a 1D array
            # Filter out empty lists before using zip
            non_empty_rationale_by_label = [r for r in rationale_by_label if len(r)>0]  # Remove empty lists
            if non_empty_rationale_by_label:
                final_rationale = [int(any(each)) for each in zip(*non_empty_rationale_by_label)]  # Union of all rationales
            else:
                final_rationale = []  # If no valid rationale exists, return an empty list
            return final_rationale
        else:
            # Otherwise, return the rationale for the specific label (may be a list of lists if rationale_union is False)
            return rationale_by_label
    
    
    def get_instance(self, idx: int, split_type: str = "test", combine_rationales: bool = True) -> Dict[str, Any]:
        """
        Retrieves a single review instance with tokens, rationale, and label.
        """
        item_idx = self._get_item(idx, split_type)
        text = self._get_text(item_idx)
        tokens = [self.tokenizer.cls_token] + self.tokenizer.tokenize(text) + [self.tokenizer.sep_token]
        
        # Create rationale mask and retrieve label
        rationale = self._get_rationale(item_idx, split_type, combine_rationales)
        true_label = self._get_ground_truth(item_idx, split_type)
        
        return {
            "text": text,
            "tokens": tokens,
            "rationale": rationale,
            "label": true_label,
        }
    
    def _create_rationale_mask(self, text: str, text_rationales: List[str], combine_rationales: bool) -> List[int]:
        """
        Converts rationale text spans to a one-hot mask for tokens.
        """
        encoded = self.tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=True)
        offsets = encoded["offset_mapping"]
        mask = np.zeros(len(offsets), dtype=int)
        
        rationale_offsets = self._get_rationale_offsets(text, text_rationales)
        if combine_rationales:
            combined_offsets = [offset for spans in rationale_offsets for offset in spans]
            mask = self._mark_rationale_tokens(offsets, combined_offsets)
        else:
            for span in rationale_offsets:
                mask = self._mark_rationale_tokens(offsets, span)
                
        return mask.tolist()
    
    def _get_rationale_offsets(self, text: str, text_rationales: List[str]) -> List[List[Tuple[int, int]]]:
        """
        Finds the start and end offsets of rationale spans within the text.
        """
        rationale_offsets = []
        for rationale_text in text_rationales:
            start_idx = text.find(rationale_text)
            if start_idx != -1:
                end_idx = start_idx + len(rationale_text)
                encoded_rationale = self.tokenizer.encode_plus(
                    text[start_idx:end_idx], return_offsets_mapping=True, add_special_tokens=False
                )
                offsets = [(start + start_idx, end + start_idx) for start, end in encoded_rationale["offset_mapping"]]
                rationale_offsets.append(offsets)
            else:
                print(f"Warning: Rationale '{rationale_text}' not found in text.")
        return rationale_offsets
    
    def _mark_rationale_tokens(self, token_offsets: List[Tuple[int, int]], rationale_offsets: List[Tuple[int, int]]) -> np.ndarray:
        """
        Marks tokens within the rationale spans as 1 in the mask.
        """
        mask = np.zeros(len(token_offsets), dtype=int)
        for token_idx, (token_start, token_end) in enumerate(token_offsets):
            for rationale_start, rationale_end in rationale_offsets:
                if token_start >= rationale_start and token_end <= rationale_end:
                    mask[token_idx] = 1
                    break
        return mask

