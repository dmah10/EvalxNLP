import numpy as np
NONE_RATIONALE = []
class HateSpeechProcessor:
    def __init__(self,tokenizer):
        """
        Initialize the TextProcessor class.

        Args:
            input_texts (list of list of str): Array of arrays containing tokenized text.
            labels (list of dict): List of dictionaries containing 'label' keys with arrays of labels.
            rationales (list of list of float): Array of arrays containing rationale scores.
        """
        self.tokenizer= tokenizer
        self.classes = [0, 1, 2]
    def process_input_texts(self, input_texts):
        """
        Join each inner array in input_texts into a single string.

        Returns:
            list of str: Processed input_texts.
        """
        return [" ".join(text_array) for text_array in input_texts]

    def process_labels(self, labels):
        """
        Find the most frequent label for each inner array in labels.

        Returns:
            list of int: Processed labels.
        """
        return [int(np.bincount(label['label']).argmax()) for label in labels]

    def _get_rationale(self, text, labels, rationales,rationale_union=True):
        
        label= labels['label']
        unique_labels, counts = np.unique(label, return_counts=True)
        rationale_label = unique_labels[np.argmax(counts)] 

        if len(rationales) > 0:
            rationale = [any(each) for each in zip(*rationales)]
            rationale = [int(each) for each in rationale]
            
        else:
            rationale = rationales
        
        token_rationale= self.get_true_rationale_from_words_to_tokens(text, rationale)
        return token_rationale

    def get_true_rationale_from_words_to_tokens(
        self, word_based_tokens: list[str], words_based_rationales: list[int]
    ):
        token_rationale = []
        for t, rationale_t in zip(word_based_tokens, words_based_rationales):
            converted_token = self.tokenizer.encode(t)[1:-1]
            for token_i in converted_token:
                token_rationale.append(rationale_t)
        return token_rationale

    def process_rationales(self, rationales):
        """
        Return the rationales as-is or apply any necessary processing.

        Returns:
            list of list of float: Processed rationales.
        """
        processed_rationales = []
        for rationale in rationales:
            if len(rationale) > 0:
                # Compute the union of the current rationale
                union = [any(each) for each in zip(*rationale)]
                union = [int(each) for each in union]
                processed_rationales.append(union)
            else:
                # If the rationale is empty, append it as-is
                processed_rationales.append(rationale)
        return processed_rationales

    def process_dataset(self,input_texts, labels, rationales):
        """
        Process input_texts, labels, and rationales.

        Returns:
            tuple: (processed_input_texts, processed_labels, processed_rationales)
        """
        processed_input_texts = self.process_input_texts(input_texts)
        processed_labels = self.process_labels(labels)
        # processed_rationales= self._get_rationale(input_texts[10], labels[10], rationales[10])
        
        processed_rationales = [
            self._get_rationale(text, label, rationale)
            for text, label, rationale in zip(input_texts, labels, rationales)
        ]
        return processed_input_texts, processed_labels, processed_rationales