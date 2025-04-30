from typing import Optional


class Explanation:
    def __init__(
        self,
        text: str,
        tokens: list,
        scores: list,
        explainer: str,
        target_pos_idx: Optional[int] = None,
        target_token_pos_idx: Optional[int] = None,
        target: Optional[str] = None,
        target_token: Optional[str] = None,
    ):
        """
        Initializes an Explanation object to store feature importance results.
        
        Parameters:
            text (str): The original input text.
            tokens (list): A list of tokens in the input text.
            scores (list): A list of feature importance scores for each token.
            explainer (str): The name of the explainer used (e.g., "Gradient (x Input)").
            helper_type (str, optional): Type of helper (e.g., token classification).
            target_pos_idx (int, optional): The index of the target position.
            target_token_pos_idx (int, optional): The index of the target token position.
            target (str, optional): The target class label.
            target_token (str, optional): The target token for token-level classification.
        """
        self.text = text  # Original input text
        self.tokens = tokens  # List of tokens in the input
        self.scores = scores  # Importance scores for each token
        self.explainer = explainer  # Name of the explainer method
        self.target_pos_idx = target_pos_idx  # Index of the target position (if applicable)
        self.target_token_pos_idx = target_token_pos_idx  # Index of the target token position
        self.target = target  # Target label for classification

    def __repr__(self):
        """
        Custom string representation of the Explanation object for easier viewing.
        """
        explanation_info = (
            f"Explanation(\n"
            f"  text={self.text},\n"
            f"  tokens={self.tokens},\n"
            f"  scores={self.scores},\n"
            f"  explainer={self.explainer},\n"
            f"  target={self.target},\n"
            f")"
        )
        return explanation_info
