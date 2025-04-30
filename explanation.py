from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Explanation:
    """Generic class to represent an Explanation"""

    text: str
    tokens: str
    scores: np.array
    explainer: str
    target_pos_idx: int
    target_token_pos_idx: Optional[int] = None
    target: Optional[str] = None
    target_token: Optional[str] = None
    rationale: Optional[np.array] = None