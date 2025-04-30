## ⚡ GPU Usage

To enable GPU acceleration for explanation generation and metric computation, ensure that your model and all framework components are properly initialized on the target device (e.g., GPU or CPU). Below is an example demonstrating how to configure and initialize the framework for GPU usage:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from explainers import IntegratedGradientsExplainer
from evaluators import SoftComprehensivenessEvaluator
import torch

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

# Initialize explainer and evaluator with device
ig = IntegratedGradientsExplainer(model, tokenizer, device=device)
sc = SoftComprehensivenessEvaluator(model, tokenizer, device=device)

# Initialize framework
xai_framework = XAIFramework(
    model,
    tokenizer,
    explainers=[ig],
    evaluators=[sc],
    device=device
)
```

This setup ensures that all components run efficiently on the GPU when available, significantly accelerating explanation generation and evaluation.

## 🧩 Add Custom Objects to the API

The `EvalxNLP` framework is fully extensible. You can create and register your own explanation methods and evaluation metrics.

### Add a Custom Explainer

To implement a new explanation method, extend the `BaseExplainer` abstract class. You must implement the following:

1. **`NAME`** (class property): A unique string identifier for your explainer.
2. **`compute_feature_importance()`**: The core method responsible for computing token-level importance scores.

```python
class MyCustomExplainer(BaseExplainer):
    NAME = "my_explainer"  # Unique identifier

    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        # Add custom initialization parameters if needed

    def compute_feature_importance(
        self,
        text: str,
        target: Union[int, str],
        target_token: Optional[str] = None,
        **kwargs
    ) -> Explanation:
    """Core method implementing the explanation logic"""
        # [Implementation details here]
        return Explanation(
            text=text,
            tokens=tokens,
            scores=scores,
            explainer=self.NAME,
            # Additional metadata if required
        )
```

### Add a Custom Evaluation Metric

To add a custom evaluation metric, extend the `BaseEvaluator` abstract class. You must implement:

1. **`NAME`** (class property): A unique string identifier for your metric.
2. **`compute()`**: A method that takes a list of Explanation objects and returns the computed score.

```python
class MyCustomEvaluator(BaseEvaluator):
    NAME = "my_metric"  # Unique identifier

    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        # Add custom initialization parameters if needed

    def compute(
        self,
        explanations
    ):
        #Add custom implementation here
        return value
```
