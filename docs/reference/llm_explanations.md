Feature attribution explanations, particularly those involving high-dimensional features, can be difficult for lay users to interpret. To address this, our tool integrates a module that leverages **Large Language Models (LLMs)** to generate natural language descriptions to help users interpret both:

1. The importance scores from various explanation methods.
2. The evaluation metric scores.

## Key Benefits

- **Improved Interpretability**: Textual descriptions bridge the gap between numerical outputs and actionable insights, helping users better understand model behavior and evaluation outcomes.
- **User-Friendly Accessibility**: Natural language explanations lower the barrier for non-technical users, making the framework more approachable without requiring deep expertise in explainability methods or an understanding of the metrics.
- **Enhanced Usability**: The combination of visual heatmaps with textual explanations offers a more accessible and holistic view of the model’s decision-making process.

The LLM is integrated into the framework via an **API**, allowing for seamless generation of textual descriptions whenever needed. This ensures the feature is both **flexible** and **scalable**, adapting to various tasks and datasets.
