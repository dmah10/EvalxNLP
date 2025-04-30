from together import Together
import pandas as pd
from pathlib import Path
from IPython.display import display, HTML
import json

class EvaluationMetricsExplainer:
    def __init__(self, api_key, model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
        """Initialize with API key and metric definitions"""
        self.model_name = model_name
        self.api_key = api_key
        self.client = Together(api_key=self.api_key)
        
        self.metric_info = {
            'Soft Comprehensiveness ↑': {
                'description': "Measures how much the model's prediction relies on important tokens by probabilistically perturbing them based on importance scores.",
                'direction': "higher is better",
                'range': "0-1 (1 = model completely relies on the important tokens)"
            },
            'Soft Sufficiency ↓': {
                'description': "Measures how well important tokens preserve the model's prediction when other tokens are perturbed.",
                'direction': "lower is better", 
                'range': "0-1 (1 = important tokens perfectly maintain the prediction)"
            },
            'FAD ↓': {
                'description': "Measures performance impact when dropping top-salient tokens, with curve steepness indicating faithfulness.",
                'direction': "lower is better",
                'range': "0-1 (0 = perfect alignment with model's true feature importance)"
            },
            'AUTPC ↓': {
                'description': "Evaluates explanation faithfulness by measuring performance drop when progressively masking top-salient tokens.",
                'direction': "lower is better",
                'range': "0-1 (0 = most faithful, performance drops immediately when critical tokens are masked)"
            },
            'AUPRC ↑': {
                'description': "Evaluates how well token importance scores match human rationales by computing precision-recall at varying thresholds.",
                'direction': "higher is better",
                'range': "0-1 (1 = perfect alignment with human rationale)"
            },
            'IOU F1 score ↑': {
                'description': "Compares explanation against human rationale by measuring token overlap ratio between top-K important tokens and ground truth.",
                'direction': "higher is better",
                'range': "0-1 (1 = perfect match with human rationale)"
            },
            'Token F1 score ↑': {
                'description': "Computes F1 score between top-K important tokens (predicted) and human rationale (ground truth).",
                'direction': "higher is better",
                'range': "0-1 (1 = perfect token-level alignment)"
            },
            'Complexity ↓': {
                'description': "Measures explanation conciseness by computing the entropy of normalized attribution scores. Lower values indicate simpler explanations focusing on few key features, while higher values suggest distributed feature importance",
                'direction': "lower is better",
                'range': "0-1"
            },
            'Sparseness ↑': {
                'description': "Quantifies explanation concentration using the Gini index. Measures whether only highly-attributed features are truly predictive.",
                'direction': "higher is better",
                'range': "0-1 (1 = perfectly sparse, 0 = uniform importance across all features)"
            }

        }

    def explain_results(self, results_data):
        """
        Process DataFrame with explainers as index and metrics as columns
        
        Args:
            results_data: DataFrame like:
                          Area under TP Curve  Complexity  ...  
                Explainer                                  
                saliency                  0.5        0.18  ...
                
        Returns:
            dict: Structured explanations
        """

        explanations = {}
        # Convert input to standardized long format
        if isinstance(results_data, pd.DataFrame):
            # DataFrame format - explainers as index, metrics as columns
            df_long = results_data.reset_index().melt(
                id_vars='Explainer',
                var_name='Metric',
                value_name='Value'
            )
        elif isinstance(results_data, dict):
            # Dict format - explainers as keys, metrics as rows
            df = pd.DataFrame.from_dict(results_data)
            df_long = df.reset_index().melt(
                id_vars='index',
                var_name='Explainer',
                value_name='Value'
            )
            df_long = df_long.rename(columns={'index': 'Metric'})
        else:
            raise ValueError("Input must be DataFrame or dictionary")
        
        # Process valid results
        valid_results = df_long.dropna(subset=['Value'])
        

        for metric, group in valid_results.groupby('Metric'):
            if metric not in self.metric_info:
                continue
                
            info = self.metric_info[metric]
            metric_explanations = {}

            for _, row in group.iterrows():
                prompt = self._create_metric_prompt(
                    metric=metric,
                    explainer=row['Explainer'],
                    value=row['Value'],
                    info=info
                )
                
                metric_explanations[row['Explainer']] = {
                    'value': float(row['Value']),
                    'interpretation': self._query_llm(prompt)
                }
            if metric_explanations:
                explanations[metric] = {
                    'definition': info['description'],
                    'direction': info['direction'],
                    'range': info['range'],
                    'explanations': metric_explanations
                }
            
                
        return explanations

    def _create_metric_prompt(self, metric, explainer, value, info):
        """Create prompt for metric explanation"""
        return f"""
        We're evaluating explanation methods using the {metric} metric.
        Context: {info['description']} (Direction: {info['direction']}, Range: {info['range']})
        
        The {explainer} method scored {value:.3f}. In 2-3 sentences:
        1. Explain what this score means
        2. Assess if this indicates good performance
        3. Compare to ideal values
        """

    def _query_llm(self, prompt):
        """Query the LLM with error handling"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.4
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Explanation unavailable: {str(e)}"

    def save_results(self, explanations, output_dir="../results/llm_explanations"):
        """Save to JSON and HTML (overwrites existing)"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save JSON
        json_path = output_path / "metrics_explanations.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(explanations, f, indent=2)
        
        # Save HTML
        html_path = output_path / "metrics_report.html"
        self._generate_html(explanations, html_path)
        
        return json_path, html_path

    def _generate_html(self, explanations, filepath):
        """Generate HTML report with universal color scheme"""
        color_scheme = {
            'background': 'transparent',  # Adapts to notebook environment
            'text': '#333333',            # Dark gray (readable on both)
            'primary': '#3f51b5',         # Indigo (works in both modes)
            'secondary': '#009688',       # Teal (good contrast)
            'card_bg': 'rgba(245, 245, 245, 0.9)',  # Slightly transparent light
            'highlight': '#ff5722',       # Deep orange for emphasis
            'border': '#e0e0e0',          # Light gray border
            'error': '#d32f2f'            # Red for errors
        }

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Metrics Report</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                    line-height: 1.6;
                    padding: 20px;
                    background-color: {color_scheme['background']};
                    color: {color_scheme['text']};
                }}
                h1 {{
                    color: {color_scheme['primary']};
                    border-bottom: 2px solid {color_scheme['secondary']};
                    padding-bottom: 10px;
                    margin-top: 0;
                }}
                h2 {{
                    color: {color_scheme['primary']};
                    margin-bottom: 8px;
                }}
                h3 {{
                    color: {color_scheme['secondary']};
                    margin: 12px 0 6px 0;
                }}
                .section {{
                    margin-bottom: 30px;
                }}
                .metric {{
                    background: {color_scheme['card_bg']};
                    padding: 16px;
                    margin: 12px 0;
                    border-radius: 8px;
                    border-left: 4px solid {color_scheme['primary']};
                    backdrop-filter: blur(2px);
                }}
                .explainer {{
                    background: rgba(255, 255, 255, 0.7);
                    padding: 12px;
                    margin: 10px 0;
                    border-radius: 6px;
                    border-left: 3px solid {color_scheme['secondary']};
                }}
                .score {{
                    font-weight: 600;
                    color: {color_scheme['highlight']};
                }}
                .interpretation {{
                    white-space: pre-wrap;
                    background: rgba(255, 255, 255, 0.5);
                    padding: 8px 12px;
                    border-radius: 4px;
                    margin-top: 8px;
                }}
                .meta {{
                    color: {color_scheme['secondary']};
                    font-size: 0.9em;
                    margin-bottom: 12px;
                }}
                .error {{
                    color: {color_scheme['error']};
                    padding: 10px;
                    border-radius: 4px;
                    background: rgba(211, 47, 47, 0.1);
                }}
            </style>
        </head>
        <body>
            <h1>Evaluation Metrics Report</h1>
        """
        
        for metric, data in explanations.items():
            html_content += f"""
            <div class="metric">
                <div class="metric-name">{metric}</div>
                <div class="metric-meta">
                    <strong>Definition:</strong> {data['definition']}<br>
                    <strong>Direction:</strong> {data['direction']} | 
                    <strong>Range:</strong> {data['range']}
                </div>
            """
            
            for explainer, details in data['explanations'].items():
                html_content += f"""
                <div class="explainer">
                    <div class="explainer-name">{explainer}</div>
                    <div><strong>Score:</strong> {details['value']:.3f}</div>
                    <div class="interpretation"><strong>Analysis:</strong> {details['interpretation']}</div>
                </div>
                """
            
            html_content += "</div>"
        
        html_content += "</body></html>"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def display_results(self, explanations):
        """Display with complete style isolation"""
        from IPython.display import display, HTML
        import base64
        import tempfile
        
        # Generate HTML
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            self._generate_html(explanations, Path(f.name))
            html_content = Path(f.name).read_text(encoding='utf-8')
        
        # Create sandboxed iframe
        iframe = f"""
        <iframe srcdoc='{html_content.replace("'", "&apos;")}'
                style="width:100%; height:600px; border:none;"
                sandbox="allow-scripts">
        </iframe>
        """
        display(HTML(iframe))
