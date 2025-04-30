import json
from pathlib import Path
from IPython.display import display, HTML
from together import Together

class LLMExplanationGenerator:
    def __init__(self, api_key, model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
        """
        Initialize with API key and model name, plus output directory setup.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.client = Together(api_key=self.api_key)
        self.output_dir = Path("../results/llm_explanations")
        self.output_dir.mkdir(exist_ok=True)

    def generate_and_save_explanations(self, exps, output_format="both"):
        """
        Generate explanations and save to file(s) with display options.
        
        Args:
            exps: List of explanation objects
            output_format: "json", "html", or "both" (default)
            
        Returns:
            Tuple: (list of explanations, path to saved files)
        """
        explanations = self.generate_explanation(exps)
        
        saved_files = []
        
        # JSON output (always created for programmatic use)
        json_path = self.output_dir / "explanations.json"
        self._save_to_json(explanations, json_path)
        saved_files.append(json_path)
        
        if output_format in ("html", "both"):
            html_path = self.output_dir / "explanations.html"
            self._save_to_html(explanations, html_path)
            saved_files.append(html_path)
            print(f"file saved in {html_path}")
        return explanations, saved_files

    def _save_to_json(self, explanations, filepath):
        """Save explanations to JSON with metadata."""
        output_data = {
            "metadata": {
                "model": self.model_name,
                "num_explanations": len(explanations)
            },
            "explanations": explanations
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

    def _save_to_html(self, explanations, filepath):
        """Generate human-readable HTML report."""
        # First generate all the explanation divs
        explanation_divs = []
        for exp in explanations:
            div = f"""
            <div class="explanation">
                <h3>{exp['method'].upper()}</h3>
                <span style='color:black'> Sentence: {exp['text']} </span><br>
                <span style='color:black'> Model Label: {exp['target']} </span>
                <pre>{exp['explanation'].replace('Interpretation of Saliency Scores', '')}</pre>
            </div>
            """
            explanation_divs.append(div)
        
        # Then build the full HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Explanations Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }}
                .explanation {{ 
                    background: #f8f9fa;
                    border-left: 4px solid #4e79a7;
                    padding: 15px;
                    margin-bottom: 20px;
                    border-radius: 4px;
                }}
                h3 {{ color: #2c3e50; margin-top: 0; }}
                .meta {{ color: #7f8c8d; font-size: 0.9em; }}
                pre {{ white-space: pre-wrap; }}
            </style>
        </head>
        <body>
            <h1>Model Explanations Report</h1>
            <div class="meta">
                Model: {self.model_name}<br>
            </div>
            <hr>
            {''.join(explanation_divs)}
        </body>
        </html>
        """
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
    # def _save_to_html(self, explanations, filepath):
    #     """Generate human-readable HTML report."""
    #     html_content = f"""
    #     <!DOCTYPE html>
    #     <html>
    #     <head>
    #         <title>Model Explanations Report</title>
    #         <style>
    #             body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }}
    #             .explanation {{ 
    #                 background: #f8f9fa;
    #                 border-left: 4px solid #4e79a7;
    #                 padding: 15px;
    #                 margin-bottom: 20px;
    #                 border-radius: 4px;
    #             }}
    #             h3 {{ color: #2c3e50; margin-top: 0; }}
    #             .meta {{ color: #7f8c8d; font-size: 0.9em; }}
    #             pre {{ white-space: pre-wrap; }}
    #         </style>
    #     </head>
    #     <body>
    #         <h1>Model Explanations Report</h1>
    #         <div class="meta">
    #             Model: {self.model_name}<br>
    #         </div>
    #         <hr>
    #         {"".join([f"""
    #             <div class="explanation">
    #                 <h3>{exp['method'].upper()} (Target: {exp['target']})</h3>
    #                 <pre>{exp['explanation'].replace('Interpretation of Saliency Scores', '')}</pre>
    #             </div>
    #             """ for exp in explanations])}
    #     </body>
    #     </html>
    #     """
        
    #     with open(filepath, 'w', encoding='utf-8') as f:
    #         f.write(html_content)

    def display_explanations(self, explanations):
        """Display explanations in notebook with formatting."""
        for exp in explanations:
            clean_explanation = exp['explanation'].replace('Interpretation of Saliency Scores', '')
            display(HTML(f"""
            <div style='
                # background: #f5f5f5;
                padding: 12px;
                margin: 12px 0;
                border-radius: 4px;
                border-left: 4px solid #4e79a7;
                color:white;
            '>
                <h4 style='margin-top:0;color:white;'>
                    {exp['method'].upper()} <span style='color:white'></span>
                </h4>
                <span style='color:white'> Sentence: {exp['text']} </span><br>
                <span style='color:white'> Model Label: {exp['target']} </span>
                <pre style='white-space:pre-wrap;margin-bottom:0;'>{clean_explanation}</pre>
            </div>
            """))
   
    def generate_explanation(self, exps):
        """
        Generate textual explanations for a list of explanations (exps), with explicit method attribution.
        """
        explanations = []

         # Handle case where exps is a dictionary of {method: [explanation_objects]}
        if isinstance(exps, dict):
            for method, exp_list in exps.items():
                # exp_list = exp_list if isinstance(exp_list, list) else [exp_list]
                for exp in exp_list:
                    prompt = self._create_prompt(exp)
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=500,
                            temperature=0.7,
                        )
                        
                        explanation = response.choices[0].message.content.strip()
                        explanations.append({
                            "method": method,  # Use the dictionary key as method
                            "explanation": explanation,
                            "target": exp.target,  # Assuming exp has a target attribute
                            "text": exp.text
                        })
                    except Exception as e:
                        print(e)
                        continue
        # Handle case where exps is a single explanation or list
        else:
            exp_list = [exps] if not isinstance(exps, list) else exps
            for exp in exp_list:
                prompt = self._create_prompt(exp)
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=500,
                        temperature=0.7,
                    )
                    
                    explanation = response.choices[0].message.content.strip()
                    explanations.append({
                        "method": exp.explainer,  # Use the explainer attribute
                        "explanation": explanation,
                        "target": exp.target,
                        "text": exp.text
                    })
                except Exception as e:
                    print(e)
                    continue
        
        return explanations
        # for exp in exps:
        #     print(exp)
        #     prompt = self._create_prompt(exp)
        #     response = self.client.chat.completions.create(
        #         model=self.model_name,
        #         messages=[{"role": "user", "content": prompt}],
        #         max_tokens=500,
        #         temperature=0.7,
        #     )
            
        #     explanation = response.choices[0].message.content.strip()
        #     explanations.append({
        #         "method": exp.explainer,
        #         "explanation": explanation,
        #         "target": exp.target
        #     })
        
        # return explanations

    def _create_prompt(self, exp):
        """
        Create a prompt that explicitly requests method-specific explanations.
        """
        prompt = f"""
        You are analyzing {exp.explainer} explanations for a text classification model.
        The target class is '{exp.target}'.
        
        Given these token importance scores from {exp.explainer}, explain:
        1. How are these scores calculated?
        2. How to interpret them to with respect to model's dccision?

        Tokens: {exp.tokens}
        Scores: {exp.scores}

        Provide concise explanation specifically for {exp.explainer} method:
        """
        return prompt

    