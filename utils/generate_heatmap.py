from collections import Counter, defaultdict
from typing import List, Dict, Optional

import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from IPython.display import display

from explanation import Explanation

def generate_explanation_heatmap(
    explanations: List[Explanation],
    omit_boundary_tokens: bool = False,
    colormap_style: str = "cool_warm",
    vmin: float = -1,
    vmax: float = 1
):
    """Generates and displays a heatmap of explanation scores.

    Args:
        explanations (List[Explanation]): List of explanation objects.
        omit_boundary_tokens (bool): Whether to remove the first and last tokens.
        colormap_style (str): The colormap style for the heatmap.
        vmin (float): Minimum value for the heatmap scale.
        vmax (float): Maximum value for the heatmap scale.
    """
    if not isinstance(explanations, list):
        explanations = [explanations]

    # Prepare data
    data = {e.explainer: e.scores for e in explanations}
    data["Token"] = explanations[0].tokens
    df = pd.DataFrame(data).set_index("Token").T

    # Remove first and last tokens if specified
    if omit_boundary_tokens:
        df = df.iloc[:, 1:-1]

    # Deduplicate column names
    col_counts = Counter(df.columns)
    seen = defaultdict(int)
    df.columns = [f"{col}_{seen[col]}" if col_counts[col] > 1 else col for col in df.columns]
    for col in df.columns:
        seen[col] += 1

    # Select colormap
    if colormap_style == "cool_warm":
        cmap = sns.diverging_palette(240, 10, as_cmap=True)
    elif colormap_style == "light_purple":
        cmap = sns.light_palette("purple", as_cmap=True)
    elif colormap_style == "reverse_purple":
        cmap = sns.light_palette("purple", as_cmap=True, reverse=True)
    elif colormap_style == "purple_centered":
        cmap = LinearSegmentedColormap.from_list("purple_gradient", ["white", "purple", "white"])
    else:
        raise ValueError(f"Invalid colormap style: {colormap_style}")

    # Plot heatmap with improved readability
    plt.figure(figsize=(12,8))  # Increased figure size #figsize=(len(df.columns) * 0.8, len(df) * 0.8)
    sns.heatmap(
        df.astype(float), 
        vmin=vmin, 
        vmax=vmax, 
        cmap=cmap, 
        annot=True, 
        fmt=".2f", 
        annot_kws={"size": 10},  # Larger annotation font
        square=False,  # Flexible cell shape
        cbar_kws={'shrink': 0.75}
    )
    plt.title("Explanation Heatmap", fontsize=14, pad=15)
    plt.xlabel("Tokens", fontsize=12, labelpad=10)
    plt.ylabel("Explainers", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate and align x-axis labels
    plt.yticks(fontsize=10)
    plt.tight_layout(pad=2.0)
    plt.show()

def show_pivot_table(
    pivot_df: pd.DataFrame,
    style: Optional[str] = "heatmap"
) -> pd.DataFrame:
    """Format pivot table into a colored table.

    Args:
        pivot_df (pd.DataFrame): The pivot table DataFrame.
        style (Optional[str]): Style to apply to the table (e.g., "heatmap").

    Returns:
        pd.DataFrame: A styled pandas DataFrame of the pivot table.
    """
    # if not style:
    #     return pivot_df.format("{:.2f}")

    if style == "heatmap":
        # Apply heatmap styling
        styled_table = pivot_df.style.background_gradient(cmap="coolwarm", axis=None).format("{:.2f}")
        display(styled_table)
        return styled_table
    else:
        raise ValueError(f"Style {style} is not supported.")