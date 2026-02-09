import matplotlib
matplotlib.use("Agg")

import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Includinng absolute paths to avoid issues when running from different directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

CSV_PATH = os.path.join(PROJECT_ROOT, "data", "All_Diets.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")  

MACRO_COLS = ["Protein(g)", "Carbs(g)", "Fat(g)"]
REQUIRED_COLS = ["Diet_type", "Cuisine_type", *MACRO_COLS]

def ts(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Avoid divide-by-zero by replacing 0 with NaN, then compute ratio."""
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator

def main():

    ts(f"CSV_PATH: {CSV_PATH}")
    ts(f"PLOTS_DIR: {PLOTS_DIR}")

    ts("Starting analysis")


    # Load dataset
    df = pd.read_csv(CSV_PATH)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nFound columns: {list(df.columns)}")
    ts(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    # Identify recipe/name column if present (helps scatter plot labels)
    possible_name_cols = ["Recipe_name", "Recipe", "Name", "recipe_name", "title"]
    name_col = next((c for c in possible_name_cols if c in df.columns), None)

    # Handle missing data (fill missing values with mean)
    for c in MACRO_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    macro_means = df[MACRO_COLS].mean(numeric_only=True)
    df[MACRO_COLS] = df[MACRO_COLS].fillna(macro_means)
    ts("Cleaned macro columns: coerced to numeric + filled NaN with column means")

    # Add new metrics (Protein-to-Carbs ratio and Carbs-to-Fat ratio)
    df["Protein_to_Carbs_ratio"] = safe_ratio(df["Protein(g)"], df["Carbs(g)"])
    df["Carbs_to_Fat_ratio"] = safe_ratio(df["Carbs(g)"], df["Fat(g)"])
    ts("Computed Protein_to_Carbs_ratio and Carbs_to_Fat_ratio")

    # Calculate the average macronutrient content for each diet type
    avg_macros = (
        df.groupby("Diet_type")[MACRO_COLS]
          .mean()
          .sort_values("Protein(g)", ascending=False)
    )
    ts("Computed average macros per Diet_type")

    # Find the top 5 protein-rich recipes for each diet type
    top_protein = (
        df.sort_values("Protein(g)", ascending=False)
          .groupby("Diet_type", as_index=False, group_keys=False)
          .head(5)
          .copy()
    )
    ts("Selected top 5 protein-rich recipes per Diet_type")

    highest_avg_protein_diet = avg_macros["Protein(g)"].idxmax()
    ts(f"Diet_type with highest average protein: {highest_avg_protein_diet}")

    most_common_cuisine = (
        df.groupby("Diet_type")["Cuisine_type"]
          .agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else np.nan)
    )
    ts("Computed most common Cuisine_type per Diet_type")

    # Print summary tables (good for screenshots)
    print("\n=== Average macronutrients per Diet_type ===")
    print(avg_macros.round(2))

    print("\n=== Most common cuisine per Diet_type ===")
    print(most_common_cuisine)

    # -----------------------------
    # Visualizations
    # -----------------------------
    os.makedirs(PLOTS_DIR, exist_ok=True)
    sns.set_theme()

    # Bar charts: avg macros by diet type (Protein, Carbs, Fat)
    ts("Creating bar chart: average macros by diet type")

    avg_macros_reset = avg_macros.reset_index()
    avg_melt = avg_macros_reset.melt(
        id_vars="Diet_type",
        value_vars=MACRO_COLS,
        var_name="Macronutrient",
        value_name="Average_grams"
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(data=avg_melt, x="Diet_type", y="Average_grams", hue="Macronutrient")
    plt.title("Average Macronutrient Content by Diet Type")
    plt.xlabel("Diet Type")
    plt.ylabel("Average (g)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    bar_path = os.path.join(PLOTS_DIR, "bar_avg_macros_by_diet.png")
    plt.savefig(bar_path, dpi=200)
    ts(f"Saved: {bar_path}")

    # Heatmaps to show the relationship between macronutrient content and diet types.
    # A clear interpretation is a heatmap of avg macros (diet_type x macro)
    ts("Creating heatmap: avg macros (Diet_type x Macronutrient)")

    heatmap_data = avg_macros.copy()  # index: Diet_type, cols: macros
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="viridis")
    plt.title("Heatmap: Average Macronutrients by Diet Type")
    plt.xlabel("Macronutrient")
    plt.ylabel("Diet Type")
    plt.tight_layout()
    heatmap_path = os.path.join(PLOTS_DIR, "heatmap_avg_macros_by_diet.png")
    plt.savefig(heatmap_path, dpi=200)
    ts(f"Saved: {heatmap_path}")

    # Scatter plots to display the top 5 protein-rich recipes and their distribution across different cuisines.
    # We'll use Protein vs Carbs, colored by Cuisine, faceted by Diet_type
    ts("Creating scatter plot: top 5 protein-rich recipes by cuisine and diet")

    scatter_df = top_protein.copy()

    g = sns.relplot(
        data=scatter_df,
        x="Carbs(g)",
        y="Protein(g)",
        hue="Cuisine_type",
        col="Diet_type",
        col_wrap=3,
        kind="scatter",
        height=4,
        facet_kws={"sharex": False, "sharey": False}
    )
    g.set_axis_labels("Carbs (g)", "Protein (g)")
    g.figure.suptitle("Top 5 Protein-Rich Recipes per Diet Type (Colored by Cuisine)", y=1.02)
    plt.tight_layout()
    scatter_path = os.path.join(PLOTS_DIR, "scatter_top5_protein_by_cuisine.png")
    plt.savefig(scatter_path, dpi=200, bbox_inches="tight")
    ts(f"Saved: {scatter_path}")

    cols = ["Diet_type", "Cuisine_type", "Protein(g)", "Carbs(g)", "Fat(g)", "Protein_to_Carbs_ratio", "Carbs_to_Fat_ratio"]
    if name_col:
        cols.insert(1, name_col)

    print("\n=== Top 5 protein-rich recipes per Diet_type (with ratios) ===")
    print(scatter_df[cols].round(3).reset_index(drop=True))

    ts("Analysis complete - all plots saved and summary tables printed")


if __name__ == "__main__":
    main()