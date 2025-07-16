# Copyright 2025 The Anthropomorphism Benchmark Project Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import traceback
from typing import Dict, List, Tuple, Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


pio.templates.default = "plotly_white"

CATEGORY_MAPPING = {
    "internal states": ["desires", "emotions", "agency"],
    "personhood": [
        "personal history",
        "personal relationships",
        "sentience",
        "first-person prounoun use",
    ],
    "physical embodiment": [
        "physical embodiment",
        "physical movement",
        "sensory input",
    ],
    "relationship building": [
        "empathy",
        "validation",
        "relatability",
        "explicit relationship status",
    ],
}


def load_data(
    rated_csv_path: str,
) -> Tuple[pd.DataFrame, Dict[str, List[str]], List[str]]:
    """Loads rated data and identifies cue columns."""

    df = pd.read_csv(rated_csv_path)

    category_mapping = CATEGORY_MAPPING

    cue_present_columns = [
        col
        for col in df.columns
        if col.endswith("_present") and "_s" not in col and "_final_present" not in col
    ]

    if not cue_present_columns:
        raise ValueError(
            f"No '*_present' columns found in {rated_csv_path}. Cannot perform analysis."
        )

    print(f"Found rating columns: {cue_present_columns}")

    all_mapped_cues = set(cue for cues in category_mapping.values() for cue in cues)
    found_cue_prefixes = set(col.replace("_present", "") for col in cue_present_columns)

    if not all_mapped_cues.issubset(found_cue_prefixes):
        missing_in_df = all_mapped_cues - found_cue_prefixes
        print(
            f"Warning: Cues defined in mapping but not found in CSV's *_present columns: {missing_in_df}"
        )
    if not found_cue_prefixes.issubset(all_mapped_cues):
        missing_in_map = found_cue_prefixes - all_mapped_cues
        print(
            f"Warning: Cues found in CSV (*_present columns) but not defined in category mapping: {missing_in_map}"
        )

    return df, category_mapping, cue_present_columns


def add_category_counts(
    df: pd.DataFrame,
    category_mapping: Dict[str, List[str]],
    cue_present_columns: List[str],
) -> pd.DataFrame:
    """Adds row-wise counts for each category based on cue presence."""
    print("Adding row-wise category counts...")
    df_analysis = df.copy()
    found_cue_prefixes = {col.replace("_present", "") for col in cue_present_columns}

    for category, cues_in_category in category_mapping.items():
        category_col_name = f"{category}_count"
        relevant_cols = [
            f"{cue}_present" for cue in cues_in_category if cue in found_cue_prefixes
        ]

        if not relevant_cols:
            print(
                f"  Skipping category '{category}': No corresponding '*_present' columns found."
            )
            df_analysis[category_col_name] = 0
            continue

        # sum only valid ratings (0 or 1), treat -1 as 0 for the sum
        df_analysis[category_col_name] = (
            df_analysis[relevant_cols]
            .applymap(lambda x: x if x in [0, 1] else 0)
            .sum(axis=1)
        )
        print(f"  Added column: {category_col_name}")

    return df_analysis


def calculate_summary_stats(
    df: pd.DataFrame,
    category_mapping: Dict[str, List[str]],
    cue_present_columns: List[str],
) -> Dict[str, Any]:
    """Calculates overall cue percentages and category totals."""
    print("Calculating summary statistics...")
    summary = {"cue_percentages": {}, "category_totals": {}}
    found_cue_prefixes = {col.replace("_present", "") for col in cue_present_columns}

    # percentages
    for col in cue_present_columns:
        cue_name = col.replace("_present", "")
        valid_ratings = df[col][df[col].isin([0, 1])]  # Filter out -1 (errors/skipped)
        if len(valid_ratings) == 0:
            percentage = 0.0
            print(f"  Cue '{cue_name}': No valid ratings found.")
        else:
            percentage = (
                valid_ratings.sum() / len(valid_ratings)
            ) * 100  # Sum is count of 1s
        summary["cue_percentages"][cue_name] = round(percentage, 2)
        print(
            f"  Cue '{cue_name}': {percentage:.2f}% present ({valid_ratings.sum()} / {len(valid_ratings)} valid turns)"
        )

    # category totals (sum of all '1's for cues in that category across all valid turns)
    for category, cues_in_category in category_mapping.items():
        category_total = 0
        relevant_cols = [
            f"{cue}_present" for cue in cues_in_category if cue in found_cue_prefixes
        ]
        if relevant_cols:
            # sum only 1s across all relevant columns and rows
            category_total = (
                df[relevant_cols].applymap(lambda x: 1 if x == 1 else 0).sum().sum()
            )
        summary["category_totals"][category] = int(
            category_total
        )  # ensure integer count
        print(f"  Category '{category}': Total count = {category_total}")

    return summary


def plot_cue_percentages(cue_percentages: Dict[str, float], output_dir: str):
    """Creates a bar chart of cue percentages."""
    if not cue_percentages:
        print("No cue percentages to plot.")
        return

    print("Generating cue percentages bar chart...")
    cues = list(cue_percentages.keys())
    percentages = list(cue_percentages.values())

    fig = go.Figure(
        [go.Bar(x=cues, y=percentages, text=percentages, textposition="auto")]
    )
    fig.update_layout(
        title="Percentage of messages where behavior occurs",
        xaxis_title="Behavior",
        yaxis_title="Percentage (%)",
        yaxis_range=[0, 100],
    )

    plot_path_png = os.path.join(output_dir, "cue_percentages.png")
    plot_path_html = os.path.join(output_dir, "cue_percentages.html")

    try:
        # try to save PNG first
        try:
            fig.write_image(plot_path_png)
            print(f"  Saved plot to: {plot_path_png}")
        except Exception as e:
            print(f"  Error saving PNG plot: {e}")
            print(
                '  To save PNGs, install kaleido: pip install -U "kaleido>=0.1.0,<0.2.0"'
            )

            # fallback to HTML if PNG fails
            fig.write_html(plot_path_html)
            print(f"  Saved HTML plot to: {plot_path_html}")
    except Exception as e:
        print(f"  Error saving plot: {e}")


def plot_category_radar(category_totals: Dict[str, int], output_dir: str):
    """Creates a radar chart of category totals."""
    if not category_totals or len(category_totals) < 3:
        print(
            f"Skipping radar plot: Need at least 3 categories with totals, found {len(category_totals)}."
        )
        return

    print("Generating category totals radar chart...")
    categories = list(category_totals.keys())
    totals = list(category_totals.values())

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=totals + [totals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Total Cue Counts",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(totals) * 1.1 if totals else 1])
        ),
        showlegend=False,
        title="Total counts of behaviors per category",
    )

    plot_path_png = os.path.join(output_dir, "category_radar.png")
    plot_path_html = os.path.join(output_dir, "category_radar.html")

    try:
        try:
            fig.write_image(plot_path_png)
            print(f"  Saved plot to: {plot_path_png}")
        except Exception as e:
            print(f"  Error saving PNG plot: {e}")
            print(
                '  To save PNGs, install kaleido: pip install -U "kaleido>=0.1.0,<0.2.0"'
            )

            # fallback to HTML if PNG fails
            fig.write_html(plot_path_html)
            print(f"  Saved HTML plot to: {plot_path_html}")
    except Exception as e:
        print(f"  Error saving plot: {e}")


def run_analysis(
    rated_csv_path: str,
    output_dir: str = "analysis_results",
    category_mapping_path: str = None,
):
    """Main function to run the analysis pipeline."""
    print("\n--- Starting Analysis ---")
    print(f"Rated CSV: {rated_csv_path}")
    print(f"Output Directory: {output_dir}")
    print("Using hardcoded category mapping")

    try:
        # create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        df, category_mapping, cue_present_columns = load_data(rated_csv_path)

        df_analysis = add_category_counts(df, category_mapping, cue_present_columns)

        # calculate overall statistics
        summary_stats = calculate_summary_stats(
            df_analysis, category_mapping, cue_present_columns
        )

        # create visualizations
        plot_cue_percentages(summary_stats["cue_percentages"], output_dir)
        plot_category_radar(summary_stats["category_totals"], output_dir)

        # save the enhanced dataframe with category counts
        enhanced_csv_path = os.path.join(output_dir, "analysis_with_categories.csv")
        df_analysis.to_csv(enhanced_csv_path, index=False)
        print(f"Saved enhanced dataframe with category counts to: {enhanced_csv_path}")

        # save summary stats to JSON
        summary_json_path = os.path.join(output_dir, "summary_stats.json")
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_stats, f, indent=2)
        print(f"Saved summary statistics to: {summary_json_path}")

        print("\n--- Analysis Complete ---")
        print(f"All results saved to: {os.path.abspath(output_dir)}")

    except Exception as e:
        print(f"Error during analysis: {e}")
        traceback.print_exc()
