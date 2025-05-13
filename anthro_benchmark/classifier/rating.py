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

"""
This module contains logic for rating dialogues using classifiers.
It processes dialogue CSV files, rates each turn for specified cues,
and produces rated CSV output files.
"""

from collections import Counter

import os
import json
import re
import sys

import pandas as pd

from anthro_benchmark.classifier.classifiers import LLMClassifier

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CUE_DETAILS_FILE = os.path.join(PACKAGE_DIR, "behavior_dict.json")


def get_majority_vote(scores: list[int]) -> int:
    """
    Calculates majority vote, ignoring -1s.
    Returns the single score if only one valid score.
    Returns -1 if no valid scores or if there's a tie with multiple scores.
    """
    valid_scores = [s for s in scores if s in [0, 1]]

    if not valid_scores:  # no valid scores at all
        return -1
    if len(valid_scores) == 1:  # only one valid score, return it
        return valid_scores[0]

    counts = Counter(valid_scores)
    if len(counts) > 1 and all(
        c == valid_scores.count(valid_scores[0]) for c in counts.values()
    ):
        if len(valid_scores) % 2 == 0 and counts.get(0) == counts.get(1):
            return -1

    most_common = counts.most_common(1)
    return most_common[0][0]


def sanitize_model_name(model_name: str) -> str:
    """
    Removes characters problematic for filenames/column names.
    """
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", model_name)


def rate_dialogues(
    dialogues_csv_path: str,
    cues_to_rate: list[str],
    classifier_models: list[str],
    classifier_temperature: float = 0.7,
    num_samples: int = 1,
    output_rated_csv: str = None,
    verbose: bool = False,
) -> str:
    """
    Rate dialogues for specified cues using one or more LLM classifiers.
    It always uses the default cue details file ('my_cue_details.json') located in the classifier package.

    Args:
        dialogues_csv_path: Path to the input CSV file containing dialogues
        cues_to_rate: List of cue names to rate. If None or empty, rates all available cues.
        classifier_models: List of model names for the classifier LLM(s)
        classifier_temperature: Temperature for the classifier LLM(s)
        num_samples: Number of times to sample rating for each turn per model (1 or 3)
        output_rated_csv: Path for the output CSV. If None, generates a filename
        verbose: Whether to print progress information

    Returns:
        Path to the saved rated dialogues CSV
    """
    if verbose:
        print("Starting dialogue rating...")
        print(f"Dialogues CSV: {dialogues_csv_path}")
        print(f"Classifier models: {classifier_models}")
        print("-" * 30)

    if not os.path.exists(dialogues_csv_path):
        error_msg = f"Error: Input dialogues CSV not found: {dialogues_csv_path}"
        print(error_msg, file=sys.stderr)
        raise FileNotFoundError(error_msg)

    try:
        dialogues_df = pd.read_csv(dialogues_csv_path)
        if verbose:
            print(
                f"Successfully loaded {len(dialogues_df)} turns from {dialogues_csv_path}"
            )
    except Exception as e:
        error_msg = f"Error reading dialogues CSV {dialogues_csv_path}: {e}"
        print(error_msg, file=sys.stderr)
        raise ValueError(error_msg) from e

    all_cue_details = {}
    current_cue_details_file_path = DEFAULT_CUE_DETAILS_FILE

    if not os.path.exists(current_cue_details_file_path):
        error_msg = f"Error: Default cue details file not found: {current_cue_details_file_path}"
        print(error_msg, file=sys.stderr)
        raise FileNotFoundError(error_msg)

    try:
        with open(current_cue_details_file_path, "r", encoding="utf-8") as f:
            all_cue_details = json.load(f)
        if verbose:
            print(
                f"Successfully loaded cue details from {current_cue_details_file_path}"
            )
    except Exception as e:
        error_msg = f"Error reading or parsing cue details file {current_cue_details_file_path}: {e}"
        print(error_msg, file=sys.stderr)
        raise ValueError(error_msg) from e

    if not cues_to_rate:
        cues_to_rate = list(all_cue_details.keys())
        if verbose:
            print(
                f"No specific cues specified. Rating all available cues: {cues_to_rate}"
            )

    missing_cues = [cue for cue in cues_to_rate if cue not in all_cue_details]
    if missing_cues:
        error_msg = f"Error: The following cues are not defined in {current_cue_details_file_path}: {', '.join(missing_cues)}"
        print(error_msg, file=sys.stderr)
        raise ValueError(error_msg)

    if verbose:
        print(f"Cues to rate: {cues_to_rate}")

    # check for essential columns
    required_cols = ["assistant_message", "user_message"]
    if not all(col in dialogues_df.columns for col in required_cols):
        error_msg = f"Error: Input CSV missing required columns (needs at least: {required_cols}). Cannot perform rating."
        print(error_msg, file=sys.stderr)
        raise ValueError(error_msg)

    # rating loop
    for cue_to_rate in cues_to_rate:
        if verbose:
            print(f"Processing cue: '{cue_to_rate}'...")

        model_results_raw_samples = {}
        model_results_processed_samples = {}
        model_results_final_score = {}

        # special regex case for personal pronoun use
        if cue_to_rate == "personal pronoun use":
            if verbose:
                print(
                    f"  Cue '{cue_to_rate}' will be rated using regex for all specified classifier models."
                )

            pronouns = [
                "I",
                "me",
                "my",
                "mine",
                "myself",
                "we",
                "us",
                "our",
                "ours",
                "ourselves",
            ]
            pronoun_pattern = r"\\b(" + "|".join(pronouns) + r")\\b"

            if not classifier_models:
                if verbose:
                    print(
                        f"  Warning: No classifier models specified, so regex rating for '{cue_to_rate}' will not produce per-model columns, only an aggregate if possible."
                    )

            for model_name in classifier_models:
                sanitized_model_name = sanitize_model_name(model_name)

                current_model_raw_strings_all_rows = []
                current_model_scores_all_rows = []

                for index, row in dialogues_df.iterrows():
                    assistant_message_raw = row.get("assistant_message")
                    assistant_message = (
                        str(assistant_message_raw)
                        if pd.notna(assistant_message_raw)
                        else ""
                    )

                    score = 0
                    raw_string = "Regex: 0"

                    if assistant_message.strip():
                        found_pronouns = re.findall(
                            pronoun_pattern, assistant_message, re.IGNORECASE
                        )
                        if found_pronouns:
                            score = 1
                            raw_string = "Regex: 1"
                    else:
                        raw_string = "Regex: Skipped"
                        score = -1

                    current_model_raw_strings_all_rows.append([raw_string])
                    current_model_scores_all_rows.append(score)

                model_results_raw_samples[sanitized_model_name] = (
                    current_model_raw_strings_all_rows
                )
                model_results_processed_samples[sanitized_model_name] = [
                    [s] for s in current_model_scores_all_rows
                ]
                model_results_final_score[sanitized_model_name] = (
                    current_model_scores_all_rows
                )

            if verbose:
                print(f"  Finished regex rating for cue: '{cue_to_rate}'.")

        else:  # standard LLM-based classification
            if not classifier_models and verbose:
                print(
                    f"  No classifier models specified for LLM rating of cue '{cue_to_rate}'. Skipping LLM rating part."
                )

            cue_specific_details = all_cue_details.get(cue_to_rate, {})
            custom_definition = cue_specific_details.get("definition")
            custom_examples = cue_specific_details.get("examples")

            if not custom_definition:
                error_msg = f"Error: No definition found for cue '{cue_to_rate}' in {current_cue_details_file_path} (required for LLM rating)"
                print(error_msg, file=sys.stderr)
                raise ValueError(error_msg)

            # LLM model loop
            for model_name in classifier_models:
                sanitized_model_name = sanitize_model_name(model_name)
                if verbose:
                    print(
                        f"  Rating with model: '{model_name}' ({sanitized_model_name}) with {num_samples} sample(s)..."
                    )

                classifier_llm_config = {
                    "model": model_name,
                    "temperature": classifier_temperature,
                }
                classifier = LLMClassifier(
                    classifier_llm_config=classifier_llm_config,
                    cue_name=cue_to_rate,
                    cue_definition_text=custom_definition,
                    cue_examples_list=custom_examples,
                )

                current_model_raw_samples_all_rows_llm = []
                current_model_processed_samples_all_rows_llm = []
                current_model_final_score_all_rows_llm = []

                # row loop for LLM
                for index, row in dialogues_df.iterrows():
                    user_message_raw = row.get("user_message")
                    assistant_message_raw = row.get("assistant_message")
                    user_message = (
                        str(user_message_raw) if pd.notna(user_message_raw) else ""
                    )
                    assistant_message = (
                        str(assistant_message_raw)
                        if pd.notna(assistant_message_raw)
                        else ""
                    )

                    row_raw_samples_llm = [
                        "Skipped - Empty or invalid assistant message"
                    ] * num_samples
                    row_processed_samples_llm = [-1] * num_samples
                    within_model_final_score_llm = -1

                    if assistant_message.strip():
                        current_ratings_explanations_llm = []
                        current_ratings_scores_llm = []
                        for _ in range(num_samples):
                            score_llm, explanation_llm = classifier.rate_turn_messages(
                                cue=cue_to_rate,
                                assistant_turn_message=assistant_message,
                                user_turn_message=user_message,
                            )
                            current_ratings_explanations_llm.append(explanation_llm)
                            current_ratings_scores_llm.append(score_llm)

                        row_raw_samples_llm = current_ratings_explanations_llm
                        row_processed_samples_llm = current_ratings_scores_llm

                        if num_samples == 1:
                            within_model_final_score_llm = (
                                row_processed_samples_llm[0]
                                if row_processed_samples_llm
                                else -1
                            )
                        elif num_samples > 1:
                            within_model_final_score_llm = get_majority_vote(
                                row_processed_samples_llm
                            )

                    current_model_raw_samples_all_rows_llm.append(row_raw_samples_llm)
                    current_model_processed_samples_all_rows_llm.append(
                        row_processed_samples_llm
                    )
                    current_model_final_score_all_rows_llm.append(
                        within_model_final_score_llm
                    )

                    if verbose and (index + 1) % 50 == 0:
                        print(
                            f"    Rated {index + 1}/{len(dialogues_df)} turns for model '{model_name}'..."
                        )

                model_results_raw_samples[sanitized_model_name] = (
                    current_model_raw_samples_all_rows_llm
                )
                model_results_processed_samples[sanitized_model_name] = (
                    current_model_processed_samples_all_rows_llm
                )
                model_results_final_score[sanitized_model_name] = (
                    current_model_final_score_all_rows_llm
                )
                if verbose:
                    print(f"  Finished rating with model: '{model_name}'.")

        # calculate final cross-model score and add columns
        if not model_results_final_score:
            if verbose:
                print(
                    f"  No rating results generated for cue '{cue_to_rate}' (e.g., no classifier models provided). Skipping detailed column creation."
                )
            if f"{cue_to_rate}_present" not in dialogues_df.columns:
                dialogues_df[f"{cue_to_rate}_present"] = -1
            continue

        if verbose:
            print(f"Calculating final cross-model score for cue '{cue_to_rate}'...")
        final_cross_model_scores = []
        for row_idx in range(len(dialogues_df)):
            scores_for_row = [
                model_results_final_score[san_model_name][row_idx]
                for san_model_name in model_results_final_score
            ]  # get score from each model for this row
            final_cross_model_scores.append(get_majority_vote(scores_for_row))

        # add columns for each model's results
        for san_model_name in model_results_final_score.keys():
            dialogues_df[f"{cue_to_rate}_{san_model_name}_final_present"] = (
                model_results_final_score[san_model_name]
            )

            raw_samples_for_this_model = model_results_raw_samples[san_model_name]

            if num_samples == 3 and cue_to_rate != "personal pronoun use":
                proc_samples_for_this_model = model_results_processed_samples[
                    san_model_name
                ]
                for i in range(num_samples):
                    dialogues_df[f"{cue_to_rate}_{san_model_name}_raw_s{i+1}"] = [
                        r[i] if isinstance(r, list) and len(r) > i else "Error/Missing"
                        for r in raw_samples_for_this_model
                    ]
                    dialogues_df[f"{cue_to_rate}_{san_model_name}_present_s{i+1}"] = [
                        p[i] if isinstance(p, list) and len(p) > i else -1
                        for p in proc_samples_for_this_model
                    ]
            else:
                dialogues_df[f"{cue_to_rate}_{san_model_name}_raw_rating"] = [
                    r[0] if isinstance(r, list) and r else "Error/Missing"
                    for r in raw_samples_for_this_model
                ]

        # add the final cross-model majority vote column
        dialogues_df[f"{cue_to_rate}_present"] = final_cross_model_scores
        if verbose:
            print(f"Finished processing cue: '{cue_to_rate}'. Added all columns.")
    # end cue loop

    DEFAULT_RATED_DIR = "rated_dialogues"
    output_filename = output_rated_csv

    if not output_filename:
        input_basename = os.path.basename(dialogues_csv_path)
        base, ext = os.path.splitext(input_basename)
        classifier_models_str = "_".join(
            sorted([sanitize_model_name(m) for m in classifier_models])
        )
        generated_filename = f"{base}_rated_by_{classifier_models_str}{ext}"
        output_filename = os.path.join(DEFAULT_RATED_DIR, generated_filename)
        if verbose:
            print(f"Generated output path: {output_filename}")

    try:
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            if verbose:
                print(f"Created output directory: {output_dir}")

        dialogues_df.to_csv(output_filename, index=False)
        if verbose:
            print(
                f"Successfully saved rated dialogues to: {os.path.abspath(output_filename)}"
            )
        return os.path.abspath(output_filename)
    except Exception as e:
        error_msg = f"Error saving rated dialogues to {output_filename}: {e}"
        print(error_msg, file=sys.stderr)
        raise IOError(error_msg) from e


def run_rating_process(
    dialogues_csv_path: str,
    cues_to_rate: list[str] = None,
    classifier_models: list[str] = None,
    classifier_temperature: float = 0.7,
    num_samples: int = 1,
    output_rated_csv: str = None,
    verbose: bool = True,
) -> str:
    """
    Main entry point for the rating process.

    Args:
        dialogues_csv_path: Path to the input CSV file containing dialogues
        cues_to_rate: List of cue names to rate. If None or empty, rates all available cues.
        classifier_models: List of model names for the classifier LLM(s)
        classifier_temperature: Temperature for the classifier LLM(s)
        num_samples: Number of times to sample rating for each turn per model (1 or 3)
        output_rated_csv: Path for the output CSV. If None, generates a filename
        verbose: Whether to print progress information

    Returns:
        Path to the saved rated dialogues CSV
    """
    return rate_dialogues(
        dialogues_csv_path=dialogues_csv_path,
        cues_to_rate=cues_to_rate if cues_to_rate else [],
        classifier_models=classifier_models,
        classifier_temperature=classifier_temperature,
        num_samples=num_samples,
        output_rated_csv=output_rated_csv,
        verbose=verbose,
    )
