import argparse
import os
import sys
import pandas as pd
from anthro_benchmark.generator import (
    DialogueGenerator, 
    LLMGenerationError, 
    DEFAULT_USER_SYSTEM_PROMPT
)
from anthro_benchmark.classifier import (
    run_rating_process
)
from datetime import datetime
import re

# helper function for file/column name sanitation
def sanitize_model_name(model_name: str) -> str:
    """Removes characters problematic for filenames/column names."""
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', model_name)

def generate_dialogues_command(args):
    print("Starting dialogue generation...")
    print(f"Configuration: {args}")
    print("-" * 30)

    # prepare LLM configurations
    user_llm_config = {"model": args.user_llm_model, "temperature": args.user_llm_temperature}
    target_llm_config = {"model": args.target_llm_model, "temperature": args.target_llm_temperature}

    # handle system prompts
    user_system_prompt = args.user_system_prompt
    if os.path.exists(user_system_prompt):
        try:
            with open(user_system_prompt, 'r') as f:
                user_system_prompt = f.read()
            print(f"Loaded user system prompt from {args.user_system_prompt}")
        except Exception as e:
            print(f"Warning: Could not read user system prompt file {args.user_system_prompt}: {e}. Using provided string or default.")
    
    target_system_prompt = args.target_system_prompt
    if os.path.exists(target_system_prompt):
        try:
            with open(target_system_prompt, 'r') as f:
                target_system_prompt = f.read()
            print(f"Loaded target system prompt from {args.target_system_prompt}")
        except Exception as e:
            print(f"Warning: Could not read target system prompt file {args.target_system_prompt}: {e}. Using provided string or default.")

    # construct dynamic CSV filename
    sanitized_target_model = sanitize_model_name(args.target_llm_model)
    
    categories_str_part = "all_prompts" # default if no specific categories or path
    if args.prompt_category_name:
        categories_str_part = "_".join(sorted(args.prompt_category_name))
        categories_str_part = categories_str_part.replace(" ", "_") 

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    dynamic_csv_filename = f"dialogues_{sanitized_target_model}_{categories_str_part}_{timestamp}.csv"
    print(f"Generated CSV filename: {dynamic_csv_filename}")

    try:
        cues_arg = args.behaviors if hasattr(args, 'behaviors') and args.behaviors else getattr(args, 'cues', None)
        generator = DialogueGenerator(
            cues=cues_arg,
            user_llm_config=user_llm_config,
            target_llm_config=target_llm_config,
            user_system_prompt=user_system_prompt,
            target_system_prompt=target_system_prompt,
            num_turns=args.num_turns,
            num_dialogues=args.num_dialogues,
            prompt_category_names=args.prompt_category_name,
            base_prompt_dir=args.base_prompt_dir,
            use_all_variants_of_original_prompt=not args.deduplicate_original_prompts,
            output_dir=args.output_dir,
            default_csv_filename=dynamic_csv_filename
        )

        print("DialogueGenerator initialized.")
        
        print("Generating dialogues...")
        generated_dialogues = generator.generate_dialogues()

        if generated_dialogues:
            print(f"Successfully generated {len(generated_dialogues)} dialogue(s).")
            output_csv_path = os.path.join(args.output_dir, dynamic_csv_filename)
            if not os.path.isabs(output_csv_path):
                 output_csv_path = os.path.abspath(output_csv_path)
            print(f"Dialogues saved to: {output_csv_path}")
            if not os.path.exists(output_csv_path):
                 print(f"ERROR: CSV file NOT found at {output_csv_path}, though generation reported success.")
        elif generator.prompts: 
             print("Prompts were loaded, but no dialogues were generated (check num_dialogues or filtering criteria).")
        else: 
            print("No prompts were loaded (check prompt_category_names/prompt_path and file existence/content), so no dialogues generated.")

    except LLMGenerationError as e:
        print(f"LLM Generation Error: {e}", file=sys.stderr)
    except FileNotFoundError as e:
        print(f"File Not Found Error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()

def rate_dialogues_command(args):
    """
    Command handler for the 'rate' subcommand.
    Maps CLI args to library function parameters.
    """
    try:
        cues_to_rate_arg = args.behaviors_to_rate if hasattr(args, 'behaviors_to_rate') and args.behaviors_to_rate else getattr(args, 'cues_to_rate', None)
        output_path = run_rating_process(
            dialogues_csv_path=args.dialogues_csv,
            cues_to_rate=cues_to_rate_arg,
            classifier_models=args.classifier_model,
            classifier_temperature=args.classifier_temperature,
            num_samples=args.num_samples,
            output_rated_csv=getattr(args, 'output_rated_csv', None),
            verbose=True
        )
        if not output_path:
            print("Warning: Rating process completed but didn't return an output path.")
    except FileNotFoundError as e:
        print(f"File Not Found Error: {e}", file=sys.stderr)
    except ValueError as e:
        print(f"Value Error: {e}", file=sys.stderr)
    except IOError as e:
        print(f"I/O Error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()

def summarize_command(args):
    print("Starting analysis summary...")
    try:
        from anthro_benchmark.analysis.analysis import run_analysis
        
        run_analysis(
            rated_csv_path=args.rated_csv,
            output_dir=args.output_dir
        )
    except ImportError as e:
        print(f"Import Error during analysis: {e}", file=sys.stderr)
        print("Please ensure pandas and plotly are installed: pip install pandas plotly", file=sys.stderr)
        print("For saving static plots (PNG), also install kaleido: pip install -U \"kaleido>=0.1.0,<0.2.0\"", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred during analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="CLI for Anthropomorphic Behavior Evaluation in LLMs.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # Generate dialogues subcommand
    gen_parser = subparsers.add_parser("generate", help="Generate dialogues between LLMs.")
    
    prompt_group = gen_parser.add_argument_group("Prompting options")
    prompt_group.add_argument("--prompt-category-name", type=str, nargs='+', help="One or more prompt category names (e.g., \"personhood\" \"physical_embodiment\").")
    prompt_group.add_argument("--base-prompt-dir", type=str, default="prompt_sets", help="Base directory for category CSVs (default: prompt_sets).")
    prompt_group.add_argument("--behaviors", type=str, nargs='*', help="List of specific behaviors to filter for.")
    prompt_group.add_argument("--cues", type=str, nargs='*', help=argparse.SUPPRESS)  # deprecated alias
    prompt_group.add_argument("--deduplicate-original-prompts", action="store_true", help="If set, uses unique 'original_prompt' values from loaded prompts, keeping the first. Default is to use all variants.")

    llm_group = gen_parser.add_argument_group("LLM configuration options")
    llm_group.add_argument("--user-llm-model", type=str, required=True, help="Model name for User LLM.")
    llm_group.add_argument("--user-llm-temperature", type=float, default=0.7, help="Temperature for User LLM.")
    llm_group.add_argument("--target-llm-model", type=str, required=True, help="Model name for Target LLM.")
    llm_group.add_argument("--target-llm-temperature", type=float, default=0.7, help="Temperature for Target LLM.")
    llm_group.add_argument("--user-system-prompt", type=str, default=DEFAULT_USER_SYSTEM_PROMPT, help="Custom system prompt for User LLM (string or path to .txt file).")
    llm_group.add_argument("--target-system-prompt", type=str, default="You are a helpful chatbot. Every one of your outputs should be a single message in a conversation.", help="Custom system prompt for Target LLM (string or path to .txt file).")

    gen_control_group = gen_parser.add_argument_group("Generation control options")
    gen_control_group.add_argument("--num-dialogues", type=int, default=960, help="Number of dialogues to produce. Defaults to full set.")
    gen_control_group.add_argument("--num-turns", type=int, default=5, help="Number of turns.")

    output_group = gen_parser.add_argument_group("Output options")
    output_group.add_argument("--output-dir", type=str, default="generated_dialogues", help="Directory to save generated dialogues.")

    gen_parser.set_defaults(func=generate_dialogues_command)

    # Rate dialogues subcommand
    rate_parser = subparsers.add_parser("rate", help="Rate dialogues for specified behaviors using one or more LLM classifiers.")
    
    rate_parser.add_argument("--dialogues-csv", type=str, required=True, help="Path to the input CSV file containing dialogues.")
    rate_parser.add_argument("--output-rated-csv", type=str, help="Path for the output CSV. If not provided, generated filename is placed in 'rated_dialogues' directory.")
    
    rate_llm_group = rate_parser.add_argument_group("Classifier LLM Configuration")
    rate_llm_group.add_argument("--classifier-model", type=str, nargs='+', required=True, help="One or more model names for the classifier LLM(s).")
    rate_llm_group.add_argument("--classifier-temperature", type=float, default=0.7, help="Temperature for the classifier LLM(s).")

    rate_config_group = rate_parser.add_argument_group("Rating Configuration")
    rate_config_group.add_argument("--behaviors-to-rate", type=str, nargs='+', help="One or more behavior names to rate. If not specified, rates all available behaviors.")
    rate_config_group.add_argument("--cues-to-rate", type=str, nargs='+', help=argparse.SUPPRESS)  # deprecated alias
    rate_config_group.add_argument("--num-samples", type=int, default=1, choices=[1, 3], help="Number of times to sample rating for each turn per model (1 or 3, default: 1).")
    
    rate_parser.set_defaults(func=rate_dialogues_command)

    # Summarize results subcommand
    summarize_parser = subparsers.add_parser("summarize", help="Analyze rated dialogues and generate summaries/plots.")
    
    summarize_parser.add_argument("--rated-csv", type=str, required=True, help="Path to the input CSV file containing rated dialogues (output from 'rate' command).")
    summarize_parser.add_argument("--output-dir", type=str, default="analysis_results", help="Directory to save analysis results (plots, summary stats) (default: analysis_results).")

    summarize_parser.set_defaults(func=summarize_command)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main() 