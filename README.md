# AnthroBench

A library for generating, rating, and analyzing dialogues to evaluate anthropomorphic behaviors in LLMs, developed in [AnthroBench: A Multi-turn Evaluation of Anthropomorphic Behaviours in Large Language Models](https://arxiv.org/abs/2502.07077).

## Structure

The library is organized into several key packages and modules:

- `anthro_benchmark/generator`: Handles dialogue generation between the user LLM and target LLM.
- `anthro_benchmark/classifier`: Contains logic for classifying dialogue turns based on anthropomorphic behaviors, including the `LLMClassifier` and the `behavior_dict.json` behavior definitions.
- `anthro_benchmark/core`: Core utilities, including `llm_client.py` for interacting with various LLM APIs.
- `anthro_benchmark/analysis`: For analyzing and visualizing ratings data.
- `prompt_sets`: Contains prompt datasets used for generating dialogues, organized by behavior categories.
- `anthro_eval_cli.py`: The command-line interface script.
- `setup.py`: For package installation and distribution.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/google-deepmind/anthro-benchmark.git
    cd anthro-benchmark
    ```

2.  Create and activate a Python virtual environment (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate 
    ```

3.  Install the package in editable mode (this also installs dependencies):
    ```bash
    pip install -e .
    ```

## API keys setup

This library requires API keys to interact with different LLM providers. You need to set up your keys as environment variables:

```bash
# OpenAI API key 
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic API key 
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Google API key 
export GOOGLE_API_KEY="your-google-api-key"

# Mistral API key
export MISTRAL_API_KEY="your-mistral-api-key"
```

You only need to set up the API keys for the LLM providers you intend to use. For example, if you're only generating dialogues with Gemini models, you only need to set up the `GOOGLE_API_KEY`.

## Prompt sets

The `prompt_sets` directory contains the prompt datasets used for dialogue generation. The primary file is:

- `first_turns.csv`: The main dataset containing all prompts. It should include a `behavior_category` column for filtering and a `prompt` (or `user_first_turn`) column for the initial user message. Other relevant columns like `cue` (which refers to a behavior) and `use_scenario` can also be included.

### Behavior categories

Prompts in `first_turns.csv` can be organized by a `behavior_category` column. There are four categories available:

- `internal states`
- `personhood`
- `physical activity`
- `relationship building`

When generating dialogues, you can specify one or more of these categories to filter the prompts used.

## Command-line interface (`anthro-eval`)

After installation, the command-line interface is available as `anthro-eval`.

### 1. Generating dialogues

Generate dialogues using prompts filtered by behavior categories:

```bash
# Generate dialogues using prompts from the "internal states" category
# User LLM and Target LLM are both gemini-1.5-flash
anthro-eval generate --user-llm-model "gemini-1.5-flash" --target-llm-model "gemini-1.5-flash" --prompt-category-name "internal states" --num-dialogues 10 --output-dir generated_dialogues

# Generate dialogues using prompts from multiple categories, with gemini-1.0-pro as the target
anthro-eval generate --user-llm-model "gemini-1.5-flash" --target-llm-model "gemini-1.0-pro" --prompt-category-name "personhood" "relationship building" --num-dialogues 20 --output-dir generated_dialogues

# Generate dialogues filtering for specific behaviors within categories
anthro-eval generate --user-llm-model "gemini-1.5-flash" --target-llm-model "gemini-1.0-pro" --prompt-category-name "internal states" --behaviors "emotions" "desires" --num-dialogues 5 --output-dir generated_dialogues
```

The system loads prompts from `prompt_sets/first_turns.csv` and filters them based on the specified `--prompt-category-name`.

### 2. Rating dialogues

Rate generated dialogues for anthropomorphic behaviors. Behaviors are defined in `anthro_benchmark/classifier/behavior_dict.json`.

```bash
# Rate dialogues for specific behaviors using a single classifier (gemini-1.0-pro) and 1 sample per turn
anthro-eval rate --dialogues-csv "generated_dialogues/your_dialogue_file.csv" --classifier-model "gemini-1.0-pro" --behaviors-to-rate "empathy" "desires" --num-samples 1

# Rate dialogues using multiple classifier models (gemini-1.0-pro and gemini-1.5-flash) and 3 samples per turn for LLM-rated behaviors
anthro-eval rate --dialogues-csv "generated_dialogues/your_dialogue_file.csv" --classifier-model "gemini-1.5-pro" "gemini-1.5-flash" --behaviors-to-rate "empathy" "validation" --num-samples 3

# Rate dialogues for all available behaviors defined in behavior_dict.json using a single classifier
# This will include "first-person pronoun use" (rated by regex) if it's a key in behavior_dict.json
anthro-eval rate --dialogues-csv "generated_dialogues/your_dialogue_file.csv" --classifier-model "gemini-1.5-flash"
```

- You can specify one or more `--classifier-model` names. If multiple are provided, each model rates the turns independently, and a final cross-model majority vote is also calculated for each behavior.
- `--num-samples` can be `1` or `3`. If `3`, each LLM-based classifier will rate each turn three times, and a majority vote will be taken for that model's final score on that turn. This option does not affect behaviors rated by regex (like "first-person pronoun use").
- If `--behaviors-to-rate` is not specified, all behaviors from `behavior_dict.json` are rated.
- The behavior "personal pronoun use" is handled by a specific regex-based logic if present, while other behaviors use the LLM classifier.
- Rated dialogues are saved in the `rated_dialogues/` directory by default.

### 3. Analyzing results

Analyze the rated dialogues to generate summaries and plots:

```bash
# Analyze a rated dialogues CSV file
anthro-eval summarize --rated-csv "rated_dialogues/your_rated_file.csv" --output-dir analysis_results
```

Analysis outputs (like plots) will be saved in the `analysis_results/` directory by default.

## License

Apache-2.0 License
