# Workflow Guide

This guide shows the direct commands for each Chartographer step. Use the README for the shortest path through the pipeline; use this file when you want to run one stage at a time, inspect outputs, or rerun a failed step.

Run commands from the repository root unless a block starts with `cd src`.

## Setup

First point Chartographer to your dataset config:

```bash
export CHARTOGRAPHER_DATASETS_FILE=/path/to/datasets.json
```

Chart reconstruction and QA regeneration use an OpenAI-compatible model by default:

```bash
export OPENAI_API_KEY=your_api_key_here
```

Prediction can also use local Hugging Face models or other API clients. If you use local model weights, set:

```bash
export CHARTOGRAPHER_MODEL_WEIGHTS_DIR=/path/to/model-weights
```

The examples below use:

```text
DATASET=my_dataset
SPLIT=dev
RECONSTRUCTION_MODEL=reconstruction-model
PREDICTION_MODEL=prediction-model
JUDGE_MODEL=judge-model
```

## End-To-End Command Order

Run these commands when you want the full pipeline without the Makefile shortcuts.

```bash
cd src

python -m pipeline.reconstruction.chart_to_code --model-name reconstruction-model --dataset my_dataset --split dev
python -m pipeline.reconstruction.render_reconstructed_charts --dataset my_dataset --split dev --round 0 --format png

# Optional: one self-refinement turn. Repeat with --round 1, then render --round 2 for a second turn.
python -m pipeline.reconstruction.diagnose_chart_issues --model-name reconstruction-model --dataset my_dataset --split dev --round 0
python -m pipeline.reconstruction.revise_chart_code --model-name reconstruction-model --dataset my_dataset --split dev --round 0
python -m pipeline.reconstruction.render_reconstructed_charts --dataset my_dataset --split dev --round 1 --format png
python -m pipeline.reconstruction.promote_revision --dataset my_dataset --split dev --round 1

python -m pipeline.reconstruction.extract_generation_assumptions --model-name reconstruction-model --dataset my_dataset --split dev
python -m pipeline.reconstruction.generate_data_modules --model-name reconstruction-model --dataset my_dataset --split dev

python -m pipeline.qa.generate_question_adapters --model-name reconstruction-model --dataset my_dataset --split dev
python -m pipeline.qa.generate_answer_modules --model-name reconstruction-model --dataset my_dataset --split dev
python -m pipeline.qa.run_qa_modules --dataset my_dataset --split dev

python -m pipeline.qa.run_seed_variants --dataset my_dataset --split dev --seed-start 0 --seed-end 9
python -m pipeline.datasets.export_chart_question_families --dataset my_dataset --split dev --output-dataset my_dataset_families --seeds 0-9
```

Then switch to the exported family dataset config:

```bash
cd ..
export CHARTOGRAPHER_DATASETS_FILE=$PWD/data/my_dataset_families/datasets.json
cd src

python -m pipeline.prediction.generate_predictions --dataset my_dataset_families --split dev --model prediction-model --max_new_tokens 1024
python -m pipeline.prediction.evaluate_predictions --dataset my_dataset_families --split dev --model prediction-model --judge-model judge-model
python -m pipeline.prediction.visualize_predictions --dataset my_dataset_families --split dev --model prediction-model
```

## Reconstruct Charts

### 1. Chart To Code

What it does: converts each chart image into Python plotting code and a JSON data file.

```bash
cd src
python -m pipeline.reconstruction.chart_to_code \
  --model-name reconstruction-model \
  --dataset my_dataset \
  --split dev
```

Common options:
- `--chart 1008`: process one chart by stem or filename
- `--max-charts N`: process only the first N charts
- `--max-workers N`: set the number of API worker threads
- `--temperature T`: set model sampling temperature

Created files:

```text
results/chartographer/{local_dir}_{split}/chart_code/reconstruction/
results/chartographer/{local_dir}_{split}/chart_data/reconstruction/
```

### 2. Extract Generation Assumptions

What it does: records assumptions from the reconstructed chart, such as labels, groups, units, and constraints needed for variant generation.

```bash
cd src
python -m pipeline.reconstruction.extract_generation_assumptions \
  --model-name reconstruction-model \
  --dataset my_dataset \
  --split dev
```

Common options:
- `--chart 1008`: process one chart
- `--max-files N`: process only the first N chart-code files
- `--max-workers N`: set the number of API worker threads

Created files:

```text
results/chartographer/{local_dir}_{split}/assumptions/
```

### 3. Generate Seed-Controlled Data Scripts

What it does: creates `generate_data(seed)` Python files that act as seed-controlled data generators for counterfactual charts.

```bash
cd src
python -m pipeline.reconstruction.generate_data_modules \
  --model-name reconstruction-model \
  --dataset my_dataset \
  --split dev
```

Common options:
- `--chart 1008`: process one chart
- `--max-files N`: process only the first N chart-code files
- `--max-workers N`: set the number of API worker threads
- `--strict-assumptions`: require an assumptions JSON file for each chart

Created files:

```text
results/chartographer/{local_dir}_{split}/generate_data/
```

### 4. Render Reconstructed Charts

What it does: runs the chart code and saves chart images.

Render the base reconstruction:

```bash
cd src
python -m pipeline.reconstruction.render_reconstructed_charts \
  --dataset my_dataset \
  --split dev \
  --round 0 \
  --format png
```

Render a seed-controlled counterfactual variant:

```bash
cd src
python -m pipeline.reconstruction.render_reconstructed_charts \
  --dataset my_dataset \
  --split dev \
  --regen-data \
  --seed 0 \
  --format png
```

Common options:
- `--chart 1008`: render one chart
- `--format png|pdf|svg`: output image format
- `--round N`: render `reconstruction` for `0`, or `revision_N` for `N > 0`
- `--regen-data --seed N`: render a seed-controlled counterfactual variant

Created files:

```text
results/chartographer/{local_dir}_{split}/images/reconstruction/
results/chartographer/{local_dir}_{split}/images/revision_N/
results/chartographer/{local_dir}_{split}/images/seed_N/
results/chartographer/{local_dir}_{split}/chart_data/seed_N/
```

## Self-Refine Reconstructions

Self-refinement improves a reconstruction over one or more turns. Each turn checks the current rendered chart, asks the model to identify issues, and creates a new revised version.

Revisions are kept separately so you can inspect them. To use the latest self-refined chart in QA and export, promote the last revision back into `reconstruction` before continuing.

Round mapping:
- `--round 0` starts from `reconstruction`
- `--round 1` starts from `revision_1`
- `revise_chart_code --round r` creates `revision_{r+1}`

### 1. Diagnose Current Reconstruction

```bash
cd src
python -m pipeline.reconstruction.diagnose_chart_issues \
  --model-name reconstruction-model \
  --dataset my_dataset \
  --split dev \
  --round 0
```

Created files:

```text
results/chartographer/{local_dir}_{split}/issues/reconstruction/
results/chartographer/{local_dir}_{split}/issues/revision_N/
```

### 2. Revise Chart Code And Data

```bash
cd src
python -m pipeline.reconstruction.revise_chart_code \
  --model-name reconstruction-model \
  --dataset my_dataset \
  --split dev \
  --round 0
```

Created files for `--round 0`:

```text
results/chartographer/{local_dir}_{split}/chart_code/revision_1/
results/chartographer/{local_dir}_{split}/chart_data/revision_1/
```

### 3. Render The Revision

```bash
cd src
python -m pipeline.reconstruction.render_reconstructed_charts \
  --dataset my_dataset \
  --split dev \
  --round 1 \
  --format png
```

For two self-refinement turns, repeat the same pattern with `--round 1`, then render `--round 2`.

### 4. Promote The Last Revision

What it does: makes `revision_N` the active `reconstruction`. By default, it also removes temporary revision files and later outputs that should be rebuilt from the promoted reconstruction.

```bash
cd src
python -m pipeline.reconstruction.promote_revision \
  --dataset my_dataset \
  --split dev \
  --round 1
```

Common options:
- `--chart 1008`: promote one chart only
- `--no-clean-revisions`: keep `revision_N` and issue files
- `--no-clean-derived`: keep assumptions, generated data scripts, QA scripts, QA JSON, seed data/images, and visualization files

After promotion, rerun:

```bash
python -m pipeline.reconstruction.extract_generation_assumptions --model-name reconstruction-model --dataset my_dataset --split dev
python -m pipeline.reconstruction.generate_data_modules --model-name reconstruction-model --dataset my_dataset --split dev
```

## QA Regeneration

### 1. Generate Question Adapters

What it does: prepares the original question so it can be reused with counterfactual chart data when the wording still makes sense.

```bash
cd src
python -m pipeline.qa.generate_question_adapters \
  --model-name reconstruction-model \
  --dataset my_dataset \
  --split dev
```

Common options:
- `--chart 1008`: process one chart
- `--max-items N`: process only N QA items
- `--max-workers N`: set the number of API worker threads
- `--overwrite`: regenerate existing adapters

Created files:

```text
results/chartographer/{local_dir}_{split}/question_adapters/
```

### 2. Generate Answer Scripts

What it does: creates executable `generate_answer(data)` Python files that compute the answer from chart data.

```bash
cd src
python -m pipeline.qa.generate_answer_modules \
  --model-name reconstruction-model \
  --dataset my_dataset \
  --split dev
```

Run question adapter generation before this step.

Common options:
- `--chart 1008`: process one chart
- `--max-items N`: process only N QA items
- `--max-workers N`: set the number of API worker threads
- `--overwrite`: regenerate existing answer scripts

Created files:

```text
results/chartographer/{local_dir}_{split}/generate_answers/
```

### 3. Run QA Logic

Run QA on the base reconstruction:

```bash
cd src
python -m pipeline.qa.run_qa_modules \
  --dataset my_dataset \
  --split dev
```

Run QA on a seed-controlled counterfactual variant:

```bash
cd src
python -m pipeline.qa.run_qa_modules \
  --dataset my_dataset \
  --split dev \
  --seed 0
```

Common options:
- `--chart 1008`: run QA for one chart

Created files:

```text
results/chartographer/{local_dir}_{split}/qa_reconstruction.json
results/chartographer/{local_dir}_{split}/qa_seed_N.json
```

### 4. Generate Counterfactual Variants For A Seed Range

What it does: renders seed-controlled counterfactual variants and runs QA for each seed.

```bash
cd src
python -m pipeline.qa.run_seed_variants \
  --dataset my_dataset \
  --split dev \
  --seed-start 0 \
  --seed-end 9
```

## Reconstruction Visualization

What it does: creates an HTML report for inspecting original charts, base reconstructions, seed-controlled counterfactual variants, and QA outputs.

```bash
cd src
python -m pipeline.reconstruction.visualize_reconstructed_charts \
  --dataset my_dataset \
  --split dev \
  --charts reconstruction seed_0
```

Common options:
- `--charts reconstruction revision_1 seed_0`: choose rendered variants
- `--out /path/to/report.html`: custom output file
- `--limit N`: visualize only N rows
- `--prefer original|generated|union`: choose which chart filenames anchor the report

Created files:

```text
results/chartographer/{local_dir}_{split}/visualize_chartographer.html
```

## Export Chart-Question Families

What it does: packages each chart-question family into a local dataset for prediction and evaluation. Each family can include the original chart, the base reconstruction, and seed-controlled counterfactual variants.

```bash
cd src
python -m pipeline.datasets.export_chart_question_families \
  --dataset my_dataset \
  --split dev \
  --output-dataset my_dataset_families \
  --seeds 0-9
```

Common options:
- `--output-split NAME`: output split name; defaults to the input split
- `--source-dataset-split NAME`: provenance value; defaults to `{dataset}_{split}`
- `--no-include-original`: skip original rows
- `--copy`: copy image/data files instead of linking them

Created files:

```text
data/{output_dataset}/{split}.jsonl
data/{output_dataset}/datasets.json
data/{output_dataset}/images/{variant}/
data/{output_dataset}/chart_data/{variant}/
```

Exported rows contain these fields:

```text
chart_id
question_id
variant
image
question
answer
chart_data
source_dataset_split
source_row_index
```

After export, switch to the exported family dataset config before running prediction:

```bash
export CHARTOGRAPHER_DATASETS_FILE=/path/to/Chartographer/data/my_dataset_families/datasets.json
```

## Run Model Predictions

### 1. Generate Predictions

What it does: asks the prediction model to answer each chart question.

```bash
cd src
python -m pipeline.prediction.generate_predictions \
  --dataset my_dataset_families \
  --split dev \
  --model prediction-model \
  --max_new_tokens 1024
```

Common options:
- `--limit N`: process the first N rows of a larger dataset and write a limited output file
- `--seed N`: base seed for local Hugging Face generation
- `--dtype auto|float16|bfloat16|float32`: local Hugging Face load dtype
- `--no-resume`: ignore existing prediction file
- `--chart 1008`: rerun predictions for one chart
- `--api_workers N`: set the number of API worker threads
- `--api_timeout_sec N`: API request timeout
- `--api_max_retries N`: API retry count

Created files:

```text
results/{dataset}/predictions/{split}/{model}.json
results/{dataset}/predictions/{split}/{model}_limitN.json
```

### 2. Evaluate Predictions

What it does: compares model predictions against dataset answers with a judge model.

```bash
cd src
python -m pipeline.prediction.evaluate_predictions \
  --dataset my_dataset_families \
  --split dev \
  --model prediction-model \
  --judge-model judge-model
```

Common options:
- `--limit N`: evaluate the limited prediction file for the same `N`
- `--chart 1008`: rerun evaluation for one chart

Created files:

```text
results/{dataset}/predictions/{split}/{model}_eval.json
results/{dataset}/predictions/{split}/{model}_limitN_eval.json
```

### 3. Visualize Predictions

What it does: creates an HTML report for inspecting one prediction model.

```bash
cd src
python -m pipeline.prediction.visualize_predictions \
  --dataset my_dataset_families \
  --split dev \
  --model prediction-model
```

Common options:
- `--predictions /path/to/predictions.json`: use an explicit prediction file
- `--eval /path/to/eval.json`: use an explicit eval file
- `--limit N`: visualize the limited prediction and evaluation files for the same `N`
- `--out /path/to/report.html`: custom output file

Created files:

```text
results/{dataset}/predictions/{split}/{model}.html
results/{dataset}/predictions/{split}/{model}_limitN.html
```

## Output Layout

```text
results/chartographer/{local_dir}_{split}/chart_code/reconstruction/
results/chartographer/{local_dir}_{split}/chart_data/reconstruction/
results/chartographer/{local_dir}_{split}/images/reconstruction/
results/chartographer/{local_dir}_{split}/assumptions/
results/chartographer/{local_dir}_{split}/generate_data/
results/chartographer/{local_dir}_{split}/question_adapters/
results/chartographer/{local_dir}_{split}/generate_answers/
results/chartographer/{local_dir}_{split}/qa_reconstruction.json
results/chartographer/{local_dir}_{split}/qa_seed_N.json
results/chartographer/{local_dir}_{split}/visualize_chartographer.html
data/{output_dataset}/{split}.jsonl
results/{dataset}/predictions/{split}/{model}.json
results/{dataset}/predictions/{split}/{model}_eval.json
results/{dataset}/predictions/{split}/{model}.html
```

## Troubleshooting

Model not found:
- The requested model name is not available from the configured provider.
- Use a valid API model name or a configured local model alias.

Answer generation cannot find question adapters:
- Run `generate_question_adapters` before `generate_answer_modules`.

Prediction and evaluation files do not match:
- If predictions were generated with `--limit N`, evaluation and visualization need the same `--limit N`.

Prediction is using the wrong dataset config:
- Reconstruction uses the source dataset config.
- Prediction on exported chart-question families should use `data/{output_dataset}/datasets.json`.
