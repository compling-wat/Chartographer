PYTHON ?= python
SRC := src

# Optional workflow settings. DATASET, SPLIT, RECONSTRUCTION_MODEL, PREDICTION_MODEL, and JUDGE_MODEL are required per target.
SEED ?= 0
SEED_START ?= 0
SEED_END ?= 9
FORMAT ?= png
ROUND ?= 0
REVISION_ROUNDS ?= 0
MAX_NEW_TOKENS ?= 1024
LIMIT ?=
CHART ?=
OUTPUT_DATASET ?=
OUTPUT_SPLIT ?=
FAMILY_SEEDS ?= 0-9

require-%:
	@if [ -z "$($*)" ]; then echo "Missing required variable: $*"; exit 2; fi
LIMIT_ARG := $(if $(LIMIT),--limit $(LIMIT),)
CHART_ARG := $(if $(CHART),--chart $(CHART),)
OUTPUT_SPLIT_ARG := $(if $(OUTPUT_SPLIT),--output-split $(OUTPUT_SPLIT),)

.PHONY: help reconstruction-workflow qa-workflow seed-workflow prediction-workflow export-family-dataset predict evaluate visualize-predictions reconstruct assumptions data-modules render diagnose-charts revise-charts promote-revision render-seed visualize-reconstruction answer-modules question-adapters qa qa-seed seed-variants

help:
	@echo "Chartographer workflow targets"
	@echo "  make reconstruction-workflow DATASET=... SPLIT=... RECONSTRUCTION_MODEL=reconstruction-model [CHART=...] [REVISION_ROUNDS=2]"
	@echo "  make qa-workflow DATASET=... SPLIT=... RECONSTRUCTION_MODEL=reconstruction-model SEED=0 [CHART=...]"
	@echo "  make seed-workflow DATASET=... SPLIT=... SEED_START=0 SEED_END=9"
	@echo "  make prediction-workflow DATASET=... SPLIT=... PREDICTION_MODEL=prediction-model JUDGE_MODEL=judge-model [CHART=...] [LIMIT=...]"
	@echo "  make export-family-dataset DATASET=... SPLIT=... OUTPUT_DATASET=... [FAMILY_SEEDS=0-9]"
	@echo ""
	@echo "Individual steps"
	@echo "  make reconstruct DATASET=... SPLIT=... RECONSTRUCTION_MODEL=reconstruction-model [CHART=...]"
	@echo "  make assumptions DATASET=... SPLIT=... RECONSTRUCTION_MODEL=reconstruction-model [CHART=...]"
	@echo "  make data-modules DATASET=... SPLIT=... RECONSTRUCTION_MODEL=reconstruction-model [CHART=...]"
	@echo "  make render DATASET=... SPLIT=... [ROUND=0] [CHART=...]"
	@echo "  make diagnose-charts DATASET=... SPLIT=... RECONSTRUCTION_MODEL=reconstruction-model [ROUND=0] [CHART=...]"
	@echo "  make revise-charts DATASET=... SPLIT=... RECONSTRUCTION_MODEL=reconstruction-model [ROUND=0] [CHART=...]"
	@echo "  make promote-revision DATASET=... SPLIT=... ROUND=1 [CHART=...]"
	@echo "  make render-seed DATASET=... SPLIT=... SEED=0 [CHART=...]"
	@echo "  make answer-modules DATASET=... SPLIT=... RECONSTRUCTION_MODEL=reconstruction-model [CHART=...]"
	@echo "  make question-adapters DATASET=... SPLIT=... RECONSTRUCTION_MODEL=reconstruction-model [CHART=...]"
	@echo "  make qa DATASET=... SPLIT=... [CHART=...]"
	@echo "  make qa-seed DATASET=... SPLIT=... SEED=0 [CHART=...]"
	@echo "  make seed-variants DATASET=... SPLIT=... SEED_START=0 SEED_END=9"
	@echo "  make predict DATASET=... SPLIT=... PREDICTION_MODEL=prediction-model [CHART=...] [LIMIT=...]"
	@echo "  make evaluate DATASET=... SPLIT=... PREDICTION_MODEL=prediction-model JUDGE_MODEL=judge-model [CHART=...] [LIMIT=...]"
	@echo "  make visualize-predictions DATASET=... SPLIT=... PREDICTION_MODEL=prediction-model [CHART=...] [LIMIT=...]"


reconstruction-workflow: require-DATASET require-SPLIT require-RECONSTRUCTION_MODEL
	@if ! expr "$(REVISION_ROUNDS)" : '^[0-9][0-9]*$$' >/dev/null; then echo "REVISION_ROUNDS must be a non-negative integer"; exit 2; fi
	$(MAKE) reconstruct DATASET=$(DATASET) SPLIT=$(SPLIT) RECONSTRUCTION_MODEL=$(RECONSTRUCTION_MODEL) CHART=$(CHART)
	$(MAKE) render DATASET=$(DATASET) SPLIT=$(SPLIT) CHART=$(CHART) FORMAT=$(FORMAT)
	@round=0; while [ $$round -lt $(REVISION_ROUNDS) ]; do \
		next_round=$$((round + 1)); \
		echo "Self-refinement turn $$next_round/$(REVISION_ROUNDS)"; \
		$(MAKE) diagnose-charts DATASET=$(DATASET) SPLIT=$(SPLIT) RECONSTRUCTION_MODEL=$(RECONSTRUCTION_MODEL) ROUND=$$round CHART=$(CHART); \
		$(MAKE) revise-charts DATASET=$(DATASET) SPLIT=$(SPLIT) RECONSTRUCTION_MODEL=$(RECONSTRUCTION_MODEL) ROUND=$$round CHART=$(CHART); \
		$(MAKE) render DATASET=$(DATASET) SPLIT=$(SPLIT) ROUND=$$next_round CHART=$(CHART) FORMAT=$(FORMAT); \
		round=$$next_round; \
	done
	@if [ "$(REVISION_ROUNDS)" -gt 0 ]; then \
		$(MAKE) promote-revision DATASET=$(DATASET) SPLIT=$(SPLIT) ROUND=$(REVISION_ROUNDS) CHART=$(CHART); \
	fi
	$(MAKE) assumptions DATASET=$(DATASET) SPLIT=$(SPLIT) RECONSTRUCTION_MODEL=$(RECONSTRUCTION_MODEL) CHART=$(CHART)
	$(MAKE) data-modules DATASET=$(DATASET) SPLIT=$(SPLIT) RECONSTRUCTION_MODEL=$(RECONSTRUCTION_MODEL) CHART=$(CHART)

qa-workflow: require-DATASET require-SPLIT require-RECONSTRUCTION_MODEL
	$(MAKE) question-adapters DATASET=$(DATASET) SPLIT=$(SPLIT) RECONSTRUCTION_MODEL=$(RECONSTRUCTION_MODEL) CHART=$(CHART)
	$(MAKE) answer-modules DATASET=$(DATASET) SPLIT=$(SPLIT) RECONSTRUCTION_MODEL=$(RECONSTRUCTION_MODEL) CHART=$(CHART)
	$(MAKE) qa DATASET=$(DATASET) SPLIT=$(SPLIT) CHART=$(CHART)
	$(MAKE) render-seed DATASET=$(DATASET) SPLIT=$(SPLIT) SEED=$(SEED) CHART=$(CHART) FORMAT=$(FORMAT)
	$(MAKE) qa-seed DATASET=$(DATASET) SPLIT=$(SPLIT) SEED=$(SEED) CHART=$(CHART)
	$(MAKE) visualize-reconstruction DATASET=$(DATASET) SPLIT=$(SPLIT) SEED=$(SEED)

seed-workflow: require-DATASET require-SPLIT
	$(MAKE) seed-variants DATASET=$(DATASET) SPLIT=$(SPLIT) SEED_START=$(SEED_START) SEED_END=$(SEED_END)

export-family-dataset: require-DATASET require-SPLIT require-OUTPUT_DATASET
	cd $(SRC) && $(PYTHON) -m pipeline.datasets.export_chart_question_families --dataset $(DATASET) --split $(SPLIT) --output-dataset $(OUTPUT_DATASET) --seeds $(FAMILY_SEEDS) $(OUTPUT_SPLIT_ARG)

prediction-workflow: require-DATASET require-SPLIT require-PREDICTION_MODEL require-JUDGE_MODEL
	$(MAKE) predict DATASET=$(DATASET) SPLIT=$(SPLIT) PREDICTION_MODEL=$(PREDICTION_MODEL) CHART=$(CHART) LIMIT=$(LIMIT) MAX_NEW_TOKENS=$(MAX_NEW_TOKENS)
	$(MAKE) evaluate DATASET=$(DATASET) SPLIT=$(SPLIT) PREDICTION_MODEL=$(PREDICTION_MODEL) JUDGE_MODEL=$(JUDGE_MODEL) CHART=$(CHART) LIMIT=$(LIMIT)
	$(MAKE) visualize-predictions DATASET=$(DATASET) SPLIT=$(SPLIT) PREDICTION_MODEL=$(PREDICTION_MODEL) CHART=$(CHART) LIMIT=$(LIMIT)

reconstruct: require-DATASET require-SPLIT require-RECONSTRUCTION_MODEL
	cd $(SRC) && $(PYTHON) -m pipeline.reconstruction.chart_to_code --model-name $(RECONSTRUCTION_MODEL) --dataset $(DATASET) --split $(SPLIT) $(CHART_ARG)

assumptions: require-DATASET require-SPLIT require-RECONSTRUCTION_MODEL
	cd $(SRC) && $(PYTHON) -m pipeline.reconstruction.extract_generation_assumptions --model-name $(RECONSTRUCTION_MODEL) --dataset $(DATASET) --split $(SPLIT) $(CHART_ARG)

data-modules: require-DATASET require-SPLIT require-RECONSTRUCTION_MODEL
	cd $(SRC) && $(PYTHON) -m pipeline.reconstruction.generate_data_modules --model-name $(RECONSTRUCTION_MODEL) --dataset $(DATASET) --split $(SPLIT) $(CHART_ARG)

render: require-DATASET require-SPLIT
	cd $(SRC) && $(PYTHON) -m pipeline.reconstruction.render_reconstructed_charts --dataset $(DATASET) --split $(SPLIT) --round $(ROUND) --format $(FORMAT) $(CHART_ARG)

diagnose-charts: require-DATASET require-SPLIT require-RECONSTRUCTION_MODEL
	cd $(SRC) && $(PYTHON) -m pipeline.reconstruction.diagnose_chart_issues --model-name $(RECONSTRUCTION_MODEL) --dataset $(DATASET) --split $(SPLIT) --round $(ROUND) $(CHART_ARG)

revise-charts: require-DATASET require-SPLIT require-RECONSTRUCTION_MODEL
	cd $(SRC) && $(PYTHON) -m pipeline.reconstruction.revise_chart_code --model-name $(RECONSTRUCTION_MODEL) --dataset $(DATASET) --split $(SPLIT) --round $(ROUND) $(CHART_ARG)

promote-revision: require-DATASET require-SPLIT
	cd $(SRC) && $(PYTHON) -m pipeline.reconstruction.promote_revision --dataset $(DATASET) --split $(SPLIT) --round $(ROUND) $(CHART_ARG)

render-seed: require-DATASET require-SPLIT
	cd $(SRC) && $(PYTHON) -m pipeline.reconstruction.render_reconstructed_charts --dataset $(DATASET) --split $(SPLIT) --regen-data --seed $(SEED) --format $(FORMAT) $(CHART_ARG)

visualize-reconstruction: require-DATASET require-SPLIT
	cd $(SRC) && $(PYTHON) -m pipeline.reconstruction.visualize_reconstructed_charts --dataset $(DATASET) --split $(SPLIT) --charts reconstruction seed_$(SEED)

answer-modules: require-DATASET require-SPLIT require-RECONSTRUCTION_MODEL
	cd $(SRC) && $(PYTHON) -m pipeline.qa.generate_answer_modules --model-name $(RECONSTRUCTION_MODEL) --dataset $(DATASET) --split $(SPLIT) $(CHART_ARG)

question-adapters: require-DATASET require-SPLIT require-RECONSTRUCTION_MODEL
	cd $(SRC) && $(PYTHON) -m pipeline.qa.generate_question_adapters --model-name $(RECONSTRUCTION_MODEL) --dataset $(DATASET) --split $(SPLIT) $(CHART_ARG)

qa: require-DATASET require-SPLIT
	cd $(SRC) && $(PYTHON) -m pipeline.qa.run_qa_modules --dataset $(DATASET) --split $(SPLIT) $(CHART_ARG)

qa-seed: require-DATASET require-SPLIT
	cd $(SRC) && $(PYTHON) -m pipeline.qa.run_qa_modules --dataset $(DATASET) --split $(SPLIT) --seed $(SEED) $(CHART_ARG)

seed-variants: require-DATASET require-SPLIT
	cd $(SRC) && $(PYTHON) -m pipeline.qa.run_seed_variants --dataset $(DATASET) --split $(SPLIT) --seed-start $(SEED_START) --seed-end $(SEED_END)

predict: require-DATASET require-SPLIT require-PREDICTION_MODEL
	cd $(SRC) && $(PYTHON) -m pipeline.prediction.generate_predictions --dataset $(DATASET) --split $(SPLIT) --model $(PREDICTION_MODEL) --max_new_tokens $(MAX_NEW_TOKENS) $(CHART_ARG) $(LIMIT_ARG)

evaluate: require-DATASET require-SPLIT require-PREDICTION_MODEL require-JUDGE_MODEL
	cd $(SRC) && $(PYTHON) -m pipeline.prediction.evaluate_predictions --dataset $(DATASET) --split $(SPLIT) --model $(PREDICTION_MODEL) --judge-model $(JUDGE_MODEL) $(CHART_ARG) $(LIMIT_ARG)

visualize-predictions: require-DATASET require-SPLIT require-PREDICTION_MODEL
	cd $(SRC) && $(PYTHON) -m pipeline.prediction.visualize_predictions --dataset $(DATASET) --split $(SPLIT) --model $(PREDICTION_MODEL) $(CHART_ARG) $(LIMIT_ARG)
