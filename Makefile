.PHONY: help simulate-realistic train-realistic train-swanlab experiment-realistic quick-simulate quick-train check format lint type test

UV ?= uv
UV_CACHE_DIR ?= /tmp/uv-cache

DATA_DIR ?= data/simulated/realistic
RUN_NAME ?= realistic-sim
RUN_DIR ?= runs/$(RUN_NAME)

SEED ?= 20260517
CLUSTERS ?= 4
TRAIN_PATIENTS ?= 3000
VAL_PATIENTS ?= 1000
TEST_PATIENTS ?= 1000
MIN_VISITS ?= 3
MAX_VISITS ?= 16
FOLLOWUP_DAYS ?= 1095
SIM_HIDDEN_SIZE ?= 128
SIM_LATENT_DIM ?= 8
ATTENTION_LAYERS ?= 4
ATTENTION_HEADS ?= 4
CENSORING_RATE ?= 0.45
WEIBULL_SHAPE ?= 1.3
X_LOW ?= -8
X_HIGH ?= 8
BETA_LOW ?= -1.8
BETA_HIGH ?= 1.8

EPOCHS ?= 80
WARMUP_EPOCHS ?= 20
BATCH_SIZE ?= 64
LEARNING_RATE ?= 3e-4
ENCODER_HIDDEN_DIM ?= 64
DECODER_HIDDEN_DIM ?= 64
MODEL_LATENT_DIM ?= 16
N_LAYERS ?= 1
DROPOUT ?= 0.0
SAVE_ARTIFACTS ?= all

SWANLAB_PROJECT ?= TRAILS
SWANLAB_EXPERIMENT ?= $(RUN_NAME)
SWANLAB_MODE ?=
SWANLAB_FLAGS ?=

SIMULATE_ARGS = \
	--clusters $(CLUSTERS) \
	--min-visits $(MIN_VISITS) \
	--max-visits $(MAX_VISITS) \
	--followup-days $(FOLLOWUP_DAYS) \
	--hidden-size $(SIM_HIDDEN_SIZE) \
	--latent-dim $(SIM_LATENT_DIM) \
	--attention-layers $(ATTENTION_LAYERS) \
	--attention-heads $(ATTENTION_HEADS) \
	--censoring-rate $(CENSORING_RATE) \
	--weibull-shape $(WEIBULL_SHAPE) \
	--x-low $(X_LOW) \
	--x-high $(X_HIGH) \
	--beta-low $(BETA_LOW) \
	--beta-high $(BETA_HIGH) \
	--seed $(SEED)

TRAIN_ARGS = \
	--data $(DATA_DIR)/train.pt \
	--val-data $(DATA_DIR)/val.pt \
	--test-data $(DATA_DIR)/test.pt \
	--clusters $(CLUSTERS) \
	--epochs $(EPOCHS) \
	--warmup-epochs $(WARMUP_EPOCHS) \
	--batch-size $(BATCH_SIZE) \
	--learning-rate $(LEARNING_RATE) \
	--encoder-hidden-dim $(ENCODER_HIDDEN_DIM) \
	--decoder-hidden-dim $(DECODER_HIDDEN_DIM) \
	--latent-dim $(MODEL_LATENT_DIM) \
	--n-layers $(N_LAYERS) \
	--dropout $(DROPOUT) \
	--seed $(SEED) \
	--save-dir $(RUN_DIR) \
	--save-artifacts $(SAVE_ARTIFACTS)

help:
	@printf "TRAILS experiment shortcuts\n\n"
	@printf "  make simulate-realistic    Generate train/val/test simulation data\n"
	@printf "  make train-realistic       Train without SwanLab\n"
	@printf "  make train-swanlab         Train with SwanLab live logging\n"
	@printf "  make experiment-realistic  Generate data, then train\n"
	@printf "  make quick-simulate        Generate a tiny smoke-test dataset\n"
	@printf "  make quick-train           Train quickly on the tiny dataset\n"
	@printf "  make check                 Run format, lint, pyright, pytest\n\n"
	@printf "Common overrides:\n"
	@printf "  make train-swanlab EPOCHS=5 WARMUP_EPOCHS=2 RUN_NAME=debug-1 SWANLAB_MODE=disabled\n"
	@printf "  make simulate-realistic DATA_DIR=data/simulated/test-1 SEED=20260517\n"

simulate-realistic:
	$(UV) run main.py simulate \
		--out $(DATA_DIR) \
		--split-patients $(TRAIN_PATIENTS) $(VAL_PATIENTS) $(TEST_PATIENTS) \
		$(SIMULATE_ARGS)

train-realistic:
	$(UV) run main.py train \
		$(TRAIN_ARGS) \
		$(SWANLAB_FLAGS)

train-swanlab: SWANLAB_FLAGS = --swanlab --swanlab-project $(SWANLAB_PROJECT) --swanlab-experiment $(SWANLAB_EXPERIMENT) $(if $(SWANLAB_MODE),--swanlab-mode $(SWANLAB_MODE),)
train-swanlab: train-realistic

experiment-realistic: simulate-realistic train-realistic

quick-simulate:
	$(MAKE) simulate-realistic \
		DATA_DIR=data/simulated/quick \
		TRAIN_PATIENTS=64 \
		VAL_PATIENTS=24 \
		TEST_PATIENTS=24 \
		SIM_HIDDEN_SIZE=32 \
		SIM_LATENT_DIM=4 \
		ATTENTION_LAYERS=2 \
		ATTENTION_HEADS=2

quick-train:
	$(MAKE) train-realistic \
		DATA_DIR=data/simulated/quick \
		RUN_NAME=quick-smoke \
		EPOCHS=2 \
		WARMUP_EPOCHS=1 \
		BATCH_SIZE=16 \
		ENCODER_HIDDEN_DIM=16 \
		DECODER_HIDDEN_DIM=16 \
		MODEL_LATENT_DIM=4 \
		SAVE_ARTIFACTS=config history test plot

check: format lint type test

format:
	$(UV) run ruff format

lint:
	$(UV) run ruff check --fix

type:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run pyright

test:
	UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run pytest
