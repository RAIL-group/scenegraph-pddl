MAKEFLAGS += --no-print-directory

# Paths to data/ and third_party/ are relative to the working directory, so run
# make from the repository root.
PYTHON = uv run python
SEED ?= 0
GOALS ?=
PLANNER ?= ff-astar

.PHONY: help
help:
	@echo ''
	@echo 'Usage: make [TARGET] [EXTRA_ARGUMENTS]'
	@echo 'Targets:'
	@echo '  install	create the virtual environment (uv sync)'
	@echo '  download	download ProcTHOR-10k and the sentence-BERT model into ./data'
	@echo '  fast-downward	clone and build Fast Downward into ./third_party'
	@echo '  pipeline	house -> scene graph -> problem.pddl -> plan for SEED'
	@echo '  test		run pytest'
	@echo '  flake8		run the flake8 format checker'
	@echo 'Extra Arguments:'
	@echo '  SEED		[0] ProcTHOR-10k house index'
	@echo '  GOALS		[random] e.g. GOALS="--goal apple fridge --goal mug sink"'
	@echo '  PLANNER	[ff-astar] ff-astar, ff-wastar2 (faster on larger houses), or a FD search string'
	@echo ''

.PHONY: install
install:
	uv sync

.PHONY: download
download:
	@command -v git-lfs >/dev/null || { echo "git-lfs is required to download ProcTHOR-10k: brew install git-lfs (macOS) or sudo apt-get install git-lfs (Linux)"; exit 1; }
	@$(PYTHON) -m procgraph.scripts.download_data

.PHONY: fast-downward
fast-downward:
	@$(PYTHON) -c "from procgraph.pddl import fast_downward; fast_downward.ensure_installed()"

.PHONY: pipeline
pipeline:
	@$(PYTHON) -m procgraph.scripts.run_pipeline --seed $(SEED) --planner $(PLANNER) $(GOALS)

.PHONY: test
test:
	@uv run pytest -v

.PHONY: flake8
flake8:
	@uv run flake8 src tests
	@echo "... no formatting issues discovered."
