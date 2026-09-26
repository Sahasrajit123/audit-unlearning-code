PYTHON ?= python
NEWTON_CONFIGS := $(wildcard config/convex_unlearning/*.yaml)
OP_CONFIGS     := $(wildcard config/output_perturbation/*.yaml)

.PHONY: help install env test test-fast run-newton run-op run-all clean

help:
	@echo "make install     pip install runtime + dev dependencies"
	@echo "make env         create the conda env from environment.yml"
	@echo "make test        run all tests (includes ~1 min end-to-end runs)"
	@echo "make test-fast   run unit tests only"
	@echo "make run-newton  run every config in config/convex_unlearning/"
	@echo "make run-op      run every config in config/output_perturbation/"
	@echo "make run-all     run-newton + run-op"
	@echo "make clean       remove caches (keeps experiment outputs)"

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

env:
	conda env create -f environment.yml

test:
	$(PYTHON) -m pytest

test-fast:
	$(PYTHON) -m pytest -m "not slow"

run-newton:
	@for c in $(NEWTON_CONFIGS); do \
		echo "=== $$c"; \
		$(PYTHON) run_privacy_attack_final.py --config $$c 2>&1 | tee log_$$(basename $$c .yaml).txt; \
	done

run-op:
	@for c in $(OP_CONFIGS); do \
		echo "=== $$c"; \
		$(PYTHON) run_output_perturbation_attack.py --config $$c 2>&1 | tee log_op_$$(basename $$c .yaml).txt; \
	done

run-all: run-newton run-op

clean:
	rm -rf __pycache__ tests/__pycache__ .pytest_cache *.egg-info build
