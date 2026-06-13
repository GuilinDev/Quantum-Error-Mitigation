# Makefile for SQAI 2026 Neural Error Mitigation Project

.PHONY: help install test lint format demo train evaluate plot clean paper

# Default target
help:
	@echo "SQAI 2026 Neural Error Mitigation"
	@echo ""
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make test        - Run unit tests"
	@echo "  make test-cov    - Run tests with coverage"
	@echo "  make lint        - Run linting"
	@echo "  make format      - Format code"
	@echo "  make demo        - Run quick demonstration"
	@echo "  make train       - Train model (VQE config)"
	@echo "  make train-qaoa  - Train model (QAOA config)"
	@echo "  make evaluate    - Evaluate trained model"
	@echo "  make plot        - Generate figures"
	@echo "  make paper       - Compile LaTeX paper"
	@echo "  make clean       - Clean generated files"
	@echo "  make all         - Run full pipeline"

# Install dependencies
install:
	pip install -r requirements.txt

# Run tests
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# Lint code
lint:
	python -m mypy src/ --ignore-missing-imports
	python -m black src/ tests/ --check
	python -m isort src/ tests/ --check-only

# Format code
format:
	python -m black src/ tests/ experiments/
	python -m isort src/ tests/ experiments/

# Run demo
demo:
	python run.py demo

# Training
train:
	python run.py train --config experiments/configs/vqe.yaml

train-qaoa:
	python run.py train --config experiments/configs/qaoa.yaml

train-noise:
	python run.py train --config experiments/configs/noise_learning.yaml

# Generate data only
data:
	python run.py train --config experiments/configs/vqe.yaml --data-only

# Evaluate
evaluate:
	python run.py evaluate --model experiments/results/checkpoints/best_model.pt --benchmark all

# Generate plots
plot:
	python run.py plot --results experiments/results --output paper/figures

# Compile paper
paper:
	cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

# Clean generated files
clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .mypy_cache
	rm -rf *.egg-info
	rm -rf dist build
	rm -f paper/*.aux paper/*.bbl paper/*.blg paper/*.log paper/*.out paper/*.toc
	find . -name "*.pyc" -delete
	find . -name ".DS_Store" -delete

# Full pipeline
all: install test demo

# Development setup
dev-setup: install
	pip install -e ".[dev]"
	pre-commit install
