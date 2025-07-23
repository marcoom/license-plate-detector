# Project Makefile for License Plate Detector
# ==========================================
# Usage: make <target>

# ------------------------------------------------------------
# PHONY TARGETS
# ------------------------------------------------------------
.PHONY: help install install-dev clean run dist \
        docs docs-html docs-pdf docs-clean \
        test test-coverage \
        docker-build docker-run

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
SRC := src
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
DOCSSOURCEDIR = docs/source
DOCSBUILDDIR  = docs/build

# Detect operating system specific commands
ifeq (Darwin,$(shell uname -s))	
	FIND := gfind  # GNU find on macOS via brew coreutils
else
	FIND := find
endif

# ------------------------------------------------------------
# ENVIRONMENT SETUP
# ------------------------------------------------------------
$(VENV):
	python -m venv $(VENV)
	$(PIP) install --upgrade pip

install: $(VENV)
	$(PIP) install -r requirements.txt

install-dev: $(VENV)
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------
run:
	@echo "Starting application..."
	$(PYTHON) src/app.py

# ------------------------------------------------------------
# CODE QUALITY
# ------------------------------------------------------------
lint:
	$(PYTHON) -m ruff check $(SRC)

format:
	$(PYTHON) -m ruff format $(SRC)
	$(PYTHON) -m black $(SRC)

type-check:
	$(PYTHON) -m mypy $(SRC)

# ------------------------------------------------------------
# TESTING
# ------------------------------------------------------------
test:
	@echo "Running tests with pytest..."
	@PYTHONPATH=src $(VENV)/bin/pytest -q

test-coverage:
	@echo "Running tests with coverage..."
	@PYTHONPATH=src $(VENV)/bin/pytest --cov=src --cov-report=term-missing

# ------------------------------------------------------------
# DOCUMENTATION
# ------------------------------------------------------------
# Build both HTML and PDF documentation
docs: docs-html docs-pdf

# Build HTML documentation
docs-html:
	@echo "Building HTML documentation..."
	$(SPHINXBUILD) -b html "$(DOCSSOURCEDIR)" "$(DOCSBUILDDIR)/html" $(SPHINXOPTS)
	@echo "HTML documentation built in $(DOCSBUILDDIR)/html/"

# Build PDF documentation
docs-pdf:
	@echo "Building PDF documentation..."
	$(SPHINXBUILD) -b pdf "$(DOCSSOURCEDIR)" "$(DOCSBUILDDIR)/pdf" $(SPHINXOPTS)
	@echo "PDF documentation built in $(DOCSBUILDDIR)/pdf/"

# Clean documentation build files
docs-clean:
	rm -rf "$(DOCSBUILDDIR)"

# ------------------------------------------------------------
# BUILD AND DISTRIBUTION
# ------------------------------------------------------------
dist: clean install-dev
	@echo "Building source and wheel distribution..."
	$(PYTHON) -m build --wheel --sdist

clean:
	@echo "Removing Python caches & build artifacts..."
	@$(FIND) . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf build dist *.egg-info .pytest_cache || true

# ------------------------------------------------------------
# DEPLOYMENT
# ------------------------------------------------------------
docker-build: dist
	@echo "Building Docker image..."
	docker build -t license-plate-detector .

docker-run:
	@echo "Running Docker container..."
	@if [ -e /dev/video0 ]; then \
		echo "Webcam detected, adding device..."; \
		docker run -it --rm --device /dev/video0:/dev/video0 -p 7860:7860 license-plate-detector; \
	else \
		echo "No webcam detected, running without webcam device..."; \
		docker run -it --rm -p 7860:7860 license-plate-detector; \
	fi

docker-remove:
	@echo "Removing Docker image..."
	docker image rm -f license-plate-detector

# ------------------------------------------------------------
# HELP
# ------------------------------------------------------------
help:  ## Show this help message
	@echo "Available commands:"
	@echo
	@echo "Environment Setup:"
	@echo "  install        Install production dependencies"
	@echo "  install-dev    Install development dependencies"
	@echo
	@echo "Run:"
	@echo "  run            Run the application"
	@echo
	@echo "Code Quality:"
	@echo "  lint           Run code quality checks"
	@echo "  format         Format code"
	@echo "  type-check     Check type annotations"
	@echo
	@echo "Testing:"
	@echo "  test           Run tests"
	@echo "  test-coverage  Run tests with coverage"
	@echo
	@echo "Documentation:"
	@echo "  docs           Build HTML and PDF documentation"
	@echo "  docs-html      Build HTML documentation only"
	@echo "  docs-pdf       Build PDF documentation only"
	@echo "  docs-clean     Remove documentation build files"
	@echo
	@echo "Build & Distribution:"
	@echo "  dist           Build source and wheel distributions"
	@echo "  clean          Remove build artifacts and cache files"
	@echo
	@echo "Deployment:"
	@echo "  docker-build   Build Docker image"
	@echo "  docker-run     Run Docker container"
	@echo "  docker-remove  Remove Docker image"
	@echo
	@echo "To display this help message, run 'make help'"

# End of Makefile
