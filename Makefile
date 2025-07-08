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
# DEVELOPMENT ENVIRONMENT
# ------------------------------------------------------------
$(VENV):
	python -m venv $(VENV)
	$(PIP) install --upgrade pip

install: $(VENV)
	$(PIP) install -r requirements.txt

install-dev: $(VENV)
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

clean:
	@echo "Removing Python caches & build artifacts..."
	@$(FIND) . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf build dist *.egg-info .pytest_cache || true

# ------------------------------------------------------------
# TESTING (commented until implemented)
# ------------------------------------------------------------
# test:
# 	@echo "Running ALL tests with pytest..."
# 	$(VENV)/bin/pytest -q
#
# test-coverage:
# 	@echo "Running tests with coverage..."
# 	$(VENV)/bin/pytest --cov=src --cov-report=term-missing

# ------------------------------------------------------------
# BUILD & RUN
# ------------------------------------------------------------
run: install
	@echo "Starting application..."
	$(PYTHON) src/app.py

dist: clean install-dev
	@echo "Building source and wheel distribution..."
	$(PYTHON) -m build --wheel --sdist

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
# DEPLOYMENT (commented until implemented)
# ------------------------------------------------------------
# docker-build:
# 	@echo "Building Docker image..."
# 	docker build -t license-plate-detector .
#
# docker-run:
# 	@echo "Running Docker container..."
# 	docker run --rm -it -p 7860:7860 license-plate-detector

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
	@echo "Development:"
	@echo "  run            Run the application"
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
	@echo "For more information about a command, run 'make help'"

# End of Makefile
