.PHONY: setup run test clean help

PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin

-include .env
export

help:
	@echo "AI Harness - available targets:"
	@echo "  setup   - create virtualenv and install dependencies"
	@echo "  run     - launch the AI Harness TUI"
	@echo "  test    - run the test suite"
	@echo "  clean   - remove generated artefacts (venv, caches)"

setup:
	@echo "==> Creating virtualenv at $(VENV)"
	@$(PYTHON) -m venv $(VENV)
	@echo "==> Upgrading pip"
	@$(BIN)/pip install --upgrade pip
	@echo "==> Installing dependencies"
	@$(BIN)/pip install -r requirements.txt
	@echo "==> Setup complete."
	@echo "==> Next:  export AI_API_KEY=<your-key>  &&  make run"

run:
	@test -n "$(AI_API_KEY)" || { echo "ERROR: AI_API_KEY is not set. Run: export AI_API_KEY=<your-key>"; exit 1; }
	@echo "==> Launching AI Harness"
	@$(BIN)/python -m harness

test:
	@cd $(CURDIR) && $(BIN)/python -m pytest tests/ -v

clean:
	@rm -rf $(VENV)
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete
	@echo "==> Cleaned"
