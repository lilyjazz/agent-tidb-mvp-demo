.PHONY: install format lint type-check clean help

help:
	@echo "Available commands:"
	@echo "  make install      Install dependencies including dev tools"
	@echo "  make format       Format code with ruff"
	@echo "  make lint         Check code with ruff"
	@echo "  make type-check   Check types with mypy"
	@echo "  make clean        Remove build artifacts and cache"

install:
	pip install -e ".[dev]"

format:
	ruff format src/ examples/

lint:
	ruff check src/ examples/

type-check:
	mypy src/

clean:
	rm -rf .venv
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
