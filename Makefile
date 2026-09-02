.PHONY: setup lint typecheck test run clean

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -e ".[dev]"

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts

typecheck:
	mypy src scripts

test:
	pytest --cov=psg_tracker --cov-report=term-missing

run:
	streamlit run src/psg_tracker/app/main.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
