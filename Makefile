.PHONY: help clean setup test test_report

include .env
export $(shell sed 's/=.*//' .env)

help:
	@echo "make clean"
	@echo "       clean project removing unnecessary files"
	@echo "make setup"
	@echo "       prepare environment"
	@echo "make test"
	@echo "       run tests"
	@echo "make test_report"
	@echo "       run tests and save tests and coverag reports"

setup: poetry.lock
poetry.lock: pyproject.toml
	@poetry install -vvv
	@touch poetry.lock

clean:
	@rm -rf build dist .eggs *.egg-info
	@rm -rf .benchmarks .coverage reports htmlcov .tox
	@find . -type d -name '.mypy_cache' -exec rm -rf {} +
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type d -name '*pytest_cache*' -exec rm -rf {} +
	@find . -type f -name "*.py[co]" -exec rm -rf {} +

test: setup
	@poetry run pytest --durations=3 -v --cov=${PWD}/huggingsound 

test_report: setup
	@poetry run pytest --durations=3 -v --cov=${PWD}/huggingsound --cov-report xml:reports/coverage.xml --junitxml=reports/tests.xml

publish: setup
	@poetry config pypi-token.pypi ${PYPI_TOKEN}
	@poetry publish --build
