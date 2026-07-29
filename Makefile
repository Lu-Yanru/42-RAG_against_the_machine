all: install run

install:
	uv sync

run:
	uv run python -m src

visualize:
	uv run python -m src -v

help:
	uv run python -m src -h

debug:
	uv run python -m pdb -m src

test:
	uv run pytest tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .mypy_cache
	rm -rf .pytest_cache

fclean: clean
	uv cache clean
	rm -rf .venv

lint:
	uv run python -m flake8 src
	uv run python -m mypy src --warn-return-any --warn-unused-ignores \
							  --ignore-missing-imports --disallow-untyped-defs \
							  --check-untyped-defs

lint-strict:
	uv run python -m flake8 src
	uv run python -m mypy src --strict

re: fclean all

.PHONY: all install run help debug clean fclean lint lint-strict re
