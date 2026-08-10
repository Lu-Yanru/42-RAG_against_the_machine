QUERY = "How to configure the OpenAI server?"

all: install run

install:
	uv sync

run: index search_dataset answer_dataset evaluate

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

# RAG commands
index:
	uv run python -m src index

search:
	uv run python -m src search $(QUERY)

search_dataset_doc:
	uv run python -m src search_dataset

search_dataset_code:
	uv run python -m src search_dataset --k 5 \
		--dataset_path "data/datasets/UnansweredQuestions/dataset_code_public.json" \
		--save_directory "data/output/search_results"

answer:
	uv run python -m src answer $(QUERY)

answer_dataset_docs:
	uv run python -m src answer_dataset

answer_dataset_code:
	uv run python -m src answer_dataset

evaluate_docs:
	uv run python -m src evaluate

evaluate_code:
	uv run python -m src evaluate \
		--student_search_results_path "data/output/search_results/dataset_code_public.json" \
		--dataset_path "data/datasets/AnsweredQuestions/dataset_code_public.json"

# moulinette
moulinette-docs:
	uv run python -m src search_dataset --k 10
	./moulinette/moulinette-ubuntu evaluate_student_search_results \
		"data/output/search_results/dataset_docs_public.json" \
		"data/datasets/AnsweredQuestions/dataset_docs_public.json" \
		--k 10 --max_context_length 2000

moulinette-code:
	uv run python -m src search_dataset --dataset_path \
		"data/datasets/UnansweredQuestions/dataset_code_public.json" \
		--k 10 \
		--save_directory "data/output/search_results"
	./moulinette/moulinette-ubuntu evaluate_student_search_results \
		"data/output/search_results/dataset_code_public.json" \
		"data/datasets/AnsweredQuestions/dataset_code_public.json" \
		--k 10 --max_context_length 2000

.PHONY: all install run help debug clean fclean lint lint-strict re \
		index search search_dataset answer answer_dataset evaluate \
		moulinette-docs moulinette-code
