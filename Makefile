QUERY = "How to configure the OpenAI server?"

all: install run

install:
	uv sync

run: index search_dataset_docs evaluate_docs answer_dataset_docs

help:
	uv run python -m src -h

debug:
	uv run python -m pdb -m src

test:
	uv run pytest tests/ -m "not slow"

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

search_dataset_docs:
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
	uv run python -m src answer_dataset \
		--student_search_results_path "data/output/search_results/dataset_code_public.json" \
		--save_directory "data/output/search_results_and_answer"

evaluate_docs:
	uv run python -m src evaluate

evaluate_code:
	uv run python -m src evaluate \
		--student_search_results_path "data/output/search_results/dataset_code_public.json" \
		--dataset_path "data/datasets/AnsweredQuestions/dataset_code_public.json"

# moulinette
moulinette_docs:
	uv run python -m src search_dataset --k 10
	./moulinette/moulinette-ubuntu evaluate_student_search_results \
		"data/output/search_results/dataset_docs_public.json" \
		"data/datasets/AnsweredQuestions/dataset_docs_public.json" \
		--k 10 --max_context_length 2000

moulinette_code:
	uv run python -m src search_dataset --dataset_path \
		"data/datasets/UnansweredQuestions/dataset_code_public.json" \
		--k 10 \
		--save_directory "data/output/search_results"
	./moulinette/moulinette-ubuntu evaluate_student_search_results \
		"data/output/search_results/dataset_code_public.json" \
		"data/datasets/AnsweredQuestions/dataset_code_public.json" \
		--k 10 --max_context_length 2000


# bonus
index_semantic:
	uv run python -m src index --method "semantic"

index_hybrid:
	uv run python -m src index --method "hybrid"

search_semantic:
	uv run python -m src search $(QUERY) --method semantic

search_hybrid:
	uv run python -m src search $(QUERY) --method hybrid

search_dataset_docs_semantic:
	uv run python -m src search_dataset --method semantic

search_dataset_docs_hybrid:
	uv run python -m src search_dataset --method hybrid

search_dataset_code_semantic:
	uv run python -m src search_dataset --k 5 --method semantic \
		--dataset_path "data/datasets/UnansweredQuestions/dataset_code_public.json" \
		--save_directory "data/output/search_results"

search_dataset_code_hybrid:
	uv run python -m src search_dataset --k 5 --method hybrid \
		--dataset_path "data/datasets/UnansweredQuestions/dataset_code_public.json" \
		--save_directory "data/output/search_results"

answer_semantic:
	uv run python -m src answer $(QUERY) --k 5 --method semantic

answer_hybrid:
	uv run python -m src answer $(QUERY) --k 5 --method hybrid


.PHONY: all install run help debug clean fclean lint lint-strict re \
		index search search_dataset answer answer_dataset evaluate \
		moulinette-docs moulinette-code index_semantic index_hybrid \
		search_semantic search_hybrid
