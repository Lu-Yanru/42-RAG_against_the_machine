import json
from pathlib import Path
from pydantic import ValidationError
import sys

from src.config import (DATASET_PATH, SEARCH_RESULTS_PATH,
                        SEARCH_RESULTS_SAVE_DIR,
                        GROUND_TRUTH_PATH)
from src.models import (AnsweredQuestion, UnansweredQuestion,
                        StudentSearchResults, StudentSearchResultsAndAnswer)


def load_dataset_unanswered(dataset_path: str = DATASET_PATH) \
        -> list[UnansweredQuestion]:
    """Load unanswered questions."""
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        print(f"Error loading dataset '{dataset_path}'. "
              "Exiting...",
              file=sys.stderr)
        exit(1)

    try:
        queries = [UnansweredQuestion.model_validate(q)
                   for q in raw["rag_questions"]]
        return queries
    except (ValidationError, KeyError, TypeError):
        print(f"Error loading dataset '{dataset_path}'. "
              "Exiting...",
              file=sys.stderr)
        exit(1)


def load_dataset_answered(dataset_path: str = GROUND_TRUTH_PATH) \
        -> list[AnsweredQuestion]:
    """Load answered questions."""
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        print(f"Error loading dataset '{dataset_path}'. "
              "Exiting...",
              file=sys.stderr)
        exit(1)

    try:
        queries = [AnsweredQuestion.model_validate(q)
                   for q in raw["rag_questions"]]
        return queries
    except (ValidationError, KeyError, TypeError):
        print(f"Error loading dataset '{dataset_path}'. "
              "Exiting...",
              file=sys.stderr)
        exit(1)


def save_search_result(res: StudentSearchResults |
                       StudentSearchResultsAndAnswer,
                       dataset_path: str = DATASET_PATH,
                       save_dir: str = SEARCH_RESULTS_SAVE_DIR) -> None:
    """Save search results to a json file."""
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(dataset_path).name

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res.model_dump(), f, indent=4)
        print(f"Saved student_search_results to {out_path}")
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error: Failed to save search results: {e} "
              "Exiting...",
              file=sys.stderr)
        exit(1)


def load_search_results(dataset_path: str = SEARCH_RESULTS_PATH) \
        -> StudentSearchResults:
    """Load search results from a json file."""
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        print(f"Error loading dataset '{dataset_path}'. "
              "Exiting...",
              file=sys.stderr)
        exit(1)

    try:
        return StudentSearchResults(
            search_results=raw["search_results"],
            k=raw["k"]
        )
    except (ValidationError, KeyError, TypeError):
        print(f"Error loading dataset '{dataset_path}'. "
              "Exiting...",
              file=sys.stderr)
        exit(1)
