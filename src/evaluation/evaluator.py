import json
from pydantic import ValidationError
import sys
from tqdm import tqdm

from src.models import AnsweredQuestion, MinimalSource, StudentSearchResults


IOU_THRESHOLD = 0.05


class Evaluator:
    def __init__(self,
                 search_results_path: str,
                 dataset_path: str) -> None:
        self.ground_truth = Evaluator.load_dataset_answered(
            dataset_path)  # list[AnsweredQuestions]
        self.search_res = Evaluator.load_search_results(
            search_results_path)  # StudentSearchResults
        self.k = self.search_res.k
        # List[MinimalSearchResults]
        self.min_search_res = self.search_res.search_results

        self.matched_res = self.match_questions()
        self.recall_per_question = []
        for predicted, gt in tqdm(self.matched_res.values()):
            self.recall_per_question.append(
                Evaluator.recall_at_k(predicted, gt))
        self.mean_recall = Evaluator.mean_recall_at_k(self.recall_per_question)

    def match_questions(self) -> dict[str, tuple[list[MinimalSource],
                                                 list[MinimalSource]]]:
        """
        Match predicted sources to ground truth sources.
        Return a dict of question_id: tuple(predicted_sources,
        ground_truth_sources).
        """
        matched_res: dict[str, tuple[list[MinimalSource],
                                     list[MinimalSource]]] = {}
        for gt in self.ground_truth:
            for res in self.min_search_res:
                if gt.question_id == res.question_id:
                    matched_res[gt.question_id] = (res.retrieved_sources,
                                                   gt.sources)

        return matched_res

    @staticmethod
    def iou(a: MinimalSource, b: MinimalSource) -> float:
        """
        Intersection-over-union of two sources' character ranges.
        Returns 0.0 whenever the two sources are in different files.
        """
        if a.file_path != b.file_path:
            return 0.0

        overlap = max(0, (min(a.last_character_index, b.last_character_index)
                      - max(a.first_character_index, b.first_character_index)))
        union = ((a.last_character_index - a.first_character_index)
                 + (b.last_character_index - b.first_character_index)
                 - overlap)
        if union <= 0:
            return 0.0
        return overlap / union

    @staticmethod
    def load_dataset_answered(dataset_path: str) \
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

    @staticmethod
    def load_search_results(dataset_path: str) \
            -> StudentSearchResults:
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

    @staticmethod
    def recall_at_k(predicted: list[MinimalSource],
                    ground_truth: list[MinimalSource]) -> float | None:
        """
        Recall@k for a single question: the fraction of `ground_truth`
        sources that have a match (same file, IoU >= 0.05) somewhere in
        `predicted`.
        Returns None when `ground_truth` is empty.
        """
        if not ground_truth:
            return None

        found = 0
        for gt in ground_truth:
            if any(Evaluator.iou(pred, gt) >= IOU_THRESHOLD
                   for pred in predicted):
                found += 1
        return found / len(ground_truth)

    @staticmethod
    def mean_recall_at_k(per_question_recalls: list[float | None]) -> float:
        """
        Macro-average recall@k across questions: the mean of each question's
        own recall@k value. Questions with no ground truth (None) are excluded
        from the average rather than counted as 0.
        Returns 0.0 if every question was excluded.
        """
        scored = [r for r in per_question_recalls if r is not None]
        if not scored:
            return 0.0
        return sum(scored) / len(scored)
