from tqdm import tqdm

from src.config import (IOU_THRESHOLD, SEARCH_RESULTS_PATH,
                        GROUND_TRUTH_PATH)
from src.models import MinimalSource
from src.utils.json_io import (load_dataset_answered,
                               load_search_results)


class Evaluator:
    def __init__(self,
                 search_results_path: str = SEARCH_RESULTS_PATH,
                 dataset_path: str = GROUND_TRUTH_PATH) -> None:
        self.ground_truth = load_dataset_answered(
            dataset_path)  # list[AnsweredQuestions]
        self.search_res = load_search_results(
            search_results_path)  # StudentSearchResults
        self.k = self.search_res.k
        # List[MinimalSearchResults]
        self.min_search_res = self.search_res.search_results

        self.matched_res = self.match_questions()
        self.recall_per_question = []
        for predicted, gt in tqdm(self.matched_res.values(),
                                  desc="Calculating recall"):
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
