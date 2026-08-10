import pytest

from src.evaluation.evaluator import (IOU_THRESHOLD, Evaluator)
from src.models import MinimalSource


def source(file_path: str, start: int, end: int) -> MinimalSource:
    return MinimalSource(file_path=file_path,
                         first_character_index=start,
                         last_character_index=end)


class TestIoU:
    def test_identical_ranges_same_file_is_one(self):
        a = source("docs/lora.md", 100, 200)
        b = source("docs/lora.md", 100, 200)
        assert Evaluator.iou(a, b) == 1.0

    def test_no_overlap_same_file_is_zero(self):
        a = source("docs/lora.md", 0, 10)
        b = source("docs/lora.md", 50, 60)
        assert Evaluator.iou(a, b) == 0.0

    def test_different_file_is_zero_even_with_identical_ranges(self):
        # Same offsets don't matter at all if the file doesn't match --
        # this is the check that must run before any range arithmetic.
        a = source("docs/lora.md", 100, 200)
        b = source("docs/quantization.md", 100, 200)
        assert Evaluator.iou(a, b) == 0.0

    def test_known_high_overlap_value(self):
        # GT1/R1 from the worked example: near-perfect overlap.
        gt = source("docs/lora.md", 4695, 6098)
        pred = source("docs/lora.md", 4700, 6000)
        assert Evaluator.iou(pred, gt) == pytest.approx(1300 / 1403)
        assert Evaluator.iou(pred, gt) > 0.9

    def test_known_low_overlap_value_below_threshold(self):
        # GT2/R3 from the worked example: clips the edge, stays under 0.05.
        gt = source("docs/lora.md", 13150, 14470)
        pred = source("docs/lora.md", 13000, 13200)
        assert Evaluator.iou(pred, gt) == pytest.approx(50 / 1470)
        assert Evaluator.iou(pred, gt) < IOU_THRESHOLD

    def test_boundary_exactly_at_threshold_is_not_zero(self):
        gt = source("docs/lora.md", 0, 10)
        pred = source("docs/lora.md", 9, 20)  # overlap=1, union=20 -> 0.05
        assert Evaluator.iou(pred, gt) == pytest.approx(0.05)

    def test_just_below_threshold(self):
        gt = source("docs/lora.md", 0, 10)
        pred = source("docs/lora.md", 9, 21)  # overlap=1, union=21 -> <0.05
        assert Evaluator.iou(pred, gt) < IOU_THRESHOLD

    def test_zero_length_ranges_do_not_divide_by_zero(self):
        a = source("docs/lora.md", 5, 5)
        b = source("docs/lora.md", 5, 5)
        assert Evaluator.iou(a, b) == 0.0

    def test_iou_is_symmetric(self):
        a = source("docs/lora.md", 100, 300)
        b = source("docs/lora.md", 200, 400)
        assert Evaluator.iou(a, b) == Evaluator.iou(b, a)


class TestRecallAtK:
    def test_no_ground_truth_returns_none(self):
        predicted = [source("docs/lora.md", 0, 100)]
        assert Evaluator.recall_at_k(predicted, []) is None

    def test_single_ground_truth_found_is_one(self):
        gt = [source("docs/lora.md", 100, 200)]
        predicted = [source("docs/lora.md", 100, 200)]
        assert Evaluator.recall_at_k(predicted, gt) == 1.0

    def test_single_ground_truth_not_found_is_zero(self):
        gt = [source("docs/lora.md", 100, 200)]
        predicted = [source("docs/lora.md", 900, 1000)]
        assert Evaluator.recall_at_k(predicted, gt) == 0.0

    def test_empty_predicted_list_is_zero(self):
        gt = [source("docs/lora.md", 100, 200)]
        assert Evaluator.recall_at_k([], gt) == 0.0

    def test_worked_example_partial_match_is_one_half(self):
        # The two-ground-truth-source example from the design discussion:
        # GT1 gets a near-perfect hit, GT2's only candidate falls just
        # short of the IoU threshold.
        gt = [
            source("docs/lora.md", 4695, 6098),
            source("docs/lora.md", 13150, 14470),
        ]
        predicted = [
            source("docs/lora.md", 4700, 6000),
            source("docs/quantization.md", 100, 300),
            source("docs/lora.md", 13000, 13200),
        ]
        assert Evaluator.recall_at_k(predicted, gt) == pytest.approx(0.5)

    def test_all_ground_truth_found_is_one(self):
        gt = [
            source("docs/a.md", 0, 100),
            source("docs/b.md", 0, 100),
        ]
        predicted = [
            source("docs/b.md", 0, 100),
            source("docs/a.md", 0, 100),
        ]
        assert Evaluator.recall_at_k(predicted, gt) == 1.0

    def test_match_can_come_from_any_predicted_source_not_just_first(self):
        gt = [source("docs/lora.md", 100, 200)]
        predicted = [
            source("docs/unrelated.md", 0, 50),
            source("docs/other.md", 500, 600),
            source("docs/lora.md", 100, 200),  # the actual match, last
        ]
        assert Evaluator.recall_at_k(predicted, gt) == 1.0

    def test_boundary_iou_counts_as_found(self):
        gt = [source("docs/lora.md", 0, 10)]
        predicted = [source("docs/lora.md", 9, 20)]  # IoU == 0.05 exactly
        assert Evaluator.recall_at_k(predicted, gt) == 1.0

    def test_just_below_boundary_iou_does_not_count(self):
        gt = [source("docs/lora.md", 0, 10)]
        predicted = [source("docs/lora.md", 9, 21)]  # IoU < 0.05
        assert Evaluator.recall_at_k(predicted, gt) == 0.0


class TestMeanRecallAtK:
    def test_macro_average_of_mixed_scores(self):
        assert Evaluator.mean_recall_at_k([1.0, 0.5, 0.0]) == pytest.approx(0.5)

    def test_none_values_are_excluded_not_treated_as_zero(self):
        # A question with no ground truth shouldn't drag the average down
        # as if the retriever had failed it.
        with_none = Evaluator.mean_recall_at_k([1.0, None, 1.0])
        without_none = Evaluator.mean_recall_at_k([1.0, 1.0])
        assert with_none == without_none == 1.0

    def test_all_none_returns_zero_not_a_crash(self):
        assert Evaluator.mean_recall_at_k([None, None]) == 0.0

    def test_empty_list_returns_zero(self):
        assert Evaluator.mean_recall_at_k([]) == 0.0
