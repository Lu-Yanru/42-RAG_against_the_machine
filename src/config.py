"""
Shared configuration constants.
"""


INDEX_DIR = "data/processed"
DATASET_PATH = "data/datasets/UnansweredQuestions/dataset_docs_public.json"
SEARCH_RESULTS_SAVE_DIR = "data/output/search_results"
SEARCH_RESULTS_PATH = ("data/output/search_results/dataset_docs_public.json")
ANSWER_SAVE_DIR = "data/output/search_results_and_answer"
GROUND_TRUTH_PATH = "data/datasets/AnsweredQuestions/dataset_docs_public.json"

IOU_THRESHOLD = 0.05

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 256
MAX_SOURCE_CHARS = 8000  # budget for concatenated source text in the prompt
NO_CONTEXT_ANSWER = "I don't have enough context to answer this question."
GENERATION_FAILED_ANSWER = "Answer generation failed for this question."
