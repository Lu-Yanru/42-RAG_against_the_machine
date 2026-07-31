import pytest
from pydantic import ValidationError
import uuid


from src.models import (MinimalSource,
                        UnansweredQuestion,
                        AnsweredQuestion,
                        RagDataset,
                        MinimalSearchResults,
                        MinimalAnswer,
                        StudentSearchResults,
                        StudentSearchResultsAndAnswer)


def test_MinimalSource():
    with pytest.raises(ValidationError):
        MinimalSource(file_path="data/raw",
                      first_character_index="abc",
                      last_character_index=23)


def test_UnansweredQuestion():
    with pytest.raises(ValidationError):
        UnansweredQuestion(question_id=str(uuid.uuid4()),
                           question=42)


def test_AnsweredQuestion():
    with pytest.raises(ValidationError):
        AnsweredQuestion(question_id=str(uuid.uuid4()),
                         question="question?",
                         sources=[],
                         answer=42)


def test_RagDataset():
    with pytest.raises(ValidationError):
        RagDataset(rag_questions="question?")


def test_MinimalSearchResults():
    with pytest.raises(ValidationError):
        MinimalSearchResults(
            question_id="abc",
            question=42,
            retrieved_sources=[])


def test_MinimalAnswer():
    with pytest.raises(ValidationError):
            MinimalAnswer(
                question_id="abc",
                question="abc",
                retrieved_sources=[],
                answer=42)


def test_StudentSearchResults():
    with pytest.raises(ValidationError):
        StudentSearchResults(
            search_results=[],
            k="abc")


def test_StudentSearchResultsAndAnswer():
    with pytest.raises(ValidationError):
            StudentSearchResults(
                search_results=[],
                k="abc")
