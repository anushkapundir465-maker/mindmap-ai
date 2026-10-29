from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.analysis import analyze_answer, keyword_coverage, tfidf_similarity
from src.preprocessing import preprocess_text
from src.quiz import generate_quiz, grade_quiz
from src.storage import load_attempts, save_attempt


class PreprocessingTests(unittest.TestCase):
    def test_normalizes_case_and_removes_punctuation_and_stop_words(self) -> None:
        self.assertEqual(preprocess_text("THE Tokenizer, SPLITS words!"), ["tokenizer", "splits", "words"])

    def test_non_string_input_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            preprocess_text(None)  # type: ignore[arg-type]


class AnalysisTests(unittest.TestCase):
    def test_keyword_coverage_and_tfidf_similarity_are_bounded(self) -> None:
        reference = "TF-IDF gives rare terms more weight across documents."
        answer = "Rare terms get more weight."
        self.assertGreater(keyword_coverage(reference, answer), 0)
        self.assertGreater(tfidf_similarity(reference, answer), 0)
        self.assertLessEqual(keyword_coverage(reference, answer), 1)
        self.assertLessEqual(tfidf_similarity(reference, answer), 1)

    def test_empty_answer_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analyze_answer("Reference text", "   ", "TF-IDF")

    def test_tfidf_similarity_handles_text_without_usable_terms(self) -> None:
        self.assertEqual(tfidf_similarity("the and or", "a an the"), 0.0)


class QuizTests(unittest.TestCase):
    def test_generated_quiz_can_be_graded(self) -> None:
        text = (
            "Tokenization splits sentences into smaller tokens for language models. "
            "TF-IDF weights informative terms across documents in a corpus. "
            "Neural networks use hidden layers to transform input features."
        )
        questions = generate_quiz(text, 3)
        self.assertEqual(len(questions), 3)
        self.assertTrue(
            all(question["options"][0] != question["answer"] for question in questions)
        )
        score, feedback = grade_quiz(
            questions,
            [str(question["answer"]) for question in questions],
        )
        self.assertEqual(score, 100)
        self.assertTrue(all(correct for correct, _ in feedback))

    def test_invalid_answer_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            grade_quiz([{"answer": "token"}], [])

    def test_unanswered_question_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Select an answer"):
            grade_quiz([{"answer": "token", "explanation": "A token."}], [None])


class StorageTests(unittest.TestCase):
    def test_attempt_round_trips_to_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "progress.json"
            save_attempt({"kind": "quiz", "topic": "Tokenization", "score": 80}, path)
            attempts = load_attempts(path)
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0]["score"], 80)
            self.assertIn("timestamp", attempts[0])

    def test_corrupt_progress_file_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            path.write_text("{", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                load_attempts(path)


if __name__ == "__main__":
    unittest.main()
