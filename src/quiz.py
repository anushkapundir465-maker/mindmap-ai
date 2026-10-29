from __future__ import annotations

import re

from src.preprocessing import STOP_WORDS


SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
WORD_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z-]{3,}\b")


def generate_quiz(text: str, question_count: int = 3) -> list[dict[str, object]]:
    """Generate experimental fill-in-the-blank MCQs from explanatory sentences."""
    if not text.strip() or question_count < 1:
        return []
    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT.split(text) if len(sentence.split()) >= 6]
    questions: list[dict[str, object]] = []
    used_answers: set[str] = set()
    for sentence in sentences:
        candidates = [
            word for word in WORD_PATTERN.findall(sentence)
            if word.lower() not in STOP_WORDS and word.lower() not in used_answers
        ]
        if not candidates:
            continue
        answer = max(candidates, key=lambda word: (len(word), word.lower()))
        used_answers.add(answer.lower())
        distractors = []
        for candidate in re.findall(WORD_PATTERN, text):
            if (
                candidate.lower() != answer.lower()
                and candidate.lower() not in STOP_WORDS
                and candidate.lower() not in {word.lower() for word in distractors}
            ):
                distractors.append(candidate)
            if len(distractors) == 3:
                break
        if len(distractors) < 3:
            continue
        options = distractors[:3]
        correct_position = sum(ord(character) for character in sentence) % len(options) + 1
        options.insert(correct_position, answer)
        question = re.sub(re.escape(answer), "________", sentence, count=1, flags=re.IGNORECASE)
        questions.append(
            {
                "question": question,
                "answer": answer,
                "options": options,
                "explanation": sentence,
            }
        )
        if len(questions) >= question_count:
            break
    return questions


def grade_quiz(
    questions: list[dict[str, object]], selected_answers: list[str | None]
) -> tuple[float, list[tuple[bool, str]]]:
    if not questions or len(questions) != len(selected_answers):
        raise ValueError("Provide one selected answer for every quiz question.")
    if any(answer is None for answer in selected_answers):
        raise ValueError("Select an answer for every question before checking.")
    feedback = []
    for question, selected in zip(questions, selected_answers):
        assert selected is not None
        correct = str(question["answer"]).casefold() == selected.casefold()
        explanation = str(question["explanation"])
        feedback.append((correct, explanation))
    score = sum(correct for correct, _ in feedback) / len(feedback) * 100
    return score, feedback
