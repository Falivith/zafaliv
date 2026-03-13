from __future__ import annotations

import re

LETTER_PATTERN = re.compile(r"\b([A-E])\b", re.IGNORECASE)


def extract_choice(text: str | None) -> str | None:
    if not text:
        return None
    match = LETTER_PATTERN.search(text.strip().upper())
    if not match:
        return None
    return match.group(1).upper()


def compute_accuracy(rows: list[dict]) -> dict:
    total = len(rows)
    answered = sum(1 for row in rows if row["predicted_choice"] is not None)
    correct = sum(1 for row in rows if row["is_correct"])
    return {
        "num_examples": total,
        "num_answered": answered,
        "num_correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "answer_rate": (answered / total) if total else 0.0,
    }
