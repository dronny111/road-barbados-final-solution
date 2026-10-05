"""Dependency-free OCR error metrics.

The competition page says longer references receive more weight. The local
implementation therefore uses corpus (micro-averaged) error rates: summed edit
distance divided by summed reference length. Keep this implementation versioned
and do not silently normalize text before scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, TypeVar

T = TypeVar("T")


def edit_distance(reference: Sequence[T], prediction: Sequence[T]) -> int:
    """Return Levenshtein distance using O(min(n, m)) memory."""

    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous = list(range(len(prediction) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, pred_item in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ref_item != pred_item),
                )
            )
        previous = current
    return previous[-1]


def combined_error(reference: str, prediction: str) -> float:
    """The scorer's per-line 0.5/0.5 error, for ranking candidates within one row.

    Corpus scores stay micro-averaged through ``score_pairs``; this is the
    per-line view a preference pair or a reward needs.
    """

    reference_tokens, prediction_tokens = reference.split(), prediction.split()
    if not reference or not reference_tokens:
        raise ValueError("reference line is empty")
    cer = edit_distance(reference, prediction) / len(reference)
    wer = edit_distance(reference_tokens, prediction_tokens) / len(reference_tokens)
    return 0.5 * cer + 0.5 * wer


@dataclass(frozen=True)
class Score:
    cer: float
    wer: float
    combined: float
    character_edits: int
    reference_characters: int
    word_edits: int
    reference_words: int
    samples: int


def score_pairs(pairs: Iterable[tuple[str, str]]) -> Score:
    """Compute micro CER/WER and the official 0.5/0.5 combined score."""

    character_edits = 0
    reference_characters = 0
    word_edits = 0
    reference_words = 0
    samples = 0

    for reference, prediction in pairs:
        if not isinstance(reference, str) or not isinstance(prediction, str):
            raise TypeError("reference and prediction values must be strings")
        reference_tokens = reference.split()
        prediction_tokens = prediction.split()
        character_edits += edit_distance(reference, prediction)
        reference_characters += len(reference)
        word_edits += edit_distance(reference_tokens, prediction_tokens)
        reference_words += len(reference_tokens)
        samples += 1

    if samples == 0:
        raise ValueError("cannot score an empty collection")
    if reference_characters == 0 or reference_words == 0:
        raise ValueError("reference corpus must contain characters and words")

    cer = character_edits / reference_characters
    wer = word_edits / reference_words
    return Score(
        cer=cer,
        wer=wer,
        combined=0.5 * cer + 0.5 * wer,
        character_edits=character_edits,
        reference_characters=reference_characters,
        word_edits=word_edits,
        reference_words=reference_words,
        samples=samples,
    )
