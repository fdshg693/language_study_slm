from __future__ import annotations

"""Abstract interface for integrating a small language model (SLM)
into a language-learning application.

This module defines data structures representing exercises and their
responses, along with an abstract base class that SLM implementations can
inherit from. Concrete implementations are expected to communicate with a
specific small language model in order to generate exercises, evaluate
learner responses, and provide feedback or hints.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Exercise:
    """Represents a single learning exercise.

    Attributes
    ----------
    prompt:
        The text presented to the learner.
    options:
        Optional answer choices for multiple-choice style questions. When
        ``None`` the learner should produce a free-form answer.
    answer:
        The expected answer. Its type is intentionally vague to allow for
        string answers, structured data, or other formats depending on the
        exercise type.
    metadata:
        Additional information about the exercise such as difficulty,
        vocabulary focus, etc.
    """

    prompt: str
    options: Optional[List[str]] = None
    answer: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Evaluation:
    """Result of evaluating a learner's response."""

    is_correct: bool
    score: float
    feedback: Optional[str] = None


class SLMInterface(ABC):
    """Abstract interface for SLM powered language-learning features."""

    @abstractmethod
    def generate_exercise(self, *, exercise_type: str, level: str, **kwargs: Any) -> Exercise:
        """Create a new exercise.

        Parameters
        ----------
        exercise_type:
            The category of exercise to generate (e.g. ``"fill_blank"``,
            ``"vocab_quiz"``).
        level:
            Difficulty or proficiency level of the learner.
        **kwargs:
            Additional model specific parameters.
        """

    @abstractmethod
    def evaluate_response(self, exercise: Exercise, response: str, **kwargs: Any) -> Evaluation:
        """Assess a learner's response to an exercise."""

    @abstractmethod
    def provide_feedback(self, exercise: Exercise, response: str, **kwargs: Any) -> str:
        """Offer hints or feedback for an exercise given a learner response."""
