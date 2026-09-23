from typing import Any

from opik.evaluation.metrics import BaseMetric
from opik.evaluation.metrics.score_result import ScoreResult


class SummaryDensityHeuristic(BaseMetric):
    """Score whether an answer's character length is within an ideal range."""

    def __init__(
        self,
        name: str = "summary_density_heuristic",
        min_length: int = 128,
        max_length: int = 1024,
        *,
        track: bool = True,
        project_name: str | None = None,
    ) -> None:
        if min_length <= 0:
            raise ValueError("min_length must be greater than zero.")
        if max_length < min_length:
            raise ValueError("max_length must be greater than or equal to min_length.")

        super().__init__(name=name, track=track, project_name=project_name)
        self.min_length = min_length
        self.max_length = max_length

    def score(
        self,
        input: str,
        output: str,
        **ignored_kwargs: Any,
    ) -> ScoreResult:
        length_score = self._compute_length_score(output)

        if length_score == 1.0:
            assessment = "Length is within the ideal range."
        elif length_score >= 0.5:
            assessment = "Length is slightly outside the ideal range."
        else:
            assessment = "Length is significantly outside the ideal range."

        return ScoreResult(
            name=self.name,
            value=length_score,
            reason=f"Output length: {len(output)} characters. {assessment}",
        )

    def _compute_length_score(self, text: str) -> float:
        length = len(text)
        if self.min_length <= length <= self.max_length:
            return 1.0

        if length < self.min_length:
            deviation = (self.min_length - length) / self.min_length
        else:
            deviation = (length - self.max_length) / self.max_length

        return max(0.0, 1.0 - deviation)
