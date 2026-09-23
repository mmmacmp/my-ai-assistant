import pytest

from my_ai_assistant_online.application.evaluation_heuristics import (
    SummaryDensityHeuristic,
)


@pytest.fixture
def metric() -> SummaryDensityHeuristic:
    return SummaryDensityHeuristic(
        min_length=128,
        max_length=1024,
        track=False,
    )


def test_density_heuristic_gives_full_score_inside_range(
    metric: SummaryDensityHeuristic,
) -> None:
    result = metric.score(input="Question", output="x" * 128)

    assert result.value == 1.0
    assert "within the ideal range" in result.reason


def test_density_heuristic_penalizes_short_output(
    metric: SummaryDensityHeuristic,
) -> None:
    result = metric.score(input="Question", output="x" * 64)

    assert result.value == 0.5
    assert "slightly outside" in result.reason


def test_density_heuristic_clamps_very_long_output_to_zero(
    metric: SummaryDensityHeuristic,
) -> None:
    result = metric.score(input="Question", output="x" * 2048)

    assert result.value == 0.0
    assert "significantly outside" in result.reason


@pytest.mark.parametrize(
    ("min_length", "max_length", "message"),
    [(0, 100, "greater than zero"), (100, 99, "greater than or equal")],
)
def test_density_heuristic_rejects_invalid_boundaries(
    min_length: int,
    max_length: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SummaryDensityHeuristic(
            min_length=min_length,
            max_length=max_length,
            track=False,
        )
