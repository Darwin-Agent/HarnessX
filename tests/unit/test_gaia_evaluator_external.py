import pytest

from benchmarks.gaia.evaluator import GAIAPipelineEvaluator


@pytest.mark.asyncio
async def test_evaluate_answer_exact_match():
    ev = GAIAPipelineEvaluator()
    result = await ev.evaluate_answer(
        final_output="The answer is 42.",
        ground_truth="42",
    )
    assert result.passed is True
    assert result.score == 1.0
    assert result.reward == 1.0


@pytest.mark.asyncio
async def test_evaluate_answer_wrong():
    ev = GAIAPipelineEvaluator()
    result = await ev.evaluate_answer(
        final_output="The answer is clearly 7.",
        ground_truth="42",
    )
    assert result.passed is False
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_evaluate_answer_empty_output():
    ev = GAIAPipelineEvaluator()
    result = await ev.evaluate_answer(final_output="", ground_truth="42")
    assert result.passed is False
    assert "no answer" in result.reason.lower()


@pytest.mark.asyncio
async def test_evaluate_answer_no_ground_truth_no_judge():
    ev = GAIAPipelineEvaluator()  # no judge
    result = await ev.evaluate_answer(
        final_output="Something",
        ground_truth="",
    )
    assert result.passed is False
    assert "no ground-truth" in result.reason.lower()
