import pytest
from harnessx.aegis.agents.critic import make_ask_evolver_tool


@pytest.mark.asyncio
async def test_ask_evolver_appends_qa_to_candidate(tmp_path):
    cand = tmp_path / "C-R5-02.md"
    cand.write_text("---\ncandidate_id: C-R5-02\n---\n\n## Root Cause\nX\n")

    async def fake_evolver_runner(cid: str, question: str) -> str:
        return "because of Y and [trajectories/t.jsonl#step_3]"

    ask = make_ask_evolver_tool(
        candidates_dir=tmp_path,
        evolver_runner=fake_evolver_runner,
        max_turns_per_candidate=2,
    )
    # ask is a Tool object; call its fn directly for test
    answer = await ask.fn(candidate_id="C-R5-02", question="Why will this not break task_07?")
    assert "step_3" in answer
    cand_text = cand.read_text()
    assert "## Ask-more Response" in cand_text
    assert "task_07" in cand_text


@pytest.mark.asyncio
async def test_ask_evolver_honors_max_turns(tmp_path):
    cand = tmp_path / "C-R5-02.md"
    cand.write_text("---\ncandidate_id: C-R5-02\n---\n\nbody\n")

    async def runner(cid, q):
        return "answer [trajectories/x.jsonl#step_1]"

    ask = make_ask_evolver_tool(
        candidates_dir=tmp_path,
        evolver_runner=runner,
        max_turns_per_candidate=2,
    )
    await ask.fn(candidate_id="C-R5-02", question="Q1")
    await ask.fn(candidate_id="C-R5-02", question="Q2")
    with pytest.raises(RuntimeError, match="max_turns"):
        await ask.fn(candidate_id="C-R5-02", question="Q3")
