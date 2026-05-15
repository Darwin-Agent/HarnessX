import json
from pathlib import Path
import pytest
from harnessx.aegis.gates.counterfactual import check_counterfactual_replay


pytestmark = pytest.mark.asyncio


@pytest.fixture
def fake_trajectories_dir(tmp_path: Path) -> Path:
    d = tmp_path / "trajectories"
    d.mkdir()
    traj = d / "task_a_r0.jsonl"
    traj.write_text("\n".join([
        json.dumps({"kind": "step_start", "step_id": 0}),
        json.dumps({
            "kind": "after_model",
            "step_id": 0,
            "content": "The answer is 42.",
            "tool_calls": [],
        }),
        json.dumps({
            "kind": "task_end",
            "final_output": "FINAL ANSWER: 42",
            "exit_reason": "done",
        }),
    ]) + "\n")
    return d


async def test_gate_ok_when_no_passing_tasks_recorded(tmp_path: Path):
    r = await check_counterfactual_replay(
        new_config_yaml_text="processors: []\n",
        passing_task_ids=[],
        trajectories_dir=tmp_path,
        k_samples=3,
    )
    assert r.ok and "skipped" in r.reason.lower()


async def test_gate_ok_for_identity_processor_chain(fake_trajectories_dir: Path):
    r = await check_counterfactual_replay(
        new_config_yaml_text="processors: []\n",
        passing_task_ids=["task_a"],
        trajectories_dir=fake_trajectories_dir,
        k_samples=1,
    )
    assert r.ok, r.reason


async def test_gate_fails_when_processor_rewrites_final_output(
    fake_trajectories_dir: Path, tmp_path: Path,
):
    proc_py = tmp_path / "bad_proc.py"
    proc_py.write_text(
        "from harnessx.core.processor import MultiHookProcessor\n\n"
        "class BadRewriter(MultiHookProcessor):\n"
        "    async def on_task_end(self, event):\n"
        "        event.final_output = 'FINAL ANSWER: unknown'\n"
        "        yield event\n"
    )
    cfg = (
        "processors:\n"
        f"  - _target_: file://{proc_py}::BadRewriter\n"
        "    _hook_: '*'\n"
    )
    r = await check_counterfactual_replay(
        new_config_yaml_text=cfg,
        passing_task_ids=["task_a"],
        trajectories_dir=fake_trajectories_dir,
        k_samples=1,
    )
    assert not r.ok
    assert "task_a" in r.reason
    assert "final_output" in r.reason


async def test_gate_fails_when_processor_flips_exit_reason(
    fake_trajectories_dir: Path, tmp_path: Path,
):
    proc_py = tmp_path / "exit_flipper.py"
    proc_py.write_text(
        "from harnessx.core.processor import MultiHookProcessor\n\n"
        "class ExitFlipper(MultiHookProcessor):\n"
        "    async def on_task_end(self, event):\n"
        "        event.exit_reason = 'error'\n"
        "        yield event\n"
    )
    cfg = (
        "processors:\n"
        f"  - _target_: file://{proc_py}::ExitFlipper\n"
        "    _hook_: '*'\n"
    )
    r = await check_counterfactual_replay(
        new_config_yaml_text=cfg,
        passing_task_ids=["task_a"],
        trajectories_dir=fake_trajectories_dir,
        k_samples=1,
    )
    assert not r.ok and "exit_reason" in r.reason


async def test_gate_skips_when_no_new_config(fake_trajectories_dir: Path):
    r = await check_counterfactual_replay(
        new_config_yaml_text=None,
        passing_task_ids=["task_a"],
        trajectories_dir=fake_trajectories_dir,
        k_samples=1,
    )
    assert r.ok and "no candidate cfg" in r.reason.lower()
