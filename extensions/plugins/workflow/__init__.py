"""WorkflowPlugin — procedural memory: learn and replay complex task workflows.

The plugin watches multi-turn conversations and, when a complex task completes,
internalises the procedure as a YAML workflow file on disk.  On future similar
tasks it injects matching workflows into the system prompt so the agent can
call ``flow_exec`` to replay the procedure directly (saving reasoning tokens).

Quick start::

    from extensions.plugins.workflow import WorkflowPlugin
    from harnessx.core.builder import HarnessBuilder
    from harnessx.bundles.control import make_context, reliability

    config = (
        HarnessBuilder()
        | make_context()
        | reliability
    ).plugin(WorkflowPlugin(
        judge_model="claude-haiku-4-5-20251001",
        extractor_model="claude-haiku-4-5-20251001",
    )).build()
"""

from .plugin import WorkflowPlugin

__all__ = ["WorkflowPlugin"]
