# Local code lookup and edits

Use HarnessX's `MCPToolRegistry` with [Sem](https://github.com/Ataraxy-Labs/sem)
and [Weave](https://github.com/Ataraxy-Labs/weave). Install both first and ensure
`sem` and `weave-mcp` are on PATH. Install HarnessX from the repository root:

```bash
pip install -e .
python examples/sem_weave/code_tools.py /path/to/checkout parse_config
```

No model key is needed: this exercises the actual tool registry directly.
The Sem subprocess is pinned to the supplied checkout. Its local mode disables
its cloud client; local indexes may be written. These flags are not a sandbox.

To try editing a disposable fixture containing `def parse_config(value):`, put
this JSON array in `edits.json`:

```json
[{"file":"sample.py","entity":"parse_config","old":"value.strip()","new":"value.lstrip()"}]
```

```bash
# Preview only; leaves source unchanged.
python examples/sem_weave/code_tools.py /path/to/checkout parse_config --edits edits.json
# Explicitly write the same edits.
python examples/sem_weave/code_tools.py /path/to/checkout parse_config --edits edits.json --apply
```

Only the two named tools are called. The example does not pass a shell validation
command or commit changes. Inspect the diff and run appropriate project tests;
structural matching is not behavioral verification. A write failure can still
leave partial changes: use a disposable branch for experiments.

For an agent integration, keep the registry connected for the run and close it
in `finally`. Tool results can contain source code and reach the configured model
provider. Configure tools/approval policy appropriate to your repository before
exposing write-capable tools to a model.
