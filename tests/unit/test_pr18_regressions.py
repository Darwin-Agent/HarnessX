import hashlib
import ast
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import MagicMock

import pytest

from recipe.verl.launch.environment import apply_defaults
from recipe.slime.launch.runtime_environment import runtime_environment

ROOT = Path(__file__).resolve().parents[2]


def test_swebench_dataset_is_unmodified():
    dataset = ROOT / "benchmarks/swebench/swebench_lite_test.json"
    assert (
        hashlib.sha256(dataset.read_bytes()).hexdigest()
        == "b91794a6779207e6904d103280141657a6a23b529911d20bdc5e3b9bad3ba007"
    )
    rows = {row["instance_id"]: row for row in json.loads(dataset.read_text())}
    assert len(rows) == 300
    assert "dummify" in json.dumps(rows["sympy__sympy-15011"])
    assert "0.19.8.1" in json.dumps(rows["matplotlib__matplotlib-26020"])
    assert "InvalidSchema-10.122.1.1:3128/" in json.dumps(rows["psf__requests-3362"])


def test_environment_defaults_are_data_and_preserve_explicit_overrides():
    value = "quotes: \" ' dollar: $HOME $(touch should-not-exist)\nsecond line"
    environment = {"EXPLICIT": "keep me", "EMPTY": ""}
    apply_defaults({"DEFAULT": value, "EXPLICIT": "replace", "EMPTY": "fallback"}, environment)
    assert environment == {"DEFAULT": value, "EXPLICIT": "keep me", "EMPTY": "fallback"}


def test_ray_runtime_json_preserves_quotes_and_dollars():
    value = 'quoted "path" \\ $HOME $(no-command)\nnext'
    result = json.loads(json.dumps(runtime_environment([f"VALUE={value}", "EQUALS=a=b"], value)))
    assert result == {"working_dir": value, "env_vars": {"VALUE": value, "EQUALS": "a=b"}}


@pytest.mark.parametrize(
    "script",
    [
        "recipe/slime/launch/data_prep.sh",
        "recipe/slime/launch/run_math_rl.sh",
        "recipe/slime/launch/run_math_sft.sh",
        "recipe/verl/launch/run_research_rl.sh",
    ],
)
@pytest.mark.parametrize("configured", [None, "", "https://mirror.example.invalid"])
def test_huggingface_endpoint_defaults_and_mirrors(script, configured):
    source = (ROOT / script).read_text()
    assignment = next(line for line in source.splitlines() if line.startswith("export HF_ENDPOINT="))
    environment = os.environ.copy()
    environment.pop("HF_ENDPOINT", None)
    if configured is not None:
        environment["HF_ENDPOINT"] = configured
    result = subprocess.run(
        ["bash", "-c", assignment + '\nprintf "%s" "$HF_ENDPOINT"'],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == (configured or "https://huggingface.co")


@pytest.mark.parametrize("script", ["run_math_rl.sh", "run_math_sft.sh"])
def test_local_ray_network_defaults_do_not_use_proxy(script):
    source = (ROOT / "recipe/slime/launch" / script).read_text()
    assignments = "\n".join(
        next(line for line in source.splitlines() if line.startswith(prefix))
        for prefix in ("export MASTER_ADDR=", "export no_proxy=")
    )
    environment = os.environ.copy()
    environment.pop("MASTER_ADDR", None)
    environment["no_proxy"] = "existing.example"
    result = subprocess.run(
        ["bash", "-c", assignments + '\nprintf "%s\\n%s" "$MASTER_ADDR" "$no_proxy"'],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    address, bypass = result.stdout.splitlines()
    assert address == "127.0.0.1"
    assert {"127.0.0.1", "localhost", "existing.example"}.issubset(bypass.split(","))
    assert "pkill" not in source


def test_docker_cdp_listens_on_container_interface_but_publishes_only_loopback(monkeypatch):
    from harnessx.sandbox.docker import DockerSandboxProvider

    provider = DockerSandboxProvider(enable_browser=True)
    client = MagicMock()
    monkeypatch.setattr(provider, "_get_client", lambda: client)
    monkeypatch.setattr(provider, "_pull_if_needed", lambda: None)
    provider._start_container(None, None)
    arguments = client.containers.run.call_args.kwargs
    assert "--remote-debugging-address=0.0.0.0" in arguments["command"]
    assert arguments["ports"]["9222/tcp"] == ("127.0.0.1", 0)
    assert arguments["network_mode"] == "bridge"


def test_gateway_and_lab_preserve_network_bind_defaults():
    gateway = ast.parse((ROOT / "gateway/main.py").read_text())
    run = next(node for node in gateway.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "_run")
    defaults = dict(zip([argument.arg for argument in run.args.args][-len(run.args.defaults) :], run.args.defaults))
    assert ast.literal_eval(defaults["host"]) == "0.0.0.0"
    lab = ast.parse((ROOT / "harnessx/cli.py").read_text())
    configurations = [
        node
        for node in ast.walk(lab)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "Config"
    ]
    assert any(
        any(
            keyword.arg == "host" and isinstance(keyword.value, ast.Constant) and keyword.value.value == "0.0.0.0"
            for keyword in call.keywords
        )
        for call in configurations
    )


@pytest.mark.parametrize(
    "configured, expected",
    [
        (None, "http://127.0.0.1:12081"),
        ("", "http://127.0.0.1:12081"),
        ("https://sandbox.example", "https://sandbox.example"),
    ],
)
def test_opensandbox_default_and_explicit_server(monkeypatch, configured, expected):
    source = ast.parse((ROOT / "benchmarks/terminal_bench_2/opensandbox/environment.py").read_text())
    assignment = next(
        node
        for node in source.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_DEFAULT_SERVER" for target in node.targets)
    )
    monkeypatch.delenv("OPENSANDBOX_URL", raising=False)
    if configured is not None:
        monkeypatch.setenv("OPENSANDBOX_URL", configured)
    namespace = {"os": os}
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "server-default", "exec"), namespace)
    assert namespace["_DEFAULT_SERVER"] == expected
