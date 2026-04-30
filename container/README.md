# HarnessX Agent Container

Base Docker image providing the isolated execution environment for agent tools.

## Build

```bash
docker build -t harnessx/agent:latest container/
```

## Usage

```python
from harnessx.sandbox.docker import DockerSandboxProvider
from harnessx import Harness, HarnessConfig, BaseTask

config = HarnessConfig(
    sandbox_provider=DockerSandboxProvider(
        image="harnessx/agent:latest",
        network="none",      # fully isolated (default)
        mem_limit="2g",
        cpu_count=2,
    ),
    ...
)
harness = Harness(config=config)
result = await harness.run(BaseTask(description="..."))
```

## Warm pool (personal assistant)

Pass `sandbox_hint_id` to reuse a running container across turns:

```python
config = HarnessConfig(
    sandbox_provider=DockerSandboxProvider(image="harnessx/agent:latest"),
    sandbox_hint_id="user-alice",   # same container reused every turn
    ...
)
```

## Included tools

| Category | Packages |
|----------|----------|
| Shell | bash, coreutils, findutils, procps |
| VCS | git |
| Search | ripgrep |
| Network | curl, wget |
| Data | jq, numpy, pandas, matplotlib |
| Documents | python-pptx, python-docx, openpyxl, libreoffice-core |
| Node | nodejs, npm |
