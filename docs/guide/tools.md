# Tools

## Built-in tools

HarnessX ships nine built-in tools that mirror the Claude Code tool set.
Tool names are PascalCase and match what models trained on Claude Code natively expect.

| Name | Description |
|------|-------------|
| `Bash` | Execute shell commands. `timeout` (ms) optional, default 120 000, max 600 000 |
| `Read` | Read file contents. `limit`/`offset` for large files; `pages` (e.g. `"1-5"`) for PDFs |
| `Write` | Write or overwrite a file |
| `Edit` | Exact-string replacement in a file |
| `Glob` | Pattern-based file search (`**/*.py`) |
| `Grep` | Regex search. Supports `-i`, `-A`, `-B`, `-C`, `type`, `multiline`, `head_limit`, `offset` |
| `WebSearch` | Internet search via Tavily |
| `WebFetch` | Fetch and parse a URL |
| `Browser` | Playwright browser automation (screenshot, click, fill, navigate) |

### Register built-in tools

```python
from harnessx.tools.builtin import build_default_tools, build_web_tools

# Full set (filesystem + web + browser)
registry = build_default_tools()

# Web only (no filesystem)
registry = build_web_tools()
```

### Workspace isolation

Pass a `Workspace` to `HarnessConfig` — the filesystem tools automatically scope all
paths to `workspace.root` and reject escape attempts:

```python
from harnessx.workspace.workspace import Workspace
from harnessx.tools.builtin import build_default_tools
from harnessx import HarnessConfig
from pathlib import Path

ws = Workspace(root=Path("/tmp/sandbox"), agent_id="demo")
config = HarnessConfig(
    workspace=ws,
    tool_registry=build_default_tools(),
)
```

## Writing a custom tool

Use the `@tool` decorator — it infers the JSON Schema from type annotations automatically:

```python
from harnessx.tools.base import tool

@tool(description="Send an email to a recipient")
async def send_email(to: str, subject: str, body: str) -> str:
    """Sends an email and returns a confirmation ID."""
    # your implementation
    return f"sent:{message_id}"
```

### Manual schema override

For complex inputs, provide the schema explicitly:

```python
from harnessx.tools.base import Tool

search_tool = Tool(
    name="VectorSearch",
    description="Semantic search over the knowledge base",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 5},
            "filter": {"type": "object", "description": "Metadata filters"},
        },
        "required": ["query"],
    },
    fn=my_search_fn,
    tags=["retrieval"],
)
```

## Registering tools

```python
from harnessx import HarnessConfig
from harnessx.tools.inmemory import InMemoryToolRegistry

registry = InMemoryToolRegistry()
registry.register(send_email)
registry.register(search_tool)

config = HarnessConfig(tool_registry=registry)
```

Or add to an existing registry:

```python
registry = build_default_tools()
registry.register(send_email)
```

## Tool tags and filtering

Tags enable CE-level filtering — show a subset of tools depending on context:

```python
@tool(description="Query the production database", tags=["db", "readonly"])
async def db_query(sql: str) -> str: ...

from harnessx.processors.tools.strategies.tool_filter import TagToolFilter
from harnessx.bundles.context import make_context
from harnessx.core.builder import HarnessBuilder

config = (
    HarnessBuilder()
    | make_context(tool_filter=TagToolFilter(allowed_tags=["readonly"]))
).build()
```

## Controlling tool approval

`ToolWhitelistProcessor` intercepts `before_tool` events to approve or reject calls at
runtime. Rejected calls receive a synthetic result instead of executing:

```python
from harnessx.processors.tools.tool_whitelist import ToolWhitelistProcessor
from harnessx.bundles.context import make_context
from harnessx.core.builder import HarnessBuilder

config = (
    HarnessBuilder()
    | make_context()
).add(ToolWhitelistProcessor(allow=["Read", "Grep", "WebSearch"])).build()
```

## Calling tools from Python (testing / development)

Each built-in tool exposes its underlying async function via `.fn`:

```python
from harnessx.tools.builtin.grep_tool import grep_tool

result = await grep_tool.fn(
    pattern="def run_loop",
    path="harnessx/core",
    **{"-i": True, "type": "py"},
)
print(result)
```
