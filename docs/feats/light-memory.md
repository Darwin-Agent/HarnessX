# Light-Memory  *(self-developed)*

> **Memory dimension (3)** · `harnessx/plugins/dimensions/light_memory/`

Light-Memory is an **original memory system** built entirely within this repository.
It is designed for personal-assistant scenarios where a single agent accumulates and recalls knowledge over long periods, without requiring any external vector database.

## How It Works

```
Task Prompt
    │
    ▼
[Keyword Extraction + Entity Augmentation]
    │
    ▼
[Grep Memory Files → Match Scores]
    │
    ▼
[Exponential Time-Decay Ranking]           ← score = importance × e^(−λ × days_since_access)
    │
    ▼
[Top-K Candidate Table → System Prompt]
    │
    ▼
Agent reads / writes / updates memory files using its built-in tools (Read, Write, Edit)
    │
    ▼
[Daily Compression]  +  [Background Organization]  +  [Git Commit]
```

## Key Mechanisms

| Mechanism | Detail |
|-----------|--------|
| **File-based storage** | Each memory is a Markdown file with YAML frontmatter — no external DB, no binary index |
| **Memory types** | `style`, `profile`, `session`, `skill`, `learning`, `entity`, `daily` |
| **Exponential decay** | `importance × exp(−ln2 × days / half_life)` — older, unaccessed memories fade naturally |
| **Daily compression** | Conversation turns are compressed into a structured daily log at session end |
| **Background organization** | A background coroutine periodically consolidates and deduplicates memory files |
| **Git versioning** | Optional git integration — every memory write is auto-committed for full history |
| **Agent-driven recall** | The agent reads memory files directly using its own tools; no opaque retrieval black-box |
| **Zero dependencies** | Only Python stdlib + filesystem; no ChromaDB, no Pinecone, no Redis required |

## Usage

```python
from harnessx.plugins.dimensions.light_memory import LightMemoryPlugin

plugin = LightMemoryPlugin(
    memory_root="~/.harnessx/memory",
    half_life_days=30,      # how fast memories fade
    top_k=15,               # max candidates injected per turn
    auto_recall=True,       # inject recall candidates at task start
    auto_capture=False,     # let the agent decide when to write memories
    auto_commit=True,       # git-commit each memory write
)

builder = HarnessBuilder().plugin(plugin)
```
