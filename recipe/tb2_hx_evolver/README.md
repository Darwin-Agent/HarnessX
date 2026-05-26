# TB2 HarnessX Evolver

自动化多轮 harness 进化实验：DigestAgent 分析失败轨迹 → EvolveAgent 生成改进配置 → TB2 重新评估，循环 N 轮。

---

## 实验设置

### 进化任务集

从 TB2 全量 89 个任务中，按以下条件采样固定实验集：

- **过滤条件**：agent timeout ≤ 900s（15 分钟）
- **采样数量**：16 个任务
- **随机种子**：42

生成的任务集：`tasks_sample16_seed42_lt15m.json`（已提交）

### 进化 Agent 模型

| Agent | 模型 | 用途 |
|-------|------|------|
| DigestAgent | `ppio/pa/claude-haiku-4-5-20251001` | 轨迹分析、失败模式提炼 |
| EvolveAgent | `ppio/pa/claude-sonnet-4-6` | 生成改进 config / processor |

### 待进化模型（task-agent）

通过 `TB2_MODEL` / `TB2_API_BASE` / `TB2_API_KEY` 指定，默认配置为 Qwen3.5-27B。

---

## 环境安装

### 1. Python 依赖

```bash
# 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

cd HarnessX
uv venv --python 3.12 .venv
source .venv/bin/activate

# 安装 harnessx（editable）
uv pip install -e .
```

### 2. Harbor（TB2 评估运行时）

```bash
uv pip install harbor==0.2.0

# 验证 harbor 可用，并预拉取实验任务镜像
harbor tasks pull --task-list recipe/tb2_hx_evolver/tasks_sample16_seed42_lt15m.json
```

### 3. Docker Compose v2

TB2 本地评估依赖 Docker Compose v2：

```bash
docker compose version   # 必须是 v2.x
```

---

## 复现任务采样

如需重新生成 `tasks_sample16_seed42_lt15m.json`（结果与已提交文件完全一致）：

```bash
python recipe/tb2_hx_evolver/scripts/sample_tasks.py \
    --seed 42 --n 16 --max-agent-timeout 900 \
    --output recipe/tb2_hx_evolver/tasks_sample16_seed42_lt15m.json
```

脚本直接从 harbor cache（`/root/.cache/harbor/tasks`）发现任务，不依赖 `recipe/tb2_evolver/`。

---

## 配置 evol.env

```bash
cp recipe/tb2_hx_evolver/scripts/evol.env.example recipe/tb2_hx_evolver/scripts/evol.env
```

编辑 `evol.env`，填写以下关键字段：

```dotenv
# ── Evol agent API（DigestAgent / EvolveAgent，Anthropic 兼容接口）
ANTHROPIC_BASE_URL=http://model.mify.ai.srv/anthropic
ANTHROPIC_API_KEY=sk-xxxxx

# ── 待进化的 task-agent 模型
TB2_MODEL=qwen3.5-27b
TB2_API_BASE=http://<sglang-host>:<port>/v1
TB2_API_KEY=sk-xxxxx

# ── Evol agent 模型（默认无需修改）
EVOL_DIGEST_MODEL=ppio/pa/claude-haiku-4-5-20251001
EVOL_EVOLVE_MODEL=ppio/pa/claude-sonnet-4-6

# ── 实验任务集（默认已指向 16-task 采样集）
EVOL_TASKS=recipe/tb2_hx_evolver/tasks_sample16_seed42_lt15m.json

# ── 基准 harness config
EVOL_CONFIG=benchmarks/terminal_bench_2/harness_baseline_config.yaml

# ── 进化轮数与并发
EVOL_NUM_ROUNDS=5
EVOL_CONCURRENT=4
EVOL_K=3
```

完整参数说明见 `scripts/evol.env.example`。

---

## 运行进化实验

### 全量多轮进化（推荐）

```bash
# 脚本自动加载同目录下的 evol.env
bash recipe/tb2_hx_evolver/scripts/run_full_evol.sh

# 或显式指定 .env
bash recipe/tb2_hx_evolver/scripts/run_full_evol.sh --env recipe/tb2_hx_evolver/scripts/evol.env
```

### Round-0 暖启动（跳过首轮评估，复用已有轨迹）

```bash
bash recipe/tb2_hx_evolver/scripts/run_full_evol.sh \
    --r0-trials .benchmarks/tb2/qwen3.5-27b-local-k5-0511
```

### 断点续跑

```bash
bash recipe/tb2_hx_evolver/scripts/run_full_evol.sh \
    --resume --run-dir .benchmarks/evolve-runs/evolve-20260513-120000
```

### 单轮进化测试（仅运行一轮 DigestAgent + EvolveAgent，不跑评估）

```bash
bash recipe/tb2_hx_evolver/scripts/run_evol_hx.sh \
    --trials-dir .benchmarks/tb2/qwen3.5-27b-local-k5-0511
```

---

## 输出目录结构

```
.benchmarks/evolve-runs/evolve-<timestamp>/
  state.json                    当前轮次 + 当前 config 路径（每轮更新）
  solvability.json              跨轮可解性追踪
  current_config.yaml           → symlink 指向最新接受的 config
  round_000/
    trials/
      evolve-r0-<run-name>/     TB2 评估输出（{task}__{trial}/agent/oh_runs/...）
    evolve/
      mechanical_signals.json   Layer 1 机械信号
      digest_report.json        DigestAgent 分析报告
      change_manifest.json      EvolveAgent 变更清单
      validation_report.json    验证结果
      evolve_result.json        {accepted, pass_rate, new_config_path, ...}
      evol-workspace/
        target_config.yaml      进化后的 harness config
        processors/             新增自定义 processor .py 文件
  round_001/
    ...
```

---

## 目录说明

```
recipe/tb2_hx_evolver/
├── run_full_evol.py                    多轮进化主程序
├── test_evol_hx.py                     单轮进化测试入口
├── score.py                            TB2 任务评分适配器
├── tasks_sample16_seed42_lt15m.json    实验任务集（16 tasks，seed=42）
└── scripts/
    ├── run_full_evol.sh                多轮进化脚本（推荐入口）
    ├── run_evol_hx.sh                  单轮进化脚本（调试用）
    ├── sample_tasks.py                 任务采样脚本
    ├── evol.env.example                .env 配置模板
    └── evol.env                        本地配置（不提交）
```
