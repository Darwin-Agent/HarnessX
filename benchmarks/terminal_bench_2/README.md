# Terminal Bench 2.0

Run Terminal Bench 2.0 challenges through HarnessX. Challenges are bash/file-system
tasks executed inside Harbor-managed containers (`terminal-bench@2.0`, 89 tasks).

## Results

### claude-opus-4-6 — OpenSandbox

**Model**: `claude-opus-4-6`
**Sandbox**: OpenSandbox (self-hosted)
**Overall**: **56 / 89 = 63.0%**

| Category | Count |
|----------|-------|
| Total tasks | 89 |
| Pass (reward = 1.0) | **56** |
| Fail (reward = 0.0, verifier ran) | 30 |
| Infrastructure error (no score) | 3 |

> Single full run (n=2, all 89 tasks).
> Official Claude Code + Opus-4-6: **58.0% ± 2.9%** (k=5) — HarnessX exceeds by ~5pp.

<details>
<summary>Task-level breakdown</summary>

#### Pass (56)
`bn-fit-modify`, `break-filter-js-from-html`, `build-cython-ext`, `build-pov-ray`,
`chess-best-move`, `circuit-fibsqrt`, `cobol-modernization`, `code-from-image`,
`compile-compcert`, `constraints-scheduling`, `count-dataset-tokens`, `crack-7z-hash`,
`custom-memory-heap-crash`, `distribution-search`, `dna-insert`, `extract-elf`,
`feal-differential-cryptanalysis`, `financial-document-processor`, `fix-code-vulnerability`,
`fix-git`, `fix-ocaml-gc`, `git-leak-recovery`, `gpt2-codegolf`, `install-windows-3.11`,
`large-scale-text-editing`, `largest-eigenval`, `llm-inference-batching-scheduler`,
`log-summary-date-ranges`, `mailman`, `mcmc-sampling-stan`, `merge-diff-arc-agi-task`,
`modernize-scientific-stack`, `openssl-selfsigned-cert`, `overfull-hbox`, `password-recovery`,
`path-tracing`, `path-tracing-reverse`, `portfolio-optimization`, `prove-plus-comm`,
`pytorch-model-cli`, `pytorch-model-recovery`, `qemu-alpine-ssh`, `qemu-startup`, `regex-log`,
`reshard-c4-data`, `rstan-to-pystan`, `sanitize-git-repo`, `schemelike-metacircular-eval`,
`sparql-university`, `sqlite-db-truncate`, `sqlite-with-gcov`, `torch-tensor-parallelism`,
`tune-mjcf`, `vulnerable-secret`, `winning-avg-corewars`, `write-compressor`

#### Fail — verifier ran (30)
`adaptive-rejection-sampler`, `build-pmars`, `caffe-cifar-10`, `cancel-async-tasks`,
`configure-git-webserver`, `db-wal-recovery`, `dna-assembly`, `extract-moves-from-video`,
`feal-linear-cryptanalysis`, `filter-js-from-html`, `gcode-to-text`, `git-multibranch`,
`headless-terminal`, `hf-model-inference`, `kv-store-grpc`, `make-doom-for-mips`,
`make-mips-interpreter`, `model-extraction-relu-logits`, `mteb-retrieve`,
`multi-source-data-merger`, `nginx-request-logging`, `polyglot-c-py`, `polyglot-rust-c`,
`protein-assembly`, `pypi-server`, `raman-fitting`, `sam-cell-seg`, `torch-pipeline-parallelism`,
`train-fasttext`, `video-processing`

#### Infrastructure error — no score (3)
`mteb-leaderboard` (AddTestsDirError), `regex-chess` (AddTestsDirError),
`query-optimize` (VerifierTimeout)

</details>

---

### claude-haiku-4-5 — OpenSandbox

**Model**: `claude-haiku-4-5-20251001`
**Sandbox**: OpenSandbox (self-hosted)
**Overall**: **28 / 89 = 31.5%**

| Category | Count |
|----------|-------|
| Total tasks | 89 |
| Pass (reward = 1.0) | **28** |
| Fail (reward = 0.0, verifier ran) | 58 |
| Infrastructure error (no score) | 3 |

<details>
<summary>Task-level breakdown</summary>

#### Pass (28)
`adaptive-rejection-sampler`, `bn-fit-modify`, `build-pmars`, `cancel-async-tasks`,
`cobol-modernization`, `compile-compcert`, `constraints-scheduling`, `crack-7z-hash`,
`custom-memory-heap-crash`, `distribution-search`, `extract-elf`, `financial-document-processor`,
`fix-code-vulnerability`, `fix-git`, `git-leak-recovery`, `log-summary-date-ranges`,
`merge-diff-arc-agi-task`, `modernize-scientific-stack`, `multi-source-data-merger`,
`overfull-hbox`, `password-recovery`, `portfolio-optimization`, `pytorch-model-cli`,
`pytorch-model-recovery`, `regex-log`, `schemelike-metacircular-eval`, `sparql-university`,
`vulnerable-secret`

#### Fail — verifier ran (58)
`break-filter-js-from-html`, `build-cython-ext`, `caffe-cifar-10`, `chess-best-move`,
`circuit-fibsqrt`, `code-from-image`, `configure-git-webserver`, `count-dataset-tokens`,
`db-wal-recovery`, `dna-assembly`, `dna-insert`, `extract-moves-from-video`,
`feal-differential-cryptanalysis`, `feal-linear-cryptanalysis`, `filter-js-from-html`,
`fix-ocaml-gc`, `gcode-to-text`, `git-multibranch`, `gpt2-codegolf`, `headless-terminal`,
`hf-model-inference`, `install-windows-3.11`, `kv-store-grpc`, `large-scale-text-editing`,
`largest-eigenval`, `llm-inference-batching-scheduler`, `mailman`, `make-doom-for-mips`,
`make-mips-interpreter`, `mcmc-sampling-stan`, `mteb-leaderboard`, `mteb-retrieve`,
`nginx-request-logging`, `openssl-selfsigned-cert`, `path-tracing`, `path-tracing-reverse`,
`polyglot-c-py`, `polyglot-rust-c`, `protein-assembly`, `prove-plus-comm`, `pypi-server`,
`qemu-alpine-ssh`, `qemu-startup`, `raman-fitting`, `regex-chess`, `reshard-c4-data`,
`rstan-to-pystan`, `sam-cell-seg`, `sanitize-git-repo`, `sqlite-db-truncate`, `sqlite-with-gcov`,
`torch-pipeline-parallelism`, `torch-tensor-parallelism`, `train-fasttext`, `tune-mjcf`,
`video-processing`, `winning-avg-corewars`, `write-compressor`

#### Infrastructure error — no score (3)
`build-pov-ray` (RewardFileNotFoundError), `model-extraction-relu-logits` (VerifierTimeout),
`query-optimize` (VerifierTimeout)

</details>

---

## Evaluation environments

Four sandbox backends are supported:

| Environment | Internet access | Notes |
|-------------|----------------|-------|
| **OpenSandbox** | depends on server | Self-hosted; full control over networking and resources |
| **Local Docker** | depends on host | Fastest to set up; requires Docker on the eval machine |
| **Daytona** | no (Tier 1/2) | Managed cloud; blocks general outbound HTTPS |
| **Modal** | yes | Managed cloud; suited for internet-dependent tasks |

For Local Docker, Daytona, Modal setup and the recommended Modal+Daytona split-backend
parallel run, see [`scripts/README.md`](scripts/README.md).

---

## Running with OpenSandbox

Set credentials and the sandbox URL:

```bash
export ANTHROPIC_API_KEY=sk-...
export OPENSANDBOX_URL=http://your-server:13081
```

Run all 89 tasks (n=2 concurrent):

```bash
bash benchmarks/terminal_bench_2/scripts/eval_opensandbox.sh
```

Single task:

```bash
bash benchmarks/terminal_bench_2/scripts/eval_opensandbox.sh -t crack-7z-hash
```

Resume an interrupted run:

```bash
bash benchmarks/terminal_bench_2/scripts/eval_opensandbox.sh --job-name my-run --resume
```

All scripts forward extra flags to `tb2_eval.py` (`--job-name`, `--resume`, `-n`, `-t`, `--max-steps`, etc.).

---

## Harness features

- **System prompt**: 9-item workflow checklist — explore first, decompose complex tasks,
  verify output files with `ls`, keep background services running, skip `apt-get update`
- **Bash-only tool set**: matches the official Harbor evaluation (no file browser, no web tools)
- **TaskTimeReminderProcessor**: time-based warnings at 70% and 90% of task wall-clock timeout;
  90% warning tells the model to write output files immediately or score 0
- **CustomSelfVerifyProcessor**: injects a one-shot verification checklist when the model
  attempts to exit without tool calls — fires at most once per task
- **CompactionProcessor**: auto-compacts context at 140k tokens or 100 messages
- **PostCompactionRefreshProcessor**: re-injects `ls -la /app` workspace snapshot after
  compaction so the model knows current file state
- **CustomEditToolProcessor**: tracks per-file Bash write counts; warns when the same file
  is written more than 7 times, then resets the counter
- **BgInstallGuard**: intercepts background package-manager/build commands (`apt-get ... &`)
  and injects a corrective warning, preventing dpkg lock failures
- **ToolCallCorrectionLayer** + **ParseRetryProcessor**: fixes malformed tool calls and
  retries on parse failures
- Bash tool routes all commands via `environment.exec()` (Harbor's container API)
- stdout/stderr output capped at 8 000 characters per call
- Trajectories saved under `agent/oh_runs/` in the Harbor output directory
