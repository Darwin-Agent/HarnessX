# PR #18 remediation record

## Scope and status

These changes address the supplied PR #18 review in the merged HarnessX tree.
The existing frontend internationalization and independent harness-evolution
code remain in place. This record describes local changes and local validation,
not a pushed revision, a GitHub CI result, or a completed GPU training run.

## Review disposition

| Review finding | Resolution |
| --- | --- |
| Unsupported SGLang provider constructor arguments | Implemented tool-call format, tool-result role, thinking-prefix handling, and strict training logprob requirements in the shared provider; propagated the settings to child agents. |
| Every failed rollout becomes a trainable EOS | Failure outputs now have zero response masks and zero reward. Interface/configuration errors propagate rather than silently becoming samples. Completely failed batches are rejected. |
| Invented or malformed behavior logprobs | Generated training tokens require complete, finite, non-positive captured logprobs with matching token IDs. Missing/corrupt values invalidate the rollout; tool-result tokens remain masked. |
| SWE-bench data damaged by global sanitization | Restored the complete 300-row file byte-for-byte from the old framework and added a SHA256 regression test, including the damaged SymPy, requests and version examples. |
| Credentials cannot round-trip | Verified that explicit `include_credentials=True` returns the actual supplied credential; tests now use an unredacted synthetic value and verify `from_dict` round-tripping. Default serialization still omits it. |
| Empty `HF_ENDPOINT` breaks downloads | All four launch/data-preparation scripts default unset or empty values to the public Hugging Face endpoint and retain explicit mirrors. |
| Docker browser CDP binds container loopback | Chromium listens on all container interfaces again; its published host-side port remains restricted to loopback. |
| Gateway / lab default bind behavior changed | Restored the previous all-interface defaults and matching documentation. A future loopback-default change is deliberately not bundled into this remediation. |
| Slime loopback traffic sent through proxies / invalid Ray node address | Restored loopback bypasses and an IP-valued default `MASTER_ADDR`. Separated Ray dashboard bind and client addresses. |
| Keyless parallel-search example requires configuration | Restored the public MCP endpoint default while preserving `MCP_SEARCH_URL` overrides. |
| Slack token types and miscellaneous prose damaged | Restored `xoxb-`/`xapp-` guidance, repaired tau2 comments and terminal-bench help text, and restored the local OpenSandbox endpoint. |
| Modified veRL copied without provenance or licenses | Removed the checked-in fork; replaced it with a checksum-verified upstream pin, a seven-file patch, upstream Apache-2.0 LICENSE and Notice, and documented patch purposes. |
| Launcher rewrites installed dependencies / swallows patch errors | Removed the runtime patch package, fake compatibility interfaces, hardcoded site-packages writes, and blanket process killing. Dependency and interface checks fail explicitly. |
| Inactive `usercustomize.py` | Removed the obsolete startup hook instead of trying to activate an implicit global patch. |
| Shell `eval` interpolates configuration | Transferred environment defaults with `os.execvpe`; Slime runtime environments use JSON serialization, including tests for quotes, dollar signs and newlines. |
| Zero reward counted as correctness | Correctness now follows `EvalResult.passed`, not the sign of the reward. |
| Debug prints and non-English recipe comments | Removed debug output and normalized the changed veRL recipe code to the repository's English convention. |
| No recipe coverage in CI | Added CPU rollout, bootstrap, pinned-dependency, launch regression and backend preflight tests, plus both frontend builds. |
| Co-evolution selected a score-winning candidate and required an initial Harness pool | The scheduler starts with one Harness, explores with the latest configuration, and hands off the newest valid AEGIS proposal unconditionally. It aggregates same-task trajectories across rounds and performs offline GRPO when cross-Harness groups have reward variation. Complete English/Chinese runbooks are in `recipe/coevolution/`. |

The SWE-bench file's expected SHA256 is:

```text
b91794a6779207e6904d103280141657a6a23b529911d20bdc5e3b9bad3ba007
```

No generalized IP/token replacement is run over datasets. Real credentials must
be removed at their configuration source, not by replacing arbitrary substrings
in benchmark content, code, versions, addresses or documentation examples.

## Additional training fixes found during validation

- Per-run evaluator/processor instances retain task-bound runtime state instead
  of losing constructor arguments during builder serialization.
- Registered nested-agent system prompts are actually applied. A regression test
  first reproduced the missing system message, then verified the correction.
- A real parent/tool/child/final-answer smoke test covers both supported tool-call
  formats, inherited provider settings, child probabilities and tool-result masks.
- Nested-agent training uses explicit episode linkage across worker concatenation and orders
  parents before children; validation excludes children.
- Children inherit their own parent's GRPO advantage, not a group average.
  Failed and padding rows do not enter parent reward normalization.
- Padding preserves valid attention inputs while masking rewards and loss.
  Fully masked microbatches have finite zero loss and gradients in all four
  supported aggregation modes.
- Adapter success, child and failure outputs validate against the actual pinned
  veRL Pydantic schema; batch tests use actual DataProto and TensorDict objects.
- Backend preflight checks the interfaces actually imported by the pinned veRL,
  not an unrelated function-call parser from the removed compatibility shim.

## Dependency boundary

veRL is pinned to release 0.7.1, commit
`bec9ef74768dd201881cd4e54cd0385e87caae27`. The bootstrap verifies the archive
checksum and the patch, rejects unsafe archive paths, and detects subsequent
source changes. It supports an offline archive and does not silently replace an
existing incompatible checkout. Generated sources live under ignored `.deps/`.

The supported compatibility baseline follows that release's SGLang extra:
SGLang 0.5.6, PyTorch 2.9.1, transformers 4.51 or later within 4.x. Installation,
preflight and launch instructions are in `recipe/verl/README.md`.

The old Qwen3.5 / SGLang 0.5.11 / transformers 5 combination is **not supported
by this pin**. It needs a separate dependency upgrade and real backend validation;
the removed fake interfaces and source rewrites are not an acceptable workaround.
The veRL recipe is a rollout/GRPO adapter, not an automatic replacement for the
separate GAIA harness-evolution or alternating orchestration entry points.

## Local validation

Validation on October 22, 2026, from the merged repository root:

| Check | Result |
| --- | --- |
| Full unit and integration suites with patched veRL on `PYTHONPATH` | 1070 passed, 5 skipped; 7 warnings from existing/optional dependencies. |
| Alternating scheduler, behavior safety and production handoff tests | 28 passed, including real CPU weight updates, round-boundary resume, accepted-config conditioning and production MetaAgent/provider construction. |
| Actual two-round CPU CLI and independent artifact verifier | Both rounds complete; `completed_rounds=2`, `weights_changed_each_round=true`, `handoff_verified=true`. |
| Real parent/child rollout smoke | Both JSON and Qwen3-Coder formats pass with a deterministic tokenizer and mocked backend. |
| Actual pinned veRL dependency tests | 10 passed, including DataProto batching, zero-mask gradients and Pydantic outputs. |
| Ruff lint / format check on changed Python sources | 54 files pass. |
| veRL bootstrap / installation integrity | Pinned archive verified, patch applied, `--check` passes. |
| Hydra training entry point | The full production scheduler argv resolves through actual pinned Hydra with `--cfg job`, including quoted paths; no GPU training is started. |
| Shell syntax | All four affected launch/data-preparation scripts pass `bash -n`. |
| English / Chinese runbooks | All four Bash blocks in each runbook pass syntax checks; both meta-model YAML examples parse through the actual ModelConfig loader. |
| SWE-bench | Expected SHA256 and 300 entries verified. |
| Frontend build | `npm --prefix frontend run build` passes. |
| Gateway console build | `npm --prefix gateway/console run build` passes. |

The five skips are one optional Feishu SDK test (`lark_oapi` is absent) and four
tests requiring installed Claude plugins. The veRL dependency tests are not
skipped in this run.

The final CPU acceptance artifacts are in the machine-local directory
`/tmp/harnessx-coevo-acceptance-20261022`. Its independent verifier output is
`/tmp/harnessx-coevo-acceptance-20261022-verification.json`; the full-suite log is
`/tmp/harnessx-coevo-full-final.log`. These paths are local evidence, not files
that need to be committed. CI also runs the two-round CPU entry point and its
artifact verifier, in addition to the regression tests.

Core reproduction commands (inside an appropriately provisioned environment):

```bash
python -m recipe.verl.bootstrap
python -m recipe.verl.bootstrap --check
PYTHONPATH="$PWD:$PWD/.deps/verl" python -m pytest tests/unit tests/integration -q
bash -n recipe/verl/launch/run_research_rl.sh
bash -n recipe/slime/launch/data_prep.sh
bash -n recipe/slime/launch/run_math_rl.sh
bash -n recipe/slime/launch/run_math_sft.sh
npm --prefix frontend ci
npm --prefix gateway/console ci
npm --prefix frontend run build
npm --prefix gateway/console run build
```

Install both frontend dependency trees before building: the gateway console
shares frontend source imports. No package-lock changes were needed.

The CPU co-evolution backend does execute multiple rounds of real tiny-policy
optimization through the shared scheduler and real Harness/provider capture.
It verifies config conditioning, changed model tensors, checkpoint feedback,
cross-harness grouping, rejected candidates, failed/no-op training, and
round-boundary resume. It uses deterministic proposals and an in-process model;
it does not prove live MetaAgent APIs, SGLang generation, CUDA/FSDP correctness,
multi-node communication, or partial optimizer-checkpoint recovery. Production
acceptance still needs a dedicated GPU environment, compatible model and real
data. The local system also has a pre-existing broken `harnessx_mtp_patch.pth`
startup hook; it was not edited. Training should use the documented isolated
environment rather than that shared installation.

## Suggested review split

1. Frontend internationalization, without backend behavior changes.
2. Sanitization regressions, data restoration, deployment compatibility and
   example/documentation repairs, with their regression tests.
3. veRL rollout recipe, pinned dependency patch, provenance, tests and CI.
4. Alternating scheduler and bilingual runbooks, reviewed against the reusable
   `mwc` modules rather than presented as an existing `mwc` scheduler.

Any future gateway/lab loopback-default proposal belongs in its own deployment
behavior PR. This workspace has no Git metadata, so this remediation does not
claim to create those commits or push updates to PR #18.
