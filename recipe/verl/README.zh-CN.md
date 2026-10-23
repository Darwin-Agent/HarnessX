# HarnessX × veRL

[English](README.md) | **简体中文**

通过 HarnessX 生成研究代理轨迹，使用 veRL 分布式 GRPO 训练模型。
配方包含 SGLang 生成词元与对数概率采集和工具结果掩码。
交替执行 Harness 搜索与模型训练，请使用[协同演进配方](../coevolution/README.zh-CN.md)。

以下命令均在仓库根目录执行。

## 安装

使用 Python 3.12，以及适用于 PyTorch 2.9.1 的 CUDA 环境。依赖基线：

| 组件 | 版本 |
| --- | --- |
| veRL | 0.7.1，提交 `bec9ef74768dd201881cd4e54cd0385e87caae27` |
| PyTorch | 2.9.1 |
| SGLang | 0.5.6 |
| transformers | >=4.51,<5 |

```bash
python3.12 -m venv .venv-verl
source .venv-verl/bin/activate
python -m pip install -e .
python -m recipe.verl.bootstrap
python -m pip install -c recipe/verl/constraints.txt -e './.deps/verl[sglang]'
export PYTHONPATH="$PWD:$PWD/.deps/verl"
```

根据模型与硬件安装对应的注意力计算和 CUDA 扩展。选择 transformers 4.x 支持的模型，
并匹配工具调用模板。研究任务配置使用 Qwen3-Coder 工具调用格式和 user 角色的工具结果。

离线安装时可指定上游源码归档：

```bash
python -m recipe.verl.bootstrap --archive /path/to/verl-archive.tar.gz
```

## 启动模型训练

准备本地模型目录，以及以下 Parquet 文件：

- `$DATA_DIR/train.parquet`
- `$DATA_DIR/val.parquet`

数据行格式见[研究代理数据说明](../coevolution/README.zh-CN.md#2-准备数据)。

```bash
export DATA_DIR=/path/to/training-data
export MODEL_DIR=/path/to/local-compatible-model
export CKPT_DIR=/path/to/checkpoints
bash recipe/verl/launch/run_research_rl.sh
```

默认配置面向八卡。根据模型、序列长度、批大小和 GPU 资源调整 `launch/config.yaml`，
或传入 Hydra 参数覆盖。启动前准备好模型权重和数据集，输出写入 `CKPT_DIR`。

## 运行配置

- **搜索：** 可选 `SERPAPI_API_KEY` 和 `TAVILY_API_KEY` 分别启用对应服务，
  公开搜索回退需要外部网络访问。
- **Hugging Face：** `HF_ENDPOINT` 默认使用 `https://huggingface.co`，
  使用镜像时设置为对应地址。
- **Ray：** 启动器管理本地运行时。`STOP_EXISTING_SERVICES=1` 会在启动前停止已有
  本地 Ray 集群，仅在专用训练节点使用。
- **可选嵌套代理模式：** MATPO 可用于使用明确父子代理关系的工作流。

## 依赖分发

配方将固定版本的上游源码展开至 `.deps/verl`，并应用仓库内的集成补丁：

| 文件 | 用途 |
| --- | --- |
| `dependencies/verl.lock.json` | 上游提交、归档地址与 SHA-256 |
| `dependencies/harnessx.patch` | HarnessX 轨迹、GRPO 与可选嵌套代理集成 |
| `dependencies/LICENSE` | 上游 Apache-2.0 许可证 |
| `dependencies/Notice.txt` | 上游署名信息 |
| `constraints.txt` | 训练依赖约束 |

安装过程检查归档完整性并记录源码与补丁哈希。保持生成的依赖目录不变；
更新版本或补丁后，将旧目录移开，再重新展开依赖。

### 集成补丁

| 上游模块 | 集成内容 |
| --- | --- |
| `experimental/agent_loop/agent_loop.py` | 子代理输出、父子关联、父代理优先排序与父代理评估 |
| `trainer/ppo/ray_trainer.py` | 父子代理优势处理、共享题目 GRPO 标识的跨 Harness 分配与批次填充 |
| `trainer/ppo/core_algos.py` | 空微批次的掩码损失处理 |
| `trainer/ppo/rollout_corr_helper.py` | 支持掩码的生成校正指标 |
| `utils/fsdp_utils.py` | 参数元数据处理 |
| `utils/torch_functional.py` | 序列并行训练使用原生 PyTorch 交叉熵 |
| `workers/actor/dp_actor.py` | 按输入类型选择多模态序列并行路径 |
