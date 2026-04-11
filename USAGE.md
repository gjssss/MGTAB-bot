# 使用指南

本文档说明四个核心脚本的用法：`download.py`、`build_dataset.py`、`train_adaboost.py`、`test.py`，以及推理入口 `predict.py` / `api.py`。

## 前置条件

- Python 环境由 `uv` 管理：`uv sync` 安装依赖。
- 训练 / 推理默认使用 GPU（`device=auto`）。本机无 CUDA 时会自动退回 CPU，但仅推荐用于小数据烟雾测试。
- 运行前必须先下载 Qwen embedding 模型到 `models/` 目录。

---

## 1. download.py — 下载 Qwen embedding 模型

从 HuggingFace 下载 Qwen3-Embedding 的某一档位到本地 `models/Qwen3-Embedding-<size>/`。

### 支持档位

| 档位 | embedding 维度 | 本地目录 | 大致体积 |
|---|---|---|---|
| `0.6B` | 1024 | `models/Qwen3-Embedding-0.6B` | ~1.2 GB |
| `4B`   | 2560 | `models/Qwen3-Embedding-4B`   | ~8 GB |
| `8B`   | 4096 | `models/Qwen3-Embedding-8B`   | ~16 GB |

### 用法

```bash
# 默认：读取环境变量 HTTPS_PROXY / HTTP_PROXY / ALL_PROXY
uv run python download.py --qwen_size 0.6B

# 指定自定义输出目录
uv run python download.py --qwen_size 8B --output /data/models/qwen-8b

# 强制清理代理变量（某些内网环境需要）
uv run python download.py --qwen_size 0.6B --no-proxy

# 指定 HF revision
uv run python download.py --qwen_size 0.6B --revision main
```

### 代理行为
- 默认会自动识别并打印所有 `HTTPS_PROXY` / `HTTP_PROXY` / `ALL_PROXY`（含大小写变体），`huggingface_hub` 底层 `requests` 会使用它们。
- `--no-proxy` 会在下载前清空这些环境变量。

---

## 2. build_dataset.py — 从 JSONL 构建训练数据集

读取标注好的 JSONL（每行一个 user dict，含 `label: bool` 和 `tweets: list[str]`），用 Qwen embedding 生成 `features.pt` + `labels_bot.pt`。

### 参数

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `--jsonl` | 是 | - | 输入 JSONL 路径 |
| `--name`  | 是 | - | 输出数据集名字，落到 `Dataset/<name>/` |
| `--qwen_size` | 是 | - | `0.6B` / `4B` / `8B`，必须与稍后 `train_adaboost.py` 一致 |
| `--batch_size` | 否 | `16` | 批大小 |
| `--max_tweets` | 否 | `200` | 每个用户最多使用多少条 tweet |
| `--limit` | 否 | `None` | 只用前 N 条记录（CPU 烟雾测试） |
| `--device` | 否 | `auto` | `auto` / `cuda` / `cpu` |
| `--dry_run` | 否 | `False` | 只跑前 10 条，**不写入**磁盘 |

### 产物
- `Dataset/<name>/features.pt` — `(N, 20 + embedding_dim)` float32 张量
- `Dataset/<name>/labels_bot.pt` — `(N,)` int64 张量
- `Dataset/<name>/dataset_info.json` — 构建元数据（qwen_size、feature_dim、label 分布、build_timestamp、sha256 等，供 `train_adaboost.py` 读取）

### 示例

**本机 CPU 烟雾测试（20 条 × 10 条 tweet，几十秒）**：
```bash
uv run python build_dataset.py \
    --jsonl raw_data/twibot20_accounts_cleaned.jsonl \
    --name twibot20_smoke \
    --qwen_size 0.6B \
    --device cpu \
    --limit 20 --max_tweets 10 --batch_size 4
```

**生产 GPU 全量构建**：
```bash
uv run python build_dataset.py \
    --jsonl raw_data/twibot20_accounts_cleaned.jsonl \
    --name twibot20 \
    --qwen_size 8B \
    --device cuda --batch_size 32
```

### 输入 JSONL 格式

每行一个 JSON 对象，必须包含 `label`（bool）字段，其他字段作为 user profile；`tweets` 字段为字符串列表。raw_data 已清洗，可直接使用：

```json
{"user_id": "...", "verified": true, "followers_count": 1234, ..., "tweets": ["...", "..."], "label": false}
```

---

## 3. train_adaboost.py — 训练 AdaBoost 分类器

在 `build_dataset.py` 产出的数据集上训练 AdaBoost，保存模型和推理配置。

**注意**：AdaBoost 本身是 sklearn 的 CPU 算法，不使用 GPU。嵌入已在 `build_dataset` 阶段完成。

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dataset_name` | - | 数据集名字，映射到 `Dataset/<name>`；与 `--dataset_dir` 二选一 |
| `--dataset_dir` | - | 数据集目录直接路径；与 `--dataset_name` 二选一 |
| `--n_estimators` | 50 | AdaBoost 弱分类器数量 |
| `--learning_rate` | 1.0 | 学习率 |
| `--random_seed` | `0 1 2 3 4` | 多 seed 取平均（空格分隔）；也可传单个 seed |
| `--limit` | None | 只用前 N 条样本（烟雾测试） |
| `--output_dir` | `checkpoints` | 输出目录 |

### 产物

- `checkpoints/adaboost_bot.joblib` — 最佳 seed 的分类器
- `checkpoints/preprocess_config.json` — **推理侧读**，包含 `qwen_size`、`embedding_dim`、`feature_dim`、归一化配置等

### 示例

**小数据烟雾（承接上一步 smoke 数据集）**：
```bash
uv run python train_adaboost.py --dataset_name twibot20_smoke --random_seed 0
```

**生产全量训练**：
```bash
uv run python train_adaboost.py --dataset_name twibot20
```

### 一致性约束（重要）
- `train_adaboost.py` 会读取 `Dataset/<name>/dataset_info.json` 里的 `qwen_size` / `feature_dim` 并写入 `checkpoints/preprocess_config.json`。
- **推理脚本（`predict.py` / `test.py` / `api.py`）都会从 `preprocess_config.json` 读 `qwen_size`，必须和训练时完全一致**。不要手动改 `preprocess_config.json`。

---

## 4. test.py — 在标注 JSONL 上评测模型

用训练好的 classifier + embedding 批量预测一个 JSONL，输出 accuracy / precision / recall / F1 / 混淆矩阵。

### 参数

| 参数 | 必填 | 说明 |
|---|---|---|
| `input` | 是 | 输入 JSONL 路径（位置参数） |
| `-o/--output` | 是 | 输出 JSON 报告路径 |

### 输入格式

每行一个 JSON 对象，必须包含 `label: bool` 字段。`user` 字段可选：
- 有 `user`：脚本取 `record["user"]` 作为 user dict。
- 无 `user`：脚本把**整条记录**（排除 `label`）当成 user dict。两种格式都支持，`raw_data/twibot20_accounts_cleaned.jsonl` 就是扁平格式。

### 产物

`--output` 指定的 JSON 文件，结构：
```json
{
  "success": true,
  "data": {
    "metrics": {
      "total": ..., "correct": ..., "accuracy": ...,
      "bot_precision": ..., "bot_recall": ..., "bot_f1": ...,
      "confusion_matrix": {"tp": ..., "tn": ..., "fp": ..., "fn": ...}
    },
    "results": [
      {"line": 1, "expected_label": "bot", "predicted_label": "bot",
       "confidence": 0.91, "correct": true},
      ...
    ],
    "errors": null
  }
}
```

### 示例

**用 raw 数据前 5 条快速测脚本（本机 CPU）**：
```bash
head -n 5 raw_data/twibot20_accounts_cleaned.jsonl > /tmp/subset.jsonl
uv run python test.py /tmp/subset.jsonl -o /tmp/subset_out.json
# 终端会打印：accuracy=... correct=X/Y failed=Z
```

**全量评测**：
```bash
uv run python test.py raw_data/twibot20_accounts_cleaned.jsonl -o reports/eval_full.json
```

### 性能提示
- `test.py` 在 `main()` 入口一次性加载 classifier + Qwen embedding model，循环内部**只做前向推理**，不会每条重新加载模型。
- 推理走的是 `encode_texts_single`（每条 tweet 单独 forward 再 mean），与 `api.py` / `predict.py` 热路径完全一致。

---

## 5. predict.py — 单条 CLI 推理

```bash
uv run python predict.py test_samples/bot_example.json
```

输入：一个 JSON 文件（扁平 user dict，包含 `tweets` 列表）。
输出：`{label, prediction, confidence, probabilities: {human, bot}}`。

---

## 6. api.py — HTTP 服务

```bash
uv run python api.py              # 默认监听 0.0.0.0:30102
```

### 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET  | `/health` | 返回服务状态 + 当前 `qwen_size` / `embedding_dim` |
| POST | `/bot` | 单条预测，请求体 `{"user": {...}}` |
| POST | `/bot/batch` | 批量预测，请求体 `{"items": [{...}, {...}]}` |

### 示例

```bash
curl http://localhost:30102/health

curl -X POST http://localhost:30102/bot \
    -H 'Content-Type: application/json' \
    --data-binary @test_samples/api_payload_bot.json
```

### Break change 说明
- 请求体 **不再接受** `lang` 字段（原来的中英文分支被 Qwen3 统一替换）。
- 带任何未声明字段（含 `lang`）的请求会返回 **HTTP 422**，这是 `extra="forbid"` 的预期行为。

---

## 典型工作流汇总

### 本机 CPU 烟雾链（0.6B，验证 pipeline，不代表模型质量）

```bash
uv run python download.py --qwen_size 0.6B

uv run python build_dataset.py \
    --jsonl raw_data/twibot20_accounts_cleaned.jsonl \
    --name twibot20_smoke --qwen_size 0.6B \
    --device cpu --limit 20 --max_tweets 10 --batch_size 4

uv run python train_adaboost.py --dataset_name twibot20_smoke --random_seed 0

head -n 5 raw_data/twibot20_accounts_cleaned.jsonl > /tmp/subset.jsonl
uv run python test.py /tmp/subset.jsonl -o /tmp/subset_out.json
```

### 生产 GPU 全量训练链

```bash
uv run python download.py --qwen_size 8B

uv run python build_dataset.py \
    --jsonl raw_data/twibot20_accounts_cleaned.jsonl \
    --name twibot20 --qwen_size 8B \
    --device cuda --batch_size 32

uv run python train_adaboost.py --dataset_name twibot20

uv run python test.py raw_data/twibot20_accounts_cleaned.jsonl -o reports/eval_full.json

uv run python api.py &
curl http://localhost:30102/health
```

---

## 常见问题

**Q: 训练时报 `features.pt dim X != dataset_info feature_dim Y`**
A: `Dataset/<name>/` 被不同 `--qwen_size` 重复生成过。删掉目录重新 `build_dataset.py`。

**Q: 推理时报 `Feature dim mismatch: classifier expects X, got Y`**
A: `checkpoints/adaboost_bot.joblib` 是用 A 档位训练的，但 `preprocess_config.json` 的 `qwen_size` 被改成了 B 档位。重新跑 `train_adaboost.py` 让两者一致。

**Q: 推理时报 `Qwen model not found at models/Qwen3-Embedding-...`**
A: 需要先跑 `download.py` 下载对应档位的模型。

**Q: CPU 模式跑全量太慢**
A: 预期行为。CPU 只用于 `--limit` 小数据烟雾测试，生产一定走 GPU。
