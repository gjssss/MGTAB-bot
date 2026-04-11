# 基于 Qwen3-Embedding 重建数据集 + utils 包统一共享逻辑

## Context（为什么做）

当前仓库的推理 (`api.py` / `predict.py` / `test.py`) 用 `roberta-base` / `chinese-roberta-wwm-ext` 现场编码推文，而训练用的 `Dataset/MGTAB/features.pt` 是来源不明的 788 维预嵌入，**训练/推理嵌入空间不一致**会静默拖低 F1。同时 `api.py` 和 `predict.py` 的嵌入加载/编码代码是复制粘贴的两份，`test.py` 通过 `predict.py` 绕一层，结构分散。

本次改造：
1. 用 **Qwen3-Embedding** 作为唯一嵌入模型（支持 0.6B / 4B / 8B 三档，脚本参数可选），取消中英文 `lang` 区分。
2. 从 `raw_data/twibot20_accounts_cleaned.jsonl` 重建 `features.pt` + `labels_bot.pt`，再用同一嵌入模型训练 AdaBoost，消除训练/推理漂移。
3. 把训练 / 构建 / 推理 / 测试间的重复逻辑抽到新 `utils/` 包。
4. 删除与 AdaBoost+API 无关的所有 GNN / ML 基线脚本（用户授权）。
5. 新增 `download.py` 用于从 HuggingFace 下载 Qwen embedding 模型，默认读取环境变量代理。
6. Break change：从 API / predict / test 彻底移除 `lang` 字段，不做向后兼容。
7. 支持 CPU 小数据烟雾测试（`--device cpu --limit N`），默认 GPU（有 CUDA 就走 GPU，否则 CPU 兜底）。

## 用户明确授权的关键决策

| 项 | 决定 |
|---|---|
| 嵌入模型 | Qwen3-Embedding，CLI 选 0.6B / 4B / 8B；本机测试默认 0.6B |
| GPU 训练含义 | 仅嵌入阶段走 GPU；AdaBoost 拟合保持 sklearn CPU |
| Pooling 数学 | `last-token pool → L2 normalize → mean → L2 normalize`；放弃和旧 MGTAB 788 维基线直接对比 |
| 无关脚本 | 直接物理删除（见文件清单） |
| `lang` 字段 | break change，不做 extra='ignore' 兼容 |
| `test_samples/` | 直接更新或重新生成，不保留旧 payload |
| 依赖 | 允许收紧 `transformers` 版本，并新增 `accelerate` / `huggingface_hub` |

## Qwen3-Embedding 三档规格

| 模型 | HF repo | `hidden_size` | 本地目录 |
|---|---|---|---|
| 0.6B | `Qwen/Qwen3-Embedding-0.6B` | 1024 | `models/Qwen3-Embedding-0.6B` |
| 4B   | `Qwen/Qwen3-Embedding-4B`   | 2560 | `models/Qwen3-Embedding-4B` |
| 8B   | `Qwen/Qwen3-Embedding-8B`   | 4096 | `models/Qwen3-Embedding-8B` |

由于不同档位维度不同，**训练与推理必须使用同一档**。策略：

- `build_dataset.py` / `train_adaboost.py` 用 `--qwen_size` 选档，`train_adaboost.py` 把档位名写入 `checkpoints/preprocess_config.json`。
- `api.py` / `predict.py` / `test.py` 启动时读 `preprocess_config.json` 得到档位名，自动加载对应模型，**不接受运行时参数覆盖**（避免尺寸不匹配）。
- `preprocess.py` 不再硬编码 `EMBEDDING_DIM` / `FEATURE_DIM` 作为断言值；改为计算型：`build_feature_vector(user, tweet_embedding)` 直接用 `tweet_embedding.shape[-1]`，`assert` 改为 `assert tweet_embedding.ndim == 1`。保留 `PROPERTY_DIM = 20` 常量。

## `utils/` 包设计

```
utils/
  __init__.py       # 空（旧 sample_mask 随 GNN 脚本一起删除）
  device.py         # resolve_device
  embedding.py      # Qwen 加载、single / batched 编码
  inference.py      # load_classifier / predict_user
  jsonl.py          # iter_jsonl_records
  config.py         # load_preprocess_config / save_preprocess_config
```

### `utils/device.py`
```python
def resolve_device(preferred: str | None = "auto") -> torch.device:
    # auto / None -> cuda if available else cpu
    # cuda -> 必须可用，否则 RuntimeError
    # cpu  -> cpu
```

### `utils/config.py`
```python
CONFIG_PATH = Path("checkpoints/preprocess_config.json")

def load_preprocess_config() -> dict
def save_preprocess_config(cfg: dict) -> None
# 关键字段: "qwen_size", "embedding_dim", "feature_dim", "property_dim", "normalization", "bool_fields"
```

### `utils/embedding.py`
```python
QWEN_SIZES = {
    "0.6B": {"repo": "Qwen/Qwen3-Embedding-0.6B", "dir": "Qwen3-Embedding-0.6B", "dim": 1024},
    "4B":   {"repo": "Qwen/Qwen3-Embedding-4B",   "dir": "Qwen3-Embedding-4B",   "dim": 2560},
    "8B":   {"repo": "Qwen/Qwen3-Embedding-8B",   "dir": "Qwen3-Embedding-8B",   "dim": 4096},
}
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MAX_LENGTH = 512

def load_embedding(qwen_size: str, device: torch.device) -> tuple[tokenizer, model, int]
    """本地优先：models/<dir>；本地不存在则报错并提示运行 download.py。
    tokenizer padding_side='left'；bf16 on cuda、fp32 on cpu；model.eval()。
    进程级缓存 (qwen_size -> (tok, model, dim))，避免重复加载。
    返回 dim 供调用方记录/比对。"""

def _last_token_pool(last_hidden, attention_mask) -> Tensor
    """Qwen 官方 pooling。支持 left padding 快路径。"""

def encode_texts_single(texts: list[str], tokenizer, model, device) -> np.ndarray
    """单用户热路径。逻辑：
      - 过滤空字符串；若最终为空 → np.zeros(dim)
      - 逐条 tokenize(truncation, max_length=MAX_LENGTH, return_tensors='pt')
      - torch.no_grad() 前向，last_token_pool
      - L2 normalize
      - 对若干条做 mean
      - 再 L2 normalize
      - .cpu().numpy().astype(np.float32)"""

def encode_texts_batched(
    users_texts: list[list[str]],
    tokenizer, model, device,
    batch_size: int = 16,
    max_tweets_per_user: int | None = 200,
    show_progress: bool = True,
) -> np.ndarray
    """build_dataset 专用。实现：flatten + offsets → 按 batch_size 批处理
      (padding=True, truncation, max_length=MAX_LENGTH) → last_token_pool → L2 normalize
      → 按 user offsets 做 mean → L2 normalize → shape (N, dim) float32。
      空用户直接填 zeros(dim)。tqdm 进度条。
      数学与 single 路径严格一致：normalize → mean → normalize。"""
```

### `utils/inference.py`
```python
def load_classifier(path: str | Path = "checkpoints/adaboost_bot.joblib")
    # 进程级缓存

def predict_user(user: dict, clf, tokenizer, emb_model, device) -> dict
    # 返回 {label, prediction, confidence, probabilities: {human, bot}}
    # build_feature_vector(user, encode_texts_single(...)) 构造
    # feature_vec.shape[1] 必须与 clf 期望维度一致 (由训练时 dataset 决定)
```

### `utils/jsonl.py`
```python
def iter_jsonl_records(path: str | Path) -> Iterator[tuple[int, dict]]
    # yield (line_no, record)，跳过空行，捕获 JSONDecodeError 抛带行号异常
```

## 实现步骤

### Step 1: 删除无关脚本和旧资产

**删除的 Python 脚本**（与 AdaBoost+API 无关，用户授权清除）：
- `Cresci15-GNN.py`
- `Cresci15-ML.py`
- `Dataset.py`（GNN Dataset 类）
- `GNN_sample_large.py`
- `HGT.py`
- `MGTAB-GNN.py`
- `MGTAB-ML.py`
- `RGT.py`
- `SHGN.py`
- `layer.py`（GNN layer）
- `models.py`（GNN models）
- `samplemodel.py`（GNN models）
- `utils.py`（被 `utils/` 包替换）

**删除的目录**：
- `Standardization/`（仅包含无用空占位目录，非 AdaBoost 需要）
- `models/roberta-base/`（物理删除，gitignored）
- `models/chinese-roberta-wwm-ext/`（物理删除，gitignored）

### Step 2: 新建 `utils/` 包

创建上述 6 个文件。`utils/__init__.py` 保持空或仅做子模块占位。

### Step 3: 更新 `preprocess.py`

- 删除 `EMBEDDING_DIM` / `FEATURE_DIM` 硬编码常量（或保留为向后兼容但不再被 `build_feature_vector` 使用）。
- `build_feature_vector(user_json, tweet_embedding)` 改为：
  ```python
  assert tweet_embedding.ndim == 1
  return np.concatenate([extract_properties(user_json), tweet_embedding]).astype(np.float32)
  ```
- 更新注释头部 col 布局说明（embedding 部分标注 "由 Qwen3-Embedding 档位决定"）。
- 不动 `NORMALIZATION_CONFIG` / `BOOL_FIELDS` / `extract_properties`。

### Step 4: 新增 `download.py`

CLI:
```
python download.py --qwen_size 0.6B [--output models/] [--no-proxy]
```

实现要点：
- 默认读取环境变量 `HTTPS_PROXY`/`HTTP_PROXY`/`ALL_PROXY`（大小写均查），显式打印检测到的代理，供 huggingface_hub 底层 requests 自动使用。
- 使用 `huggingface_hub.snapshot_download(repo_id=..., local_dir=models/<dir>, local_dir_use_symlinks=False)`。
- 校验下载后目录包含 `config.json`、`tokenizer.json`、`model.safetensors`（或分片）。
- `--no-proxy` 强制清理环境中的代理变量（对某些内网镜像场景有用）。
- 失败时给出明确报错（网络 / 权限 / 磁盘），不要吞错误。

### Step 5: 重构 `api.py`

删除：
- `EMBEDDING_MODELS` / `MODELS_DIR` / `tokenizers` / `embed_models` globals
- `load_embedding` / `encode_tweets` 函数
- `SingleRequest.lang` / `BatchRequest.lang` 字段
- `UserProfile` 中与 lang 相关逻辑（无）

新增：
```python
from utils.device import resolve_device
from utils.embedding import load_embedding
from utils.inference import load_classifier, predict_user
from utils.config import load_preprocess_config

device = resolve_device("auto")
cfg = load_preprocess_config()
qwen_size = cfg["qwen_size"]

async def lifespan(_app):
    load_classifier()
    load_embedding(qwen_size, device)  # 预热
    yield
```

`predict_single(user)` 使用 `predict_user(user, clf, tokenizer, model, device)`。

`/health` 的 `embedding_models` 字段改为 `{"qwen_size": qwen_size, "embedding_dim": cfg["embedding_dim"]}`。

`SingleRequest` / `BatchRequest` 显式禁止 `lang`（Pydantic 默认 extra='ignore'，但我们通过删除字段让旧客户端传 lang 时不会被处理；如果需要严格报错可用 `model_config = ConfigDict(extra="forbid")` —— 用户要求 break change，**采用 forbid**）。

### Step 6: 重构 `predict.py`

完全瘦身为 CLI wrapper：
```python
from utils.device import resolve_device
from utils.embedding import load_embedding
from utils.inference import load_classifier, predict_user
from utils.config import load_preprocess_config

def predict_bot(user_json: dict, model_path="checkpoints/adaboost_bot.joblib") -> dict:
    device = resolve_device("auto")
    cfg = load_preprocess_config()
    clf = load_classifier(model_path)
    tok, model, _ = load_embedding(cfg["qwen_size"], device)
    return predict_user(user_json, clf, tok, model, device)
```

CLI：`python predict.py <input.json>`（移除 `[lang]` 参数）。

### Step 7: 重构 `test.py`

- CLI 移除 `--lang`。
- `normalize_record` 移除 lang 处理。
- `main()` 开头一次性加载 device / classifier / embedding（避免每条 predict 重新进缓存函数）；循环内调 `predict_user`。
- 用 `utils.jsonl.iter_jsonl_records` 替换手写循环。
- 输出 JSON 字段保持（`metrics`/`results`/`errors`），删除 per-result 的 `lang` 字段。

### Step 8: 新增 `build_dataset.py`

CLI:
```
python build_dataset.py \
    --jsonl raw_data/twibot20_accounts_cleaned.jsonl \
    --name twibot20 \
    --qwen_size 0.6B \
    [--batch_size 16] \
    [--max_tweets 200] \
    [--limit N]          # 前 N 条，用于 CPU 烟雾测试
    [--device auto]      # auto / cpu / cuda
    [--dry_run]          # 只处理前 10 条，不写入
```

流程：
1. `device = resolve_device(args.device)`；CPU 且未设 `--limit` 时打印明显警告。
2. `tok, model, dim = load_embedding(args.qwen_size, device)`；本地目录不存在 → 报错并提示运行 `python download.py --qwen_size {size}`。
3. `records = [rec for _, rec in iter_jsonl_records(args.jsonl)]`；若 `--limit` 截断。
4. `props = np.stack([extract_properties(r) for r in records])  # (N, 20)`
5. `users_tweets = [[t for t in r.get("tweets", []) if isinstance(t, str) and t.strip()] for r in records]`
6. `tweet_mat = encode_texts_batched(users_tweets, tok, model, device, batch_size=args.batch_size, max_tweets_per_user=args.max_tweets)  # (N, dim)`
7. `features = np.concatenate([props, tweet_mat], axis=1).astype(np.float32)  # (N, 20+dim)`
8. `labels = np.array([int(bool(r["label"])) for r in records], dtype=np.int64)`
9. 保存到 `Dataset/<name>/features.pt` + `labels_bot.pt`
10. 写 `Dataset/<name>/dataset_info.json`：source、sha256、n_samples、property_dim=20、embedding_dim=dim、feature_dim=20+dim、qwen_size、max_tweets_per_user、batch_size、device、label 分布、build_timestamp。
11. Sanity check：shape 断言、label 分布打印。

### Step 9: 更新 `train_adaboost.py`

新增 CLI：
- `--dataset_name`（优先于 `--dataset_dir`，映射到 `Dataset/<name>`）
- `--limit N`（从 features/labels 取前 N 条）
- 其余保持

`main()` 改动：
- 根据 `dataset_name` / `dataset_dir` 解析路径
- `features.shape[1]` 动态读取，不再 `== 788`
- 读取对应的 `Dataset/<name>/dataset_info.json` 获取 `qwen_size` / `embedding_dim`（若无则报错提示先跑 build_dataset）
- 写 `checkpoints/preprocess_config.json`：
  ```json
  {
    "qwen_size": "0.6B",
    "embedding_dim": 1024,
    "property_dim": 20,
    "feature_dim": 1044,
    "normalization": { ...NORMALIZATION_CONFIG },
    "bool_fields": { ...BOOL_FIELDS },
    "n_estimators": 50,
    "learning_rate": 1.0,
    "best_seed": 2,
    "trained_samples": 10000
  }
  ```

### Step 10: 更新 `Dockerfile`

- `COPY` 列表删除所有已删除的脚本。
- 新 `COPY`: `api.py predict.py preprocess.py train_adaboost.py build_dataset.py download.py test.py`
- 新 `COPY utils/ ./utils/`
- `models/` 依然整个 COPY（假设镜像构建时已将 `Qwen3-Embedding-{size}` 放到 `models/` 下），由使用者自行选档。

### Step 11: 更新 `pyproject.toml`

- `transformers>=4.51,<5`（Qwen3-Embedding 需要 >=4.51）
- 新增 `accelerate>=0.26`
- 新增 `huggingface_hub>=0.23`（显式依赖，虽然 transformers 会拉）
- 新增 `tqdm`（build_dataset 进度条）
- 其余保持

### Step 12: 更新 / 重建 `test_samples/`

- 删除所有含 `lang` 字段的旧样例。
- 重新生成 2 份：一个 bot + 一个 human（从 `raw_data/twibot20_accounts_cleaned.jsonl` 各取一条），去掉 `lang` 字段，保留 `user` 或直接放顶层。

## 文件清单（修改/新增/删除）

| 操作 | 路径 | 说明 |
|---|---|---|
| 新增 | `utils/__init__.py` | 空或占位 |
| 新增 | `utils/device.py` | `resolve_device` |
| 新增 | `utils/embedding.py` | Qwen 加载 + single/batched 编码 |
| 新增 | `utils/inference.py` | `load_classifier` + `predict_user` |
| 新增 | `utils/jsonl.py` | `iter_jsonl_records` |
| 新增 | `utils/config.py` | `load_preprocess_config` / `save_preprocess_config` |
| 新增 | `download.py` | HF 下载 + 代理环境变量 |
| 新增 | `build_dataset.py` | 重建数据集 CLI |
| 修改 | `preprocess.py` | 移除硬编码维度，动态化 |
| 修改 | `api.py` | 移除 lang + 重复逻辑，import utils |
| 修改 | `predict.py` | 移除 lang + 重复逻辑，import utils |
| 修改 | `test.py` | 移除 lang，复用 utils.inference，模型只加载一次 |
| 修改 | `train_adaboost.py` | `--dataset_name` / `--limit`，dataset_info 读取，config 写入 |
| 修改 | `Dockerfile` | COPY 清单更新，新增 utils/ |
| 修改 | `pyproject.toml` | 收紧 transformers、新增 accelerate/huggingface_hub/tqdm |
| 删除 | `utils.py` | 被 `utils/` 包替换 |
| 删除 | `Cresci15-GNN.py` / `Cresci15-ML.py` | 无关 GNN/ML 基线 |
| 删除 | `Dataset.py` | 仅 GNN 用 |
| 删除 | `GNN_sample_large.py` / `HGT.py` / `MGTAB-GNN.py` / `MGTAB-ML.py` / `RGT.py` / `SHGN.py` | 无关 |
| 删除 | `layer.py` / `models.py` / `samplemodel.py` | GNN 模块 |
| 删除 | `Standardization/` | 空占位 |
| 删除 | `models/roberta-base/` / `models/chinese-roberta-wwm-ext/` | 旧模型 |
| 删除 / 重建 | `test_samples/*` | 去 lang |
| 输出 | `Dataset/twibot20/features.pt` | (10000, 20+dim) float32 |
| 输出 | `Dataset/twibot20/labels_bot.pt` | (10000,) int64 |
| 输出 | `Dataset/twibot20/dataset_info.json` | 可复现性元数据 |
| 输出 | `checkpoints/adaboost_bot.joblib` | 新分类器 |
| 输出 | `checkpoints/preprocess_config.json` | 含 qwen_size |

## 端到端验证

### 0. 本机 CPU 烟雾（0.6B）
```bash
python download.py --qwen_size 0.6B
python -c "from utils.embedding import load_embedding; import torch; load_embedding('0.6B', torch.device('cpu'))"
python build_dataset.py --jsonl raw_data/twibot20_accounts_cleaned.jsonl \
    --name twibot20_smoke --qwen_size 0.6B \
    --device cpu --limit 20 --max_tweets 10 --batch_size 4
python train_adaboost.py --dataset_name twibot20_smoke --random_seed 0
python predict.py test_samples/bot.json       # 应该返回 bot
python predict.py test_samples/human.json     # 应该返回 human
python test.py test_samples/mixed.jsonl -o /tmp/test_out.json
```

预期：build_dataset 输出 `Dataset/twibot20_smoke/features.pt` shape=(20, 1044) float32；train_adaboost 正常输出 F1；predict.py / test.py 不报 lang 相关错误。

### 1. GPU 完整训练（视实际 GPU 选档）
```bash
python download.py --qwen_size 0.6B   # 或 4B / 8B
python build_dataset.py --jsonl raw_data/twibot20_accounts_cleaned.jsonl \
    --name twibot20 --qwen_size 0.6B
python train_adaboost.py --dataset_name twibot20
```

预期：`Dataset/twibot20/features.pt` shape=(10000, 1044)；train F1 与记录一致。

### 2. API 端到端
```bash
python api.py &
sleep 5
curl -X POST http://localhost:30102/bot \
    -H 'Content-Type: application/json' \
    --data-binary @test_samples/bot.json
curl http://localhost:30102/health
```

预期：`/health` 返回 `{"embedding_models": {"qwen_size": "0.6B", "embedding_dim": 1024}}`；`/bot` 对 bot 样本返回 `{"label": "bot"}`。

### 3. break change 验证
```bash
# 发送带 lang 字段的旧 payload，预期 422 (extra='forbid')
curl -X POST http://localhost:30102/bot \
    -H 'Content-Type: application/json' \
    -d '{"lang": "en", "user": {"screen_name": "x"}}'
```
预期：422 Unprocessable Entity。

### 4. 导入自检
```bash
python -c "
import preprocess
import utils.device, utils.embedding, utils.inference, utils.jsonl, utils.config
import api, predict, test, train_adaboost, build_dataset, download
print('all imports ok')
"
```

### 5. 单条 vs 批量一致性
```bash
python -c "
import torch, numpy as np
from utils.embedding import load_embedding, encode_texts_single, encode_texts_batched
dev = torch.device('cpu')
tok, m, _ = load_embedding('0.6B', dev)
texts = ['hello world', 'bot detection test']
v1 = encode_texts_single(texts, tok, m, dev)
v2 = encode_texts_batched([texts], tok, m, dev, batch_size=4, show_progress=False)[0]
assert np.allclose(v1, v2, atol=1e-4), (v1[:5], v2[:5])
print('single == batched OK')
"
```

## 风险与缓解

| 风险 | 缓解 |
|---|---|
| Qwen3-Embedding-8B 对本机 CPU/内存压力过大 | 0.6B 本机测试默认；CLI 报错提示 |
| 首次下载需要科学上网 | `download.py` 默认读环境代理；找不到模型时给明确报错信息 |
| 单条 vs 批量数值漂移 | 数学严格 `normalize→mean→normalize`；验证步骤 5 用 `np.allclose(atol=1e-4)` 回归 |
| 删除 `utils.py` 破坏 `from utils import sample_mask` | 所有 import 点都在被删的 GNN 脚本中，一并清理 |
| `preprocess_config.json` 与 `features.pt` 不匹配（用户跑错组合） | train_adaboost 强制把 dataset_info.json 的 qwen_size 写入 config；api/predict 启动时断言 classifier feature 数量 = PROPERTY_DIM + QWEN_SIZES[qwen_size]["dim"]，不匹配直接 raise |
| Docker 镜像体积 | 由用户自行选择入镜像的 qwen 档位；Dockerfile 仅 COPY `models/` 目录，镜像大小由档位决定 |
| break change 冲击线上客户端 | 用户明确授权 |

## 复用函数位置参考

- `preprocess.extract_properties` — preprocess.py:63
- `preprocess.build_feature_vector` — preprocess.py:140（本次修改）
- `preprocess.NORMALIZATION_CONFIG` / `BOOL_FIELDS` — preprocess.py:31-54
- 旧嵌入加载（将删除）— api.py:61-75 / predict.py:33-58
- 旧单条编码（将删除）— api.py:78-100 / predict.py:61-90
- 训练入口 — train_adaboost.py:23-106
- 测试脚本 — test.py:78-145

## 记忆回填

按用户偏好「计划完成后写一份到项目根目录 plan.md」，本轮实现阶段会将最终版同步回 `/Volumes/disk/Code/py/gongan/MGTAB-2/plan.md`。
