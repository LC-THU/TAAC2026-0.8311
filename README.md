# PCVRHyFormer

> 面向 TAAC 2026：在 Baseline 的多序列 HyFormer 主干上增强输入结构、时间建模、特征交互与目标兴趣读取，AUC 从 `0.8127` 提升至 **`0.8311`**（`+0.0184`）。

## 上分总览

| 版本 | 相对上一版本的主要变化 | AUC | 单步提升 | 累计提升 |
| --- | --- | ---: | ---: | ---: |
| v1 | Baseline | 0.8127 | - | - |
| v3 | + Counterfactual Interest Residual (CIR) | 0.8142 | +0.0015 | +0.0015 |
| v4 | + Aligned Pair Tokenizer + Time Dense Features | 0.8150 | +0.0008 | +0.0023 |
| v5 | + Fourier Time Encoding + Feature Interaction | 0.8196 | +0.0046 | +0.0069 |
| v6 | + 2-layer Gated Deep Feature Interaction | 0.8289 | **+0.0093** | +0.0162 |
| v7 | + DIN Target Attention + CIR Gate Update | **0.8311** | +0.0022 | **+0.0184** |

![结果](result.png)

> 上表是逐步迭代路径，不是完全独立消融。`0.8311` 是单次最佳结果；同配置复现实验得到 `0.8285` 与 `0.8272`，小幅差异需要结合多次运行判断。

## 相比 Baseline 改了什么

Baseline 已包含 `RankMixerNSTokenizer + MultiSeqQueryGenerator + MultiSeqHyFormerBlock + RankMixer`：将用户/商品非序列特征投影为 token，用查询 token 读取多域行为序列，再做分类。本方案保留这条主干，主要补上以下能力：

| 能力 | Baseline | PCVRHyFormer 0.8311 |
| --- | --- | --- |
| 对齐字段表示 | 离散特征、连续特征分别进入通用路径 | 对有位置对应关系的 ID/value 单独联合编码为 4 个 token |
| 时间表示 | 序列 time bucket embedding | + 58 维时间统计特征；+ 连续 Fourier 时间编码 |
| 非序列特征交互 | token 投影后直接送入查询生成与 RankMixer | 送入主干前先经过 2 层门控乘法交互 |
| 目标相关历史读取 | 主要依赖融合 query 读取序列 | + CIR 目标/通用兴趣残差；+ DIN item-to-history 注意力 |

## 关键改进

### 1. CIR：增加目标兴趣残差

**Baseline** 通过融合后的 query 从历史中提取总体表示，但没有专门区分“用户本来就偏好什么”和“当前商品额外触发了什么兴趣”。

**本方案** 增加 `CounterfactualInterestResidual`，对同一组序列分别执行通用查询和商品条件查询：

```text
generic_interest = Attention(generic_query, history)
target_interest  = Attention(item_query, history)
cir_delta        = Adapter(target_interest - generic_interest, ...)
```

`generic_query` 不携带商品信息，`item_query` 来自候选商品 token。两者相减后得到更偏向当前商品的增量兴趣。CIR 的 adapter 末层采用零初始化，使该残差从 0 开始接入，不会在训练初始阶段直接破坏 baseline 主干。

### 2. Aligned Pair Tokenizer 与 58 维时间密集特征

**Baseline** 将用户离散字段与 dense 字段作为通用输入处理，即使某些字段的第 `j` 个 ID 与第 `j` 个连续值描述的是同一对象，也没有显式保存这种配对关系。

**本方案** 对 `fid 62, 63, 64, 65, 66, 89, 90, 91` 启用 `AlignedIntDensePairTokenizer`：

```text
同一位置的 (id_j, value_j)
  -> ID embedding + signed_log1p(value_j) 的 MLP 表示
  -> scale-shift 调制 ID embedding
  -> field 内注意力池化
  -> 4 个 learnable query 压缩为 4 个 aligned-pair tokens
```

训练配置使用 `scale_shift`：连续 value 不只被相加，还会学习缩放对应的 ID embedding。已单独编码的字段会从基础 user token 与 dense 投影路径中移除，避免重复计入。

同时，dataset 侧新增 `TimeDenseFeaturesComputer`，计算 `58` 维特征：

| 部分 | 维度 | 内容 |
| --- | ---: | --- |
| 全局时间 | 6 | 小时/星期周期编码、是否周末、是否工作时段 |
| 每个行为域 | 12 x 4 | 序列长度、最近行为间隔、`1h/6h/1d/7d` 行为计数、近期衰减统计 |
| 跨域汇总 | 4 | 活跃域数、总序列长度、最大近期计数、平均最近间隔 |

这相当于在 baseline 的“发生过哪些行为”之外，补上“当前是否处于活跃或临近转化的时间状态”。

### 3. Fourier 时间编码与显式特征交互

**Baseline** 的时间信息主要是离散间隔桶；非序列 token 在进入主干前没有专门的乘法交互模块。

**本方案** 将原始时间戳编码为多频率连续表示并加到序列 token 上：

```text
Fourier(t) = Linear([sin(t / period_k), cos(t / period_k)])
period_k: 1 天到 40 天的 12 个对数间隔周期
```

它补充了日内、周内和更长周期模式，而不是只描述行为距离当前样本有多远。

显式特征交互指的不是手工构造交叉特征，而是在模型内直接加入可学习的乘法项：

```text
X <- X + X * expert_proj(X)
```

其中 `X` 是非序列 token。Baseline 更多依赖后续网络间接学习组合关系；乘法交互则直接提供“用户状态 x 商品属性”“近期活跃 x 偏好匹配”一类条件组合的表达通道。

### 4. 两层门控 Deep Feature Interaction：最关键的提升

**与 Baseline 的差别最直接：**

```text
Baseline:
    ns_tokens -> QueryGenerator / RankMixer

0.8311:
    ns_tokens -> DeepFeatureInteraction x 2 -> QueryGenerator / RankMixer
```

两层门控交互的实现为：

```python
for layer in range(2):
    interaction = X * expert_proj_layer(X)
    X = X + sigmoid(gate_layer) * interaction
```

每层使用独立 `expert_proj`，且 `gate` 初始化为 `-3`，初始融合强度约为 `4.7%`。因此模型可以在保留 baseline 表示的基础上，逐渐学习更复杂的特征组合。

为什么它重要：

- baseline 没有这条显式乘法路径，复杂组合只能由后续主干间接拟合；
- 一层交互主要提供二阶组合，两层交互可继续组合已形成的交互表示；
- 小门控初始化降低了乘法交互在训练初期放大噪声的风险。

这一结论也有最强的实验支持：

| 对比 | AUC 变化 |
| --- | ---: |
| v5 单层交互 -> v6 两层门控交互 | `0.8196 -> 0.8289`（`+0.0093`） |
| v7 两层交互 -> v25 改回一层 | `0.8311 -> 0.8184`（`-0.0127`） |

### 5. DIN Target Attention 如何与 CIR 联合

**Baseline** 没有额外的 item-to-history 读取分支。CIR 引入后，模型能够计算目标兴趣相对通用兴趣的残差；v7 又加入 DIN，让候选商品直接查询进入 HyFormer 之前的序列 token：

```text
DIN:
    item_vec -> query
    sequence_tokens -> key/value
    din_delta = item 对各历史位置的加权汇总
```

DIN 与 CIR 不是串联，也没有共享注意力权重；当前实现是在主干输出上做两条并行残差融合：

```python
output = base_output
output = output + sigmoid(cir_gate) * cir_delta
output = output + sigmoid(din_gate) * din_delta
```

v7 中 `cir_gate=0`（初始权重 `0.5`），`din_gate=-3`（初始权重约 `0.047`）。可以理解为：

- CIR 强化“目标兴趣相对通用兴趣的增量”；
- DIN 补充“当前商品最匹配历史中的哪些具体位置”；
- 两条路径从不同角度补足 baseline 的目标感知能力，最后共同修正主干输出。

`v6 -> v7` 提升 `+0.0022`；将 CIR gate 回退的 v9 为 `0.8296`。这支持 DIN 与 CIR 联合是有效方向，但由于单次运行存在波动，二者的独立贡献仍应通过多 seed 消融判断。

## `run.sh` 配置对照

除模型结构外，最终启动配置也与 baseline 不同。下表来自两份 `run.sh` 及对应 `train.py` 的可核查差异；这些配置与结构改动共同构成最终方案，不等价于独立消融。

| 配置 | Baseline | PCVRHyFormer 0.8311 | 说明 |
| --- | ---: | ---: | --- |
| `--user_ns_tokens` | `5` | `4` | 为新增对齐字段 token 重新分配用户侧容量 |
| `--item_ns_tokens` | `2` | `3` | 增加商品侧 token 容量 |
| `--num_queries` | `2` | `1` | 配合新 token 组成调整 RankMixer 输入预算 |
| `--emb_skip_threshold` | `1000000` | `60000000` | 保留更多高基数特征 embedding |
| `--use_aligned_pair_tokens` | 无此模块 | 开启，`4` 个 token | 引入 ID/value 对齐表示 |
| `--aligned_pair_value_interaction` | 无此模块 | `scale_shift` | 连续值调制对应 ID embedding |
| `--drop_aligned_pair_from_base` | 无此模块 | 开启 | 避免对齐字段在基础路径重复计入 |
| `--num_interaction_layers` | 无此模块 | `2` | 启用核心两层门控特征交互 |
| `--seed` | 默认 `42` | `2002` | 固定最终复现实验随机种子 |
| 启动方式 | 单进程 `python3 -u` | 单卡 `python3 -u` / 多卡 `torchrun` | 新增 DDP 训练支持 |

两版默认均保留 `d_model=64`、`emb_dim=64`、`num_hyformer_blocks=2`、`num_heads=4`、`dropout_rate=0.01`、`loss_type=bce` 与相同序列长度配置；dense learning rate 则由 baseline 的 `1e-4` 调整为 `1.5e-4`。

补充说明：当前 `run.sh` 注释中出现了 `SlidingWindow RelativeTimeBias`，但代码仅在序列长度 `L <= 128` 时应用该分支，默认 `256/512` 序列长度下并未启用，因此这里不将 RTB 计作确认的涨分组件。

## 更多说明

更完整的模块公式、实验边界与扩展方向见 [TECHNICAL_ANALYSIS.md](TECHNICAL_ANALYSIS.md)。

当前最可靠的结论是：在保持 baseline 多序列主干不变的前提下，**两层门控 Deep Feature Interaction** 是主要性能来源；结构化输入、连续时间表示以及 CIR/DIN 目标读取则共同补全了模型对 pCVR 任务的表达能力。
