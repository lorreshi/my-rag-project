# Requirements Document

（Phase D 检索增强 — phase-d-retrieval-enhancements）

## Introduction

（引言）

本特性对**已实现的 Phase D 检索流水线**（`QueryProcessor` / `DenseRetriever` / `SparseRetriever` / `Fusion` / `HybridSearch` / `Reranker` / `scripts/query.py`）做一次**重构与增强**：修复"配置漂移"与"规范落地缺口"，并补齐 `DEV_SPEC.md` 已声明但代码未兑现的能力，同时纳入四项进阶检索增强（多查询、HyDE、MMR、相关性阈值）。

总体原则：

- 所有新能力均为**可插拔、默认关闭（或默认安全值）**的开关，默认配置下行为与现状逐项一致（向后兼容）。
- 凡"会改变 BM25 token 形态"的归一化，必须放在**索引侧与查询侧共用的归一化层**，保证词表对称。
- 保留既有的**优雅降级**（单路失败不影响整体）与 metadata 过滤的 **strict/lenient** 双策略。

本文档需求与设计文档（`design.md`）的增强点 #1–#11 及 Correctness Properties 一一对应，供后续派生 tasks 与属性测试引用。

## Glossary

（术语）

- **词表对称性**：同一文本经索引侧（`SparseEncoder`）与查询侧（`QueryProcessor`）产出的 BM25 token 序列相同。
- **abstain（无答案）**：检索结果整体相关性不足时，返回空而非强行返回低质结果。
- **候选池宽度**：融合前每路检索取回的候选数量。

## Requirements

### Requirement 1: 确定性文本归一化（NFKC + 大小写 + 可选繁简）

**User Story:** 作为中文知识库的使用者，我希望查询中的全角/半角、大小写、（可选）繁简差异被统一处理，以便相同含义的不同写法都能命中同一批文档。

#### Acceptance Criteria

1. THE SYSTEM SHALL 提供一个共享的 `normalize_text` 工具，依次执行 NFKC 归一化、大小写折叠（casefold），并在启用时执行繁简归一（t2s）。
2. THE SYSTEM SHALL 让索引侧（`SparseEncoder`）与查询侧（`QueryProcessor`）的分词入口调用同一 `normalize_text`，且两侧使用相同的归一化参数。
3. WHEN 对同一文本 `t` 分别在索引侧与查询侧归一化并分词，THE SYSTEM SHALL 产出完全相同的 token 序列（词表对称性）。
4. THE SYSTEM SHALL 让 `QueryProcessor._normalize`（服务稠密侧）复用同一 `normalize_text`，使稠密查询文本与稀疏词表处于同一归一化 form。
5. WHEN 对任意文本重复应用 `normalize_text`，THE SYSTEM SHALL 得到与单次应用相同的结果（幂等）。
6. WHERE `normalize_to_simplified` 为开启，IF 运行环境未安装 OpenCC，THEN THE SYSTEM SHALL 记录 warning 并跳过繁简步骤（NFKC 与 casefold 仍照常执行），不得中断查询或摄取。
7. WHILE `enable_nfkc` / `normalize_casefold` / `normalize_to_simplified` 全部按默认值（开/开/关）配置，THE SYSTEM SHALL 仅启用零外部依赖的归一化（NFKC + casefold）。
8. THE SYSTEM SHALL 在文档中标注：启用任一改变 token 形态的归一化后，必须重跑摄取以重建 BM25 索引。

### Requirement 2: 从查询文本解析 filters

**User Story:** 作为使用者，我希望系统能从自然语言查询中识别结构化约束（如指定某张表），以便检索范围自动收窄到正确的数据子集。

#### Acceptance Criteria

1. THE SYSTEM SHALL 提供可插拔的 `BaseFilterExtractor` 接口及一个规则实现 `RuleBasedFilterExtractor`。
2. WHERE `enable_filter_extraction` 为开启，THE SYSTEM SHALL 在查询处理阶段调用抽取器，将识别到的结构化约束并入通用 `filters`。
3. WHEN 外部预置 filters 与抽取得到的 filters 存在同一键，THE SYSTEM SHALL 以外部预置值为准（外部优先，不被抽取结果覆盖）。
4. THE SYSTEM SHALL 在合并后丢弃所有值为 None 的 filter 键。
5. WHERE `enable_filter_extraction` 为关闭（默认），THE SYSTEM SHALL 表现得与现状一致，即 `filters` 等于"仅丢弃 None 的外部 filters"。
6. IF 抽取器内部发生异常，THEN THE SYSTEM SHALL 吞掉异常并返回空抽取结果（`{}`），不得中断查询。
7. THE SYSTEM SHALL 保证抽取出的结构化键（`sheet_name`/`row_start`/`row_end`/`is_table`）能被既有的 strict 过滤策略消费。

### Requirement 3: 候选池宽度配置接线

**User Story:** 作为运维/调参者，我希望 `settings.yaml` 中的 `top_k_dense` / `top_k_sparse` 等召回宽度配置真正生效，以便无需改代码即可调节召回。

#### Acceptance Criteria

1. THE SYSTEM SHALL 依据 `settings.retrieval.top_k_dense` 与 `top_k_sparse` 计算稠密/稀疏两路的候选池宽度，而非忽略它们。
2. THE SYSTEM SHALL 从 `settings.retrieval.candidate_multiplier` 读取候选池倍数，不再硬编码为 2。
3. WHEN 计算每路候选池宽度，THE SYSTEM SHALL 取"请求 top_k 与对应配置值的较大者"再乘以 `candidate_multiplier`。
4. WHERE 相关配置缺省，THE SYSTEM SHALL 回退到与当前实现等价的默认值（`top_k_dense=20`、`top_k_sparse=20`、`candidate_multiplier=2`）。

### Requirement 4: 加权融合与 FusionFactory

**User Story:** 作为调参者，我希望融合算法可配置且可对两路加权，以便按查询类型偏向 BM25 或稠密召回。

#### Acceptance Criteria

1. THE SYSTEM SHALL 提供 `FusionFactory`，依据 `settings.retrieval.fusion_algorithm` 创建融合器（支持 `rrf` 与 `weighted_sum`），与项目其它工厂形态一致。
2. THE SYSTEM SHALL 抽出 `BaseFusion` 抽象接口，`ReciprocalRankFusion` 与 `WeightedSumFusion` 均实现它。
3. THE SYSTEM SHALL 让 `HybridSearch.from_settings` 通过 `FusionFactory` 创建融合器，而非硬编码 `ReciprocalRankFusion`。
4. THE SYSTEM SHALL 让 `ReciprocalRankFusion` 支持每路可选权重，按公式 `fused = Σ weight_i / (k + rank)` 计算。
5. WHEN `fusion_weights` 缺省或各路权重相等，THE SYSTEM SHALL 产出与现有无权 RRF 逐项一致的排序（向后兼容）。
6. WHEN 融合多条排名列表，THE SYSTEM SHALL 使结果与列表传入顺序无关（融合可交换）。
7. IF `fusion_algorithm` 为未知值，THEN THE SYSTEM SHALL 抛出 `ValueError`（在构建期暴露，而非查询期静默）。
8. THE SYSTEM SHALL 从 `settings.retrieval.rrf_k` 读取 RRF 常数，不再硬编码为 60。

### Requirement 5: 稀疏路前置过滤

**User Story:** 作为使用者，我希望在多集合/带过滤的查询中，BM25 路也能前置过滤，以便稀疏召回不被无关候选稀释。

#### Acceptance Criteria

1. THE SYSTEM SHALL 让 `SparseRetriever.retrieve` 接受可选 `filters` 形参。
2. WHEN 传入 filters，THE SYSTEM SHALL 按 `sparse_filter_overfetch` 倍数超额取回，在 `get_by_ids` 解析阶段按 metadata 过滤，再截断到 top_k。
3. THE SYSTEM SHALL 在稀疏前置过滤中复用与 `HybridSearch._apply_metadata_filters` 完全相同的 strict/lenient 判定（建议抽为共享 helper）。
4. THE SYSTEM SHALL 保留融合后的后置过滤作为 safety net（前置不替代后置）。
5. WHEN 对同一候选全集比较，THE SYSTEM SHALL 使"前置过滤 + 后置兜底"的最终命中集合等价于"仅后置过滤"的命中集合（前移只缩小候选，不增删合规命中）。
6. WHERE 未传入 filters，THE SYSTEM SHALL 表现得与现状一致（不超额取回、不过滤）。

### Requirement 6: rerank 分数尺度统一

**User Story:** 作为使用者/调试者，我希望最终结果列表的分数在整列上一致可比，以便正确解读相关性并为阈值提供基准。

#### Acceptance Criteria

1. WHEN 重排成功并合并精排段（head, cross-encoder 分数）与未重排段（tail, RRF 分数），THE SYSTEM SHALL 使合并列表的 `score` 整列单调不增。
2. THE SYSTEM SHALL 使 head 段全部排在 tail 段之前。
3. THE SYSTEM SHALL 将统一前的原始分数与来源保存到 metadata（`raw_score` 与 `score_source ∈ {cross_encoder, rrf}`）。
4. THE SYSTEM SHALL 保留既有的 `rerank_fallback` 与 `rerank_backend` metadata。
5. WHERE 走 fallback（backend 失败退回融合全序）或 `NoneReranker`，THE SYSTEM SHALL 输出本就单调的分数，无需额外处理。

### Requirement 7: 同义词/别名 OR-扩展进 BM25

**User Story:** 作为使用者，我希望关键词的同义词/别名能并入 BM25 查询，以便专有名词的不同写法也能召回，同时不增加稠密侧成本。

#### Acceptance Criteria

1. THE SYSTEM SHALL 在 `ProcessedQuery` 中新增 `expanded_keywords` 字段（原词 + 同义词/别名）。
2. THE SYSTEM SHALL 保证 `expanded_keywords` 去重保序，且其前缀等于 `keywords`（原词在前）。
3. WHERE `enable_synonym_expansion` 为开启，THE SYSTEM SHALL 将 `expanded_keywords` 传给稀疏路；稠密路仍使用 `normalized_query` 单次检索。
4. WHERE `enable_synonym_expansion` 为关闭（默认），THE SYSTEM SHALL 表现得与现状一致（稀疏路使用原始 `keywords`）。
5. IF `synonym_source` 指向不存在的文件，THEN THE SYSTEM SHALL 记录 warning 并降级为空同义词表（等同不扩展）。

### Requirement 8: 多查询扩展（Multi-Query，可选）

**User Story:** 作为使用者，我希望系统能把一个问题改写成多个变体并分别稠密检索，以便提升召回覆盖面，且该能力可开关。

#### Acceptance Criteria

1. THE SYSTEM SHALL 提供可插拔的 `BaseQueryTransform`，模式由 `retrieval.query_transform` 选择（`none | multi_query | hyde`），默认 `none`。
2. WHERE 模式为 `multi_query`，THE SYSTEM SHALL 调用 LLM 将查询改写为最多 `multi_query_count` 个变体，并保证原始查询始终在变体集合内、去重保序。
3. THE SYSTEM SHALL 对每个变体各执行一次稠密检索，产出多条稠密列表，并连同稀疏列表一起交给既有 `Fusion.fuse`（融合层不改动）。
4. THE SYSTEM SHALL 通过 `query_transform_concurrency` 限制并发 embedding 调用数，并支持 `query_transform_cache`（query→变体）缓存。
5. IF LLM 改写调用失败，THEN THE SYSTEM SHALL 降级为单查询（`[query]`）并标记 `degraded=True`，不得中断查询。
6. WHERE 模式为 `none`（默认），THE SYSTEM SHALL 仅以原始查询执行单次稠密检索（行为不变）。

### Requirement 9: HyDE 假设性文档嵌入（可选）

**User Story:** 作为非结构化文档（PDF/Markdown 等）的使用者，我希望系统先生成假设性答案再嵌入检索，以便缓解问题与答案的向量空间错位。

#### Acceptance Criteria

1. WHERE 模式为 `hyde`，THE SYSTEM SHALL 调用 LLM 生成假设性答案文档，并将其嵌入用于稠密检索。
2. WHERE `hyde_augment` 为开启，THE SYSTEM SHALL 同时使用原始查询与假设文档（两者均作为稠密列表进入融合）；为关闭时仅使用假设文档。
3. WHEN 目标 `doc_type` 命中 `hyde_skip_doc_types`（如 `xlsx`），THE SYSTEM SHALL 跳过 HyDE 并退化为以原始查询单次稠密检索。
4. IF LLM 生成调用失败，THEN THE SYSTEM SHALL 降级为单查询并标记 `degraded=True`，不得中断查询。
5. THE SYSTEM SHALL 复用既有 `LLMFactory` / `BaseLLM`，不引入新的 LLM 依赖。

### Requirement 10: MMR 多样性 / 去重（可选）

**User Story:** 作为使用者，我希望最终结果在保持相关性的同时减少冗余，以便缓解同字/同表相似行扎堆的问题。

#### Acceptance Criteria

1. WHERE `enable_mmr` 为开启，THE SYSTEM SHALL 在重排之后、最终 top_k 裁剪之前执行 MMR 重排。
2. THE SYSTEM SHALL 按公式 `score = λ·相关性 − (1−λ)·与已选集合的最大相似度` 选择结果，其中 λ 取自 `mmr_lambda`。
3. THE SYSTEM SHALL 优先复用候选已有的稠密向量计算相似度；IF 某候选缺少向量，THEN THE SYSTEM SHALL 经 embedding 客户端补算，或在无法获取时记录 warning 并跳过 MMR（退回原序）。
4. WHEN `mmr_lambda >= 1.0`，THE SYSTEM SHALL 输出与未启用 MMR 相同的顺序。
5. WHERE `enable_mmr` 为关闭（默认），THE SYSTEM SHALL 不改变结果顺序（恒等）。

### Requirement 11: 相关性阈值 / 无答案（abstain，通用基线）

**User Story:** 作为上层 MCP/LLM 的集成方，我希望当检索整体相关性不足时系统返回"无答案"，以便上游不被低质上下文误导而产生幻觉。

#### Acceptance Criteria

1. THE SYSTEM SHALL 在流水线末端（重排/MMR 之后）施加可配置的 `min_score_threshold` 闸门。
2. WHEN `min_score_threshold > 0` 且 Top1 结果分数低于该阈值，THE SYSTEM SHALL 返回空结果（abstain）。
3. WHERE `min_score_threshold <= 0`（默认），THE SYSTEM SHALL 不介入，原样返回结果。
4. WHEN 触发 abstain，THE SYSTEM SHALL 使上层 MCP 工具据此输出"知识库未覆盖"语义，而非强行拼装答案。
5. THE SYSTEM SHALL 在文档中标注：阈值具体取值的校准依赖评估集，延后到后续阶段；本期仅落通用机制。

### Requirement 12: 向后兼容与优雅降级（横切）

**User Story:** 作为维护者，我希望在所有新开关关闭时系统行为与现状完全一致，并且任何单点失败都不会拖垮整条查询，以便安全地分阶段启用新能力。

#### Acceptance Criteria

1. WHILE `query_transform=none` 且 `enable_mmr=false` 且 `min_score_threshold<=0` 且 `enable_filter_extraction=false` 且 `enable_synonym_expansion=false`，THE SYSTEM SHALL 产出与 #1–#7 基线逐项一致的结果（进阶增强默认关 = 基线）。
2. IF 稠密路或稀疏路任一抛出异常，THEN THE SYSTEM SHALL 降级为另一路结果而非整体失败（保留既有降级行为）。
3. THE SYSTEM SHALL 保持 metadata 过滤的 strict（结构化键 missing→exclude）/ lenient（通用键 missing→include）双策略不变。
4. THE SYSTEM SHALL 为所有新增配置项提供使行为退化为现状的默认值。
5. WHERE 任一可选增强所需的外部依赖（OpenCC / LLM / 向量）不可用，THE SYSTEM SHALL 记录 warning 并降级该增强，不得中断查询。
