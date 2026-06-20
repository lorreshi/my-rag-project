# Requirements Document

## Introduction

本需求文档源自已批准的设计文档（`design.md` 的"本轮功能完善设计"一节），覆盖 Ingestion Pipeline 在 demo 阶段的**功能完善**改动，目标是补齐文档格式覆盖与中文/Token 检索质量，并支持按结构过滤。**不包含**性能/成本类优化、内容类型自动识别、语义/父子分块。

本轮共五块功能改动：

1. 多格式 Loader（Markdown / docx / xlsx）+ 工厂路由。
2. 共享分词器（jieba 中文词级，摄取与查询统一）。
3. 切分契约升级 + 结构化 metadata（支持按结构过滤）。
4. 默认递归切分增强（CJK 分隔 + Token 度量 + overlap 修正 + size 配置化/按 collection 覆盖）。
5. 表格感知切分 + 按 `doc_type` 路由。

设计文档中的 Correctness Property 8–14 在下列需求的验收标准中以「Validates: Property N」标注。

## Glossary

- **Loader**：把源文件解析为统一 `Document`（规范化 Markdown 文本 + metadata）的组件。
- **Splitter**：把文本切为若干切片的组件；本轮输出升级为 `SplitPiece(text, metadata)`。
- **SplitPiece**：切分单元，含文本与可选结构化 metadata（如 `sheet_name`、行区间）。
- **DocumentChunker**：`Splitter` 之上的业务适配层，产出带 id/metadata 的 `Chunk`。
- **doc_type**：文档来源格式标识（`pdf`/`markdown`/`docx`/`xlsx`），用于切分器路由。
- **collection**：知识库集合名，用于隔离数据与按集合覆盖切分参数。
- **jieba**：中文词级分词库，用于 BM25 稀疏索引词表（摄取与查询共用）。
- **tiktoken**：与 embedding 模型对齐的 BPE token 计数器，用于切分大小度量；与 BM25 词表无关。
- **BM25**：基于词频/逆文档频率的稀疏检索算法。

## Requirements

### 需求 1：多格式 Loader 与工厂路由

**User Story:** 作为知识库维护者，我希望除 PDF 外还能摄取 Markdown / docx / xlsx，这样我能在不同格式上验证检索效果。

#### 验收标准

1. WHEN 摄取一个文件 THEN 系统 SHALL 通过 `LoaderFactory.create(path)` 按扩展名选择对应 Loader。
2. WHEN 扩展名为 `.md`/`.markdown` THEN 系统 SHALL 使用 `MarkdownLoader` 直接读取文本（不经 MarkItDown），并解析 `![alt](path)` 本地图片链接为 `ImageRef`。
3. WHEN 扩展名为 `.docx` 或 `.xlsx` THEN 系统 SHALL 使用对应 Loader 经 MarkItDown 转为规范化 Markdown。
4. WHEN 任一 Loader 产出 Document THEN `metadata` SHALL 至少包含 `source_path`、`doc_type`、`doc_hash`；`doc_id` SHALL 形如 `{doc_type}_{doc_hash[:12]}`。
   - Validates: Property 10（统一输出契约）
5. IF 扩展名未注册 THEN `LoaderFactory.create` SHALL 抛出含可用扩展名列表的 `ValueError`。
   - Validates: Property 9（Loader 路由确定性）
6. IF 解析失败或可选依赖缺失（如图片提取）THEN 系统 SHALL 记录警告、返回尽力而为的结果且不阻塞摄取。
7. WHEN 装配 Pipeline THEN `IngestionPipeline.from_settings` 与 `scripts/ingest.py` SHALL 通过 `LoaderFactory` 选择 Loader，而非硬编码 `PdfLoader`。

### 需求 2：共享分词器（jieba）

**User Story:** 作为开发者，我希望摄取与查询使用同一套中文分词，这样 BM25 词表能对齐、中文检索不再按单字匹配。

#### 验收标准

1. WHEN 需要分词 THEN 系统 SHALL 通过单一 `BaseTokenizer` 实现（默认 `JiebaTokenizer`）产出 token，中文走 `jieba.cut`、ASCII 词/数字按正则、统一小写并去停用词。
2. WHEN 摄取侧 `SparseEncoder` 与查询侧 `QueryProcessor` 分词 THEN 二者 SHALL 使用由同一配置创建的同一分词器，对相同文本产出完全相同的 token 序列。
   - Validates: Property 8（分词器一致性）
3. WHEN 配置 `retrieval.tokenizer=regex` THEN 系统 SHALL 回退到旧的字符级行为（便于对比/降级）。
4. WHEN 分词器从 jieba 切换或升级 THEN 系统 SHALL 在文档/日志中提示**既有 BM25 索引需重建（重新摄取）**。
5. WHEN 实现共享分词器后 THEN `SparseEncoder` 与 `QueryProcessor` 内重复的字符级正则 SHALL 被移除，改为调用共享分词器。

### 需求 3：切分契约升级与结构化 metadata

**User Story:** 作为开发者，我希望切分器能携带结构信息到 chunk，这样下游能按 sheet/行等结构维度过滤与引用。

#### 验收标准

1. WHEN 定义切分接口 THEN `BaseSplitter` SHALL 提供 `split(text) -> list[SplitPiece]`，其中 `SplitPiece` 含 `text` 与 `metadata`。
2. WHEN 既有调用使用 `split_text(text) -> list[str]` THEN 系统 SHALL 保持向后兼容（默认基于 `split()` 仅取 `text`）。
3. WHEN `DocumentChunker` 构造 `Chunk` THEN 系统 SHALL 把对应 `SplitPiece.metadata` 合并进 `chunk.metadata`，并与既有 `chunk_index`/`image_refs`/`source_ref` 共存。
4. WHEN 散文型切分器（递归）产出切片 THEN `SplitPiece.metadata` SHALL 为空 dict，使下游行为与现状一致。
   - Validates: Property 14（路由确定性 / 散文型 metadata 为空）
5. WHEN 结构化 metadata 仅在切分时可得（如 `sheet_name`、行区间）THEN 系统 SHALL NOT 依赖 transform 阶段还原该信息（内容增强器无法可靠重建结构）。

### 需求 4：默认递归切分增强（CJK + Token + overlap + 配置）

**User Story:** 作为中文知识库维护者，我希望切分对中文友好、按 token 度量并可配置，这样中文长段落不被打成单字、中英文块大小可比。

#### 验收标准

1. WHEN 切分含中文的长文本 THEN 递归切分器 SHALL 在退到字符级之前先在中文标点（`。！？；，`）与英文句末处断开。
2. WHEN 不含换行的长中文段落被切分 THEN 结果 SHALL NOT 出现长度为 1 的字符级碎片（除非该段确无任何标点且超长，作为兜底）。
   - Validates: Property 11（中文不退化为单字）
3. WHEN `size_unit=token` THEN 切分器 SHALL 使用与 embedding 模型对齐的 tiktoken 计数器度量大小，且每块 token 数不超过 `chunk_size`（结构不可分的兜底块除外）。
   - Validates: Property 12（Token 度量生效）
4. WHEN `size_unit=char` THEN 切分器 SHALL 退化为字符计数（旧行为）。
5. WHEN 切分 THEN `chunk_size`/`chunk_overlap` SHALL 从 `settings.splitter` 读取，不再硬编码。
6. WHEN 应用 overlap THEN 系统 SHALL NOT 在衔接处强插空格（对 CJK 友好）。
7. WHEN 文本含代码块 THEN 切分 SHALL 保护代码块结构不被破坏。

### 需求 5：表格感知切分与 doc_type 路由

**User Story:** 作为维护者，我希望 Excel 表格按行/表头/sheet 智能切分并带结构信息，这样表格内容能被准确检索并支持按 sheet 过滤。

#### 验收标准

1. WHEN `DocumentChunker` 处理文档 THEN 系统 SHALL 依据 `document.metadata["doc_type"]` 选择切分器：命中 `splitter.by_doc_type` 用专用切分器，未命中用默认递归。
   - Validates: Property 14（路由确定性）
2. WHEN `doc_type=xlsx` THEN 系统 SHALL 路由到 `TableAwareSplitter`，输入为 loader 产出的 Markdown 表格文本（不绕开 Markdown 中转）。
3. WHEN 表格切分 THEN 每个表格 chunk 文本 SHALL 以表头开头（重复表头），且 SHALL NOT 跨 sheet 混行。
   - Validates: Property 13（表格 chunk 自包含）
4. WHEN 表格切分 THEN 每个表格 `SplitPiece.metadata` SHALL 含 `sheet_name`、`row_start`/`row_end`、`is_table=True`。
   - Validates: Property 13
5. WHEN 表格文本中夹杂非表格内容 THEN `TableAwareSplitter` SHALL 把非表格部分回退给默认递归切分处理。
6. IF 未配置 `by_doc_type` THEN 所有文档 SHALL 走默认递归切分（行为与现状一致）。

### 需求 6：按 collection 的切分参数覆盖

**User Story:** 作为维护者，我希望对不同集合用不同的 chunk 大小（短知识用小块、长教程用大块），这样无需改代码也无需系统自动猜测内容类型。

#### 验收标准

1. WHEN `split_document(document, collection)` 执行 THEN 系统 SHALL 按 `collection` 解析生效的 `chunk_size`/`chunk_overlap`（`splitter.overrides.{collection}` 优先于默认值）。
2. WHEN 某 collection 无覆盖配置 THEN 系统 SHALL 使用 `splitter` 默认值。
3. WHEN 区分长/短文档 THEN 系统 SHALL 仅依据显式配置，而 SHALL NOT 自动识别内容类型。

### 需求 7：查询侧按结构化 metadata 过滤

**User Story:** 作为查询用户，我希望能限定在某个 sheet 内检索，这样能精确定位表格来源。

#### 验收标准

1. WHEN 查询提供结构化过滤条件（如 `sheet_name`）THEN 系统 SHALL 通过既有通用 `filters` 通道按 chunk metadata 过滤，仅返回匹配的 chunk。
2. WHEN 引用展示 THEN 系统 SHALL 能在结果中携带表格 chunk 的 `sheet_name`/行区间（若存在）。
3. WHEN 引入结构化过滤 THEN 查询主流程（process → dense + sparse → fusion → rerank）SHALL 保持不变；该能力为可选过滤维度。
