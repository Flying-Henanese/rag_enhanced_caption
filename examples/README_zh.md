# examples 使用指南

这个目录包含两个示例脚本，用来演示从 Markdown 文档处理到 LlamaIndex 高级检索的完整前置流程。

## 脚本说明

### 1. data_ingestion_pipeline.py

这个脚本负责把一个 Markdown 文件处理成适合 RAG 入库的 JSONL 数据。

它会完成：

- 读取 Markdown 文档
- 进行语义分块
- 清洗空块并修复父子关系
- 调用 VLM 为图片、表格等多模态内容生成增强描述
- 输出面向向量、docstore 和 BM25 检索的 JSONL 文件

默认输入文件是：

```text
test_resource/paddleocr.md
```

默认输出目录是：

```text
output/
```

输出文件格式为：

```text
output/<输入文件名>_index_new.jsonl
output/<输入文件名>_docstore_new.jsonl
output/<输入文件名>_sparse.jsonl
```

其中：

- `*_index_new.jsonl`：保存用于 embedding 和向量检索的精简文本
- `*_docstore_new.jsonl`：保存完整 Markdown 内容、标题路径、父子关系和元素类型
- `*_sparse.jsonl`：保存 backend-neutral searchable objects，供 BM25 / 关键词召回使用

### 2. llama_index_advanced_rag.py

这个脚本负责读取上一步生成的 JSONL 文件，并转换成 LlamaIndex 可以使用的数据结构和混合检索组件。

它会完成：

- 读取 `*_index_new.jsonl`、`*_docstore_new.jsonl` 和可选的 `*_sparse.jsonl`
- 将普通文本块转换为 LlamaIndex `TextNode`
- 将图片、表格等多模态块转换为 `IndexNode` 和完整内容节点
- 根据 `header_path` 构建章节父子关系
- 使用 `SimpleDocumentStore` 保存完整节点图
- 使用 `VectorStoreIndex` 建立向量索引
- 使用 vector top-k + BM25 top-k、RRF、`RecursiveRetriever`、可选 rerank、短上下文扩展和 `AutoMergingRetriever` 进行召回和上下文合并

需要注意：这个脚本目前是检索演示，不是完整问答系统。它只会召回最相关的一些上下文节点并打印出来，不会调用 LLM 生成最终答案。

## 安装依赖

建议在项目根目录执行：

```powershell
uv sync --extra demo
```

如果只运行第一个脚本，核心依赖通常已经足够；如果要运行第二个脚本，需要安装 `llama-index-core` 等 demo 依赖。

## 配置环境变量

两个脚本会从项目根目录的 `.env` 文件或系统环境变量中读取配置。

### VLM 配置

`data_ingestion_pipeline.py` 会调用 OpenAI-compatible 的视觉语言模型接口。

示例：

```env
VLM_API_KEY=你的_api_key
VLM_ENDPOINT=https://你的服务地址/v1/chat/completions
VLM_MODEL_NAME=你的视觉模型名称
```

如果不配置，默认会尝试访问：

```text
http://127.0.0.1:8000/v1/chat/completions
```

### 可选 PDF OCR 配置

`paddleocr_pdf_ocr.py` 和 `data_ingestion_pipeline.py --use-ocr` 使用硅基流动
PaddleOCR-VL，将 PDF 先逐页渲染为图片，再异步转换为 Markdown。该能力不会改变
默认 Markdown 入库流程。

```env
OCR_API_KEY=你的_api_key
OCR_ENDPOINT=https://api.siliconflow.cn/v1/chat/completions
OCR_MODEL_NAME=PaddlePaddle/PaddleOCR-VL-1.5
OCR_MAX_CONCURRENCY=2
OCR_TIMEOUT=120
```

需要在系统中安装 Poppler，以提供 `pdftoppm` 命令。例如 macOS 可执行
`brew install poppler`。OCR 结果默认写入 `output/ocr/<PDF 文件名>.md`，渲染页图
始终保存在同级的 `output/ocr/<PDF 文件名>/page-001.png` 等文件中。模型不会返回
PDF 内嵌图片文件；这些图片是本地渲染的页面图。

默认以 144 DPI 渲染页面，与 PaddleOCR 的 `fitz.Matrix(2, 2)` 默认渲染倍率一致；
可传入 `--dpi` 覆盖默认值。

### Embedding 配置

`llama_index_advanced_rag.py` 会根据 `EMBEDDING_MODEL_NAME` 判断是否启用 OpenAI-compatible embedding。

示例：

```env
EMBEDDING_API_KEY=你的_api_key
EMBEDDING_ENDPOINT=https://api.siliconflow.cn/v1/embeddings
EMBEDDING_MODEL_NAME=BAAI/bge-m3
```

### Rerank 配置

rerank 是可选能力。如果没有配置 `RERANK_API_KEY`，脚本会跳过 rerank，直接使用基础召回结果。

示例：

```env
RERANK_API_KEY=你的_api_key
RERANK_ENDPOINT=https://api.siliconflow.cn/v1/rerank
RERANK_MODEL_NAME=Qwen/Qwen3-Reranker-0.6B
```

## 推荐运行流程

### 第一步：生成 JSONL 数据

在项目根目录执行：

```powershell
uv run python examples\data_ingestion_pipeline.py
```

默认会处理：

```text
test_resource/paddleocr.md
```

并生成：

```text
output/paddleocr_index_new.jsonl
output/paddleocr_docstore_new.jsonl
output/paddleocr_sparse.jsonl
```

如需从 PDF 开始处理并启用 OCR：

```powershell
uv run python examples\data_ingestion_pipeline.py --input path\to\document.pdf --use-ocr
```

如需将页图链接写入 Markdown，让现有多模态增强流程处理它们：

```powershell
uv run python examples\data_ingestion_pipeline.py --input path\to\document.pdf --use-ocr --include-page-images
```

也可单独执行 OCR，并仅生成 Markdown：

```powershell
uv run python examples\paddleocr_pdf_ocr.py path\to\document.pdf --output-dir output\ocr
```

### 第二步：运行 LlamaIndex 检索示例

`llama_index_advanced_rag.py` 默认读取的是：

```text
output/rag-anything_index_new.jsonl
output/rag-anything_docstore_new.jsonl
output/rag-anything_sparse.jsonl
```

因此，如果想直接运行第二个脚本，需要先让第一个脚本处理 `test_resource/rag-anything.md`。

打开 `examples/data_ingestion_pipeline.py`，将末尾的：

```python
TARGET_MD_FILE = "test_resource/paddleocr.md"
```

改成：

```python
TARGET_MD_FILE = "test_resource/rag-anything.md"
```

然后再次执行：

```powershell
uv run python examples\data_ingestion_pipeline.py
```

成功后会生成：

```text
output/rag-anything_index_new.jsonl
output/rag-anything_docstore_new.jsonl
output/rag-anything_sparse.jsonl
```

之后运行：

```powershell
uv run python examples\llama_index_advanced_rag.py
```

脚本会使用内置的几个查询问题，打印召回并合并后的上下文。

## JSONL 到 LlamaIndex 的转换关系

第一步输出的 JSONL 不是直接交给 LlamaIndex 使用，而是在 `llama_index_advanced_rag.py` 中转换成 LlamaIndex 节点。

### index 文件

`*_index_new.jsonl` 示例结构：

```json
{
  "id": "chunk_id",
  "text_for_embedding": "用于向量检索的文本",
  "metadata": {
    "element_type": "text"
  }
}
```

这部分主要用于构建向量索引。

### docstore 文件

`*_docstore_new.jsonl` 示例结构：

```json
{
  "id": "chunk_id",
  "full_content": "完整 Markdown 内容",
  "parent_id": null,
  "header_path": ["一级标题", "二级标题"],
  "element_type": "text",
  "entities": []
}
```

这部分主要用于保存完整内容和上下文结构。

### sparse 文件

`*_sparse.jsonl` 示例结构：

```json
{
  "id": "stable-field-id",
  "owner_node_id": "chunk_id",
  "searchable_text": "项目：MiniRAG；描述：...",
  "field_type": "table_row",
  "metadata": {
    "row_index": 2
  },
  "schema_version": 1
}
```

这部分主要用于 BM25 / 关键词召回。`owner_node_id` 会映射回同一批 LlamaIndex node ID，因此 BM25 和向量召回最终仍然共用 docstore 中的完整内容。

### 普通文本块

普通文本块会被转换成 `TextNode`：

```python
TextNode(
    id_=chunk_id,
    text=text_for_embedding,
    metadata={"type": "chunk", "full_content": full_content},
)
```

其中：

- `text` 用于 embedding 和向量检索
- `metadata["full_content"]` 用于保存完整上下文

### 图片和表格等多模态块

图片、表格等非文本块会被拆成两类节点：

```python
element_node = TextNode(
    id_=f"{chunk_id}_full",
    text=full_content,
    metadata={"type": "element"},
)
```

```python
index_node = IndexNode(
    id_=chunk_id,
    text=text_for_embedding,
    index_id=element_node.id_,
)
```

这样设计的目的是：

- `IndexNode` 负责用 VLM 摘要参与向量检索
- `element_node` 负责保存完整的图片、表格或复杂 Markdown 内容
- 检索命中 `IndexNode` 后，`RecursiveRetriever` 可以跳转到完整节点

### 标题路径和父子关系

`header_path` 会被转换成章节聚合节点，并通过 LlamaIndex 的 `NodeRelationship.PARENT` 和 `NodeRelationship.CHILD` 建立层级关系。

这些关系会被短上下文扩展和 `AutoMergingRetriever` 使用，用于在召回后补齐相邻短节点，并自动合并更完整的上下文。

## 常见问题

### 为什么第二个脚本找不到 JSONL 文件？

通常是因为文件名前缀不一致。

第二个脚本默认读取：

```text
output/rag-anything_index_new.jsonl
output/rag-anything_docstore_new.jsonl
output/rag-anything_sparse.jsonl
```

如果第一步处理的是 `paddleocr.md`，则实际生成的是：

```text
output/paddleocr_index_new.jsonl
output/paddleocr_docstore_new.jsonl
output/paddleocr_sparse.jsonl
```

解决方法是让两个脚本使用同一个输入文件前缀，或者修改 `llama_index_advanced_rag.py` 中的 `DOCSTORE_PATH`、`INDEX_PATH` 和 `SPARSE_PATH`。

### 第二个脚本会生成答案吗？

不会。当前脚本只做检索和上下文合并，不做最终问答生成。

如果要扩展成问答系统，需要在：

```python
final_results = adv_auto_merging_retriever.retrieve(query)
```

之后，把召回结果拼接为上下文，再调用 LLM 生成答案。

### 不配置 rerank 可以运行吗？

可以。没有 `RERANK_API_KEY` 时，脚本会跳过 rerank。

### 不配置 VLM 可以运行第一个脚本吗？

如果文档中存在需要增强的图片或表格，通常需要配置 VLM。否则在调用 VLM 时可能失败。若只想测试纯文本分块，可以选择输入不包含图片和复杂表格的 Markdown，或自行改造脚本跳过 VLM 增强阶段。
