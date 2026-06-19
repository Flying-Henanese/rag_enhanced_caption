# RAG 增强注释工具包

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

[**English**](README.md) | [**中文**](README_zh.md)

`rag-enhanced-caption` 是一个面向 RAG 入库前最后一公里处理的轻量 Python 工具包，专门处理已经解析好的 Markdown 文档。

它主要解决三件事：
- 从 Markdown 结构中提取图片、表格的局部上下文
- 用视觉大模型为多模态元素生成可检索的语义摘要
- 生成保留文档结构的语义切片，供向量检索和回表使用

项目默认假设上游已经完成 PDF / Office / 图片到 Markdown 的转换。它本身不是 PDF 解析器，也不是 OCR 引擎。

## 输出结果

CLI 处理一份 Markdown 后，会产出三类文件：
- `<name>_enhanced.md`：注入多模态分析结果后的 Markdown 预览稿
- `<name>_index.jsonl`：面向向量库 / Embedding 的轻量记录
- `<name>_docstore.jsonl`：面向文档库的完整父子记录

这种拆分方式的目标是：让向量检索只看干净的语义文本，同时保留完整 Markdown、表格、图片引用供召回后展开。

## 核心能力

- 基于 `markdown-it-py` 的结构感知 Markdown 切分
- 通过 `header_path` 保留标题层级路径
- 自动绑定图片与题注，减少语义断裂
- 通过 OpenAI 兼容接口调用 VLM，对表格 / 图片做语义增强
- 输出适合多向量检索的 parent-child 结构
- 支持远程或本地 Embedding 客户端用于语义切分

## 当前仓库范围

当前仓库的核心内容是：
- `rag-caption` CLI，负责端到端处理
- `src/rag_enhanced_caption/` 下的可复用 Python 模块
- `examples/` 下的高级入库与检索流程
- 针对分块、HTML 清洗、VLM 增强和检索相关行为的测试

下面这份 README 以当前仓库实际内容为准。历史版本里提到的独立 `backend/` 应用不在当前仓库中，因此不再保留相关说明。

## 安装

### 直接安装当前项目

```bash
pip install .
```

### 使用 `uv` 搭建开发环境

```bash
git clone https://github.com/Flying-Henanese/rag_enhanced_caption.git
cd rag_enhanced_caption
uv sync
```

## 环境变量

建议在项目根目录或运行目录放置本地 `.env` 文件。

### VLM 配置

当你希望对图片 / 表格生成语义摘要时，需要配置：

```env
VLM_API_KEY=your_api_key
VLM_ENDPOINT=https://api.siliconflow.cn/v1/chat/completions
VLM_MODEL_NAME=Qwen/Qwen2.5-VL-72B-Instruct
```

说明：
- `VLM_ENDPOINT` 需要兼容 OpenAI Chat Completions 协议。
- 如果你接的是本地兼容服务，也可以复用同一套客户端。

### Embedding 配置

用于语义切分阶段的远程 Embedding：

```env
EMBEDDING_API_KEY=your_api_key
EMBEDDING_ENDPOINT=https://api.siliconflow.cn/v1/embeddings
EMBEDDING_MODEL_NAME=BAAI/bge-m3
EMBEDDING_TIMEOUT=60
```

可选的本地 Embedding 模型：

```env
LOCAL_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

说明：
- 如果没有可用的远程 Embedding 客户端，解析器会退化为更简单的文本切分策略。
- 本地 Embedding 依赖 `sentence-transformers`，默认并不会自动安装。

## CLI 用法

安装完成后可直接执行：

```bash
rag-caption INPUT_MARKDOWN OUTPUT_DIR
```

示例：

```bash
rag-caption ./test_resource/paddleocr.md ./output
```

预期输出：

```text
output/
├── paddleocr_enhanced.md
├── paddleocr_index.jsonl
└── paddleocr_docstore.jsonl
```

## JSONL 结构示例

`_index.jsonl` 存放面向向量化的紧凑记录：

```json
{"id": "p_xxx", "text_for_embedding": "...", "metadata": {"type": "parent", "source": "demo.md"}}
{"id": "c_xxx_0", "parent_id": "p_xxx", "text_for_embedding": "...", "metadata": {"type": "child", "source": "demo.md"}}
```

`_docstore.jsonl` 存放召回后展开使用的完整记录：

```json
{
  "id": "c_xxx_0",
  "parent_id": "p_xxx",
  "type": "child",
  "text_for_embedding": "语义摘要",
  "full_content": "![image](path)\n\n...analysis...",
  "metadata": {
    "chunk_type": "multimodal",
    "image_url": "path",
    "entities": ["entity_a", "entity_b"]
  }
}
```

## Python 用法

### 1. 对 Markdown 做结构化切分

```python
from rag_enhanced_caption import semantic_chunk_with_metadata

markdown = """
# Demo

Some text.

| Name | Value |
| --- | --- |
| A | 1 |
"""

records = semantic_chunk_with_metadata(
    markdown_content=markdown,
    file_id="demo",
    filename="demo.md",
    parser_config={"chunk_token_num": 512},
)

for record in records:
    print(record["id"], record["metadata"]["element_type"])
```

### 2. 使用 VLM 增强多模态切片

```python
import asyncio

from rag_enhanced_caption import (
    MarkdownMultimodalProcessor,
    create_default_vlm_client,
    semantic_chunk_with_metadata,
)

markdown = """
# Demo

![chart](./chart.png)
Figure 1. Revenue growth by region.
"""


async def main() -> None:
    chunks = semantic_chunk_with_metadata(
        markdown_content=markdown,
        file_id="demo",
        filename="demo.md",
    )

    processor = MarkdownMultimodalProcessor(
        vlm_func=create_default_vlm_client(),
        max_concurrency=2,
    )

    enriched = await processor.enrich_chunks(chunks, base_dir=".")
    for chunk in enriched:
        print(chunk["id"], chunk.get("text_for_embedding", ""))


asyncio.run(main())
```

## BM25 + Embedding 混合检索

高级示例提供了第二条面向精确关键词的召回路径，用于补充可能被语义摘要弱化或省略的字面信息。它尤其适合表头、表格行、日期、邮箱、产品名称、标识符等内容。

这条路径以增量方式工作，不改变 `rag-caption` CLI 的输出约定。`examples/data_ingestion_pipeline.py` 会在向量索引和 docstore 文件之外，额外生成 `<name>_sparse.jsonl`。其中每一行都是与后端无关的 searchable object：

```json
{
  "id": "stable-field-id",
  "owner_node_id": "rag-anything_chunk_89",
  "searchable_text": "项目：MiniRAG；描述：...",
  "field_type": "table_row",
  "metadata": {"row_index": 2},
  "schema_version": 1
}
```

searchable object 的抽取不依赖 LLM。目前的确定性抽取器支持 Markdown / HTML 表格的表头和数据行，以及日期、邮箱等固定格式字段。抽取接口本身是可扩展的，后续可以继续增加新的固定格式字段或存储后端。当前使用 JSONL 持久化；其数据结构与 BM25 实现解耦，未来可以映射到 Elasticsearch 或 MongoDB。

查询时，`examples/llama_index_advanced_rag.py` 使用以下流程：

```text
Embedding top-k + BM25 top-k
               ↓
        倒数秩融合（RRF）
               ↓
RecursiveRetriever → reranker → 短上下文扩展 → AutoMerge
```

RRF 根据排名位置进行融合，避免直接比较量纲不同的向量分数和 BM25 分数。两条召回路径最终都映射到相同的 LlamaIndex node ID，因此完整内容仍然以现有 docstore 为准。如果 sparse JSONL 不存在或内容为空，示例会自动退化为纯向量检索。

可以先为 `rag-anything.md` 构建示例数据，再运行高级检索示例：

```bash
uv run python -c "import asyncio; from examples.data_ingestion_pipeline import process_document; asyncio.run(process_document('test_resource/rag-anything.md'))"
uv run python examples/llama_index_advanced_rag.py
```

入库命令会生成：

```text
output/
├── rag-anything_index_new.jsonl
├── rag-anything_docstore_new.jsonl
└── rag-anything_sparse.jsonl
```

## 包结构

```text
src/rag_enhanced_caption/
├── cli.py
├── chunker/
│   ├── dispatcher.py
│   ├── embed_client.py
│   ├── parsers/semantic.py
│   └── utils/
├── enhancer/
│   ├── cleaning_utils.py
│   ├── context_extractor.py
│   ├── processor.py
│   ├── prompts.py
│   └── vlm_client.py
├── lexical_search/
│   ├── bm25.py
│   ├── builder.py
│   ├── extractors.py
│   ├── fusion.py
│   ├── repository.py
│   └── schema.py
└── integrations/llama_index/
    └── hybrid_retriever.py
```

## 测试覆盖

当前仓库包含以下类型的测试：
- 语义切分与解析器边界场景
- HTML 表格清洗 / 提取
- VLM 增强行为
- searchable object 抽取、JSONL 持久化、BM25 排序和 RRF 融合
- 向量与 BM25 候选结果的 LlamaIndex 集成
- 面向集成的检索、重排相关流程

常用命令示例：

```bash
uv run pytest tests/test_semantic_parser_fixes.py
uv run pytest tests/test_html_table.py
uv run pytest tests/test_processor.py
```

部分测试依赖外部 API 凭证；如果缺少对应环境变量，会自动跳过。

## 注意事项与限制

- 输入必须已经是 Markdown。
- VLM 增强依赖外部或本地兼容接口。
- SVG 图片当前会跳过 VLM 分析。
- 某些检索相关测试依赖可选集成代码和外部服务。

## 开源协议

Apache-2.0。
