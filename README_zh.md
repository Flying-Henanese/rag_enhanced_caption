# RAG 增强注释与分块工具包 (RAG Enhanced Caption & Chunking Toolkit)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

[**English**](README.md) | [**中文**](README_zh.md)

这是一个模块化的多模态 RAG (Retrieval-Augmented Generation) 增强工具包。本项目提供了高度解耦的模块，用于精准的 Markdown 上下文提取、基于视觉大模型 (VLM) 的图片/表格注释增强，以及感知文档结构的语义分块。

灵感来源于 [RAG-Anything](https://github.com/HKUDS/RAG-Anything)。

## 💡 我们的理念：意图驱动 vs OCR驱动

不同于标准的仅仅执行 OCR 和转录图片文字的多模态 RAG 工具，本工具包专注于**解释意图 (Illustrative Intent)**：
- **感知上下文 (Context-Aware)**：VLM 不仅能“看”到图片/表格，还能知道它在文档中的**位置**，以及周围的文本讲述了**什么**。
- **聚焦意图 (Intent-Focused)**：工具会解释一个元素**为什么**存在（例如：“该图表证明了在 2.1 节中提到的高并发能力”），而不是单纯地罗列图表中的数字。
- **防止注意力劫持 (No Attention Hijacking)**：通过生成简洁、意图驱动的摘要，我们能够防止大语言模型 (LLM) 在生成答案时被图片中无关的原始数据“分散注意力”。

## 🎯 适用范围与前提：文档处理的“最后一公里”

请注意，**本工具包被设计为处理“已经解析好”的 Markdown 文档。** 它并非一个 PDF 解析器或 OCR 引擎（你可以在上游流程使用像 MinerU、Docling 或 Marker 等专门的文档提取工具）。

本项目作为数据灌入向量数据库 (Vector DB) 进行检索，或者用于构建知识图谱 (Knowledge Graph) 之前的关键**“最后一公里”**加工步骤——负责精炼、强化和结构化原本零散的 Markdown 数据。

## ✨ 核心能力

- **基于 VLM 的增强注释 (`enhancer`)**：自动分析文档中的图片和表格。分析结果将被封装在可折叠的 `<details>` HTML 块中，并精准注入到该元素的**正下方**，确保最佳的阅读体验和逻辑连贯性。
- **外科手术式上下文提取**：利用 `markdown-it-py` 进行语法树解析，为 VLM 收集深度的情境感知信息（包括面包屑导航、父级标题和相邻段落）。
- **结构感知的语义分块 (`chunker`)**：区别于传统机械地按字数 (Character-count) 切分，我们的引擎基于抽象语法树 (AST) 解析 Markdown，不仅能保留文档的原始层级结构和上下文，更结合 Embedding 模型进行真正的语义聚类切分。

### 🧬 混合分块流水线 (Hybrid Chunking Pipeline)

我们的分块引擎采用多阶段处理流水线，旨在最大程度保留文档的原始逻辑语境，同时剔除检索噪音：

1.  **阶段一：AST 结构化解析**：利用 `markdown-it-py` 将文档解析为 Token 语法树。这确保了诸如表格 (Table)、数学公式 (Math Block) 和代码块 (Code Fence) 等结构化元素被视为原子单元，而不会被暴力切断。
2.  **阶段二：标题路径聚合 (Breadcrumbs)**：系统会为每一个切片自动注入完整的层级路径（例如：`# 一级标题|二级标题|子标题`）作为前缀。这种“情境感知”能力确保了即使是深层的孤立切片，也依然携带其在文档中的位置信息。
3.  **阶段三：图文/表文绑定**：专门的探测器会寻找图片链接及其随后的题注（如：`![alt](url)\n图 1：描述文本`）。系统会强制将它们绑定在同一个切片中，防止在检索时发生语义断裂。
4.  **阶段四：层次化语义切分**：
    *   **段落级优先**：首先尝试在自然的段落边界 (`\n\n`) 进行切分。
    *   **行级细化**：如果段落依然超长，则尝试在换行符 (`\n`) 处切分。
    *   **向量聚类 (兜底策略)**：如果单行文本仍然超过 Token 限制（例如极长的长难句），引擎将触发**基于 Embedding 的聚类算法**。它将句子转化为向量，通过语义相似度寻找最合乎逻辑的“切割点”，确保即使在极端情况下，切分也发生在语义转折处，而非单词中间。

- **父子块 RAG 策略 (Parent-Child RAG)**：内置的业务工作流能够将文档拆分为**易于检索的高密度“子块”**（仅包含 AI 意图摘要）和**富含上下文的“父块”**（原始文本内容）。
- **完全解耦**：你可以根据需要，单独使用分块器 (Chunker) 、增强器 (Enhancer)，或是将它们结合使用。

## 🔄 双轨处理系统

本项目提供两套并行的处理输出，以满足不同的使用场景。不仅适合本地原型测试，更为企业级的大规模 RAG 部署（读写分离、存储分离架构）做好了数据接口的准备。

1.  **轨道一：Markdown 切片与预览**
    *   **增强后的 Markdown (`_enhanced.md`)**：生成带分割符 (`---`) 的 Markdown 文件，包含完整的 AI 注释，非常适合人类审查、对比或用于构建文档知识门户。
2.  **轨道二：企业级父子层级分离存储 (Parent-Child Storage)**
    *   **向量数据库载荷 (`_index.jsonl`)**：极其轻量的语义挂钩（仅包含 AI 意图摘要和父级指针）。专为昂贵、要求高性能的内存计算型向量数据库（如 Milvus, Qdrant）设计，确保检索极速且不撑爆内存。
    *   **文档存储库载荷 (`_docstore.jsonl`)**：包含完整原文、Markdown 表格和 VLM 深度解析 JSON 的重型“父块”数据。专为便宜、大容量的 NoSQL 数据库（如 MongoDB, Redis 等）设计。检索时，向量库秒级命中轻量的子块摘要（Child Hook），随后系统根据指针去文档库召回这段豪华的完整上下文（Parent Context）并喂给 LLM。

### 🌊 架构工作流 (Architecture & Workflow)

```mermaid
graph TD
    A[输入解析后的 Markdown] --> B(上下文提取)
    B -->|面包屑、父级标题、相邻段落| C{VLM 增强注释}
    C -->|AI 意图摘要| D[增强版 Markdown]
    D --> E(结构感知语义分块)
    E --> F[父子块拆分]
    F -->|子块 (用于检索)| G[(向量数据库索引)]
    F -->|父块 (用于召回)| H[(文档存储库)]
```

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/Flying-Henanese/rag_enhanced_caption.git
cd rag_enhanced_caption

# 使用 uv 安装依赖
uv sync
uv sync --extra local # 可选：如果需要本地 Embedding 模型支持
```

### ⚙️ 环境配置
在项目根目录下创建一个 `.env` 文件：
```env
VLM_API_KEY=your_key_here
VLM_ENDPOINT=https://api.siliconflow.cn/v1/chat/completions
VLM_MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct
# 如果配置了结合远程 Embedding 的语义分块：
EMBEDDING_API_KEY=your_key_here
EMBEDDING_ENDPOINT=https://api.siliconflow.cn/v1/embeddings
EMBEDDING_MODEL_NAME=BAAI/bge-m3
```

### 1. 命令行工具使用 (推荐)

运行提供的 CLI 脚本，将文档通过完整的处理管线（语义分块 -> VLM 增强注释 -> 父子块拆分输出）：

```bash
# 用法：python cli.py <输入Markdown文件路径> <输出目录>
python cli.py ./docs/input.md ./output_folder
```
运行完成后，指定的输出目录会生成以下三个文件：
- `*_enhanced.md`：带 AI 完整注释增强的可读 Markdown 文件。
- `*_index.jsonl`：轻量级的向量数据库索引文件（包含摘要的向量文本）。
- `*_docstore.jsonl`：完整的文档存储文件（包含元数据及原始段落供生成时使用）。

### 2. 多模态增强注释 (独立使用)

```python
import asyncio
from rag_enhanced_caption import MarkdownMultimodalProcessor, create_default_vlm_client

async def main():
    vlm_client = create_default_vlm_client()
    processor = MarkdownMultimodalProcessor(vlm_func=vlm_client)
    
    with open("doc.md", "r", encoding="utf-8") as f:
        enriched_md = await processor.enrich_markdown(f.read(), base_dir="./")
    print(enriched_md)

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. 语义分块 (独立使用)

```python
from rag_enhanced_caption.chunker import semantic_chunk_with_metadata

chunks = semantic_chunk_with_metadata(
    markdown_content="# Header\nContent...",
    file_id="doc_001",
    filename="doc.md",
    parser_config={"chunk_token_num": 512}
)
```

### 4. 生态对接 (LangChain & LlamaIndex)

本工具输出的 JSONL 文件（`_docstore.jsonl` 和 `_index.jsonl`）结构清晰，开箱即用，可以无缝接入主流的 RAG 框架。以下是如何将生成的父块加载到您的业务流水线中的示例代码。

**对接 LlamaIndex：**
```python
import json
from llama_index.core import Document

documents = []
with open("output_folder/input_docstore.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        # 提取增强后的文本和元数据
        documents.append(Document(
            text=data.get("full_content", ""),
            metadata=data.get("metadata", {})
        ))

# 构建您的向量索引
# index = VectorStoreIndex.from_documents(documents)
```

**对接 LangChain：**
```python
import json
from langchain_core.documents import Document

documents = []
with open("output_folder/input_docstore.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        documents.append(Document(
            page_content=data.get("full_content", ""),
            metadata=data.get("metadata", {})
        ))

# 存入向量数据库
# vectorstore = Chroma.from_documents(documents, embedding_model)
```

## 🛠️ 项目结构

```text
rag_enhanced_caption/
├── cli.py                           # 命令行界面 (主入口)
├── src/
│   ├── example_orchestrator.py      # 端到端的 Parent-Child RAG 流水线演示脚本
│   └── rag_enhanced_caption/        # 顶层包目录
│       ├── chunker/                 # 结构感知的语义分块模块
│       └── enhancer/                # 视觉到文本的多模态增强模块
└── tests/                           # 统一的单元与集成测试套件
```

## 📄 开源协议
Apache-2.0 License.