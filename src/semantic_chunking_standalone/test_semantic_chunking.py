"""
测试模块：验证核心的 Markdown 语义化分块功能
"""
import pytest
import numpy as np
import os
from ragflow_like.parsers import semantic
from ragflow_like.clients import (
    get_remote_embedding_client, 
    get_local_embedding_client
)

@pytest.fixture
def remote_embed_fn():
    """
    Fixture: 初始化远程 Embedding 客户端 (如 SiliconFlow API)。
    如果在 CI 环境或本地未配置相关密钥，自动跳过依赖该 fixture 的测试。
    """
    client = get_remote_embedding_client()
    if client is None:
        pytest.skip("未配置远程 API 环境 (缺少 EMBEDDING_API_KEY 等)")
    return client

@pytest.fixture
def local_embed_fn():
    """
    Fixture: 初始化本地模型 Embedding 客户端。
    如果在未安装 sentence-transformers 的环境中运行，则自动跳过。
    """
    client = get_local_embedding_client()
    if client is None:
        pytest.skip("本地模型环境未就绪 (可能是缺少依赖)")
    return client

@pytest.fixture
def sample_markdown():
    """完全恢复的最原始长文档测试样本，包含标题级联、图片题注、JSON 块、表格等复杂元素。"""
    return """
## 痛点与挑战

在传统的RAG知识库前置处理过程中，我们面临着诸多挑战：

- **格式繁杂兼容难**：企业文档格式多样，传统工具（例如python-docx和pymupdf）难以同时高质量处理 Office 文档和复杂的 PDF 扫描件。
- **语义割裂检索差**：简单的按字符切分导致上下文丢失，检索匹配度低，无法满足业务需求。
- **解析效率瓶颈**：面对海量存量文档，缺乏高并发处理能力，响应速度慢，难以支撑大规模应用。

## 技术方案概览

为了解决上述问题，我们构建了一套高性能、可扩展的文档处理架构：

- **全格式支持**：全面覆盖 **Word、Excel、PDF、PPT和图片**等主流办公文档格式。
- **高精度解析**：通过引入MinerU VLM模型实现准确的布局解析 and 元素提取。
- **跨平台支持**：同时支持**英伟达CUDA和华为CANN框架。**
- **知识库应用优化**：为了更好支持上层应用，解析后增加了智能切分和实体识别功能，增强后续的检索效果。
- **高并发架构设计**：
    - 采用 **FastAPI** 作为服务入口，保障请求的高效接收与响应。
    - 引入 **Celery** 分布式任务队列，实现请求响应与处理过程的完全解耦，有效应对流量洪峰，保障系统稳定性和可用性。
    - **资源利用最大化**：通过创建多组 **Celery Worker**（执行CPU密集任务） + **vLLM Server**（执行GPU密集推理任务）的Pod单元，实现对多核CPU+多卡GPU的并行调用，大幅提升单机处理吞吐量。
    - 使用Docker Compose进行服务编排，不同功能划分为不同的进程，保证服务的并发量和性能。

![image.png](attachment:393b3113-298b-4063-b572-db9694dddb4b:image.png)

## 架构设计

| 服务名称 | 容器名称 | 职责描述 |
| --- | --- | --- |
| **mineru-api** | `mineru-api` | **API 网关：**基于 FastAPI，提供对外 HTTP 接口 。负责接收请求、鉴权及任务分发。 |
| **mineru-worker** | `mineru-worker` | **异步任务处理器：**基于 Celery，负责 PDF 解析、OCR、文档转换等耗时任务。 |
| **vllm** | `mineru-vllm` | **推理后端：**运行 MinerU 视觉大模型 (VLM)，提供高并发的文档理解与提取能力。 |
| **redis** | `mineru-redis` | **消息中间件：**作为 Celery 的 Broker 和 Backend，同时缓存部分运行时数据。 |

## 核心技术亮点

### 1. 多模态融合的高精度解析

针对不同类型的文档，我们采用了差异化的解析策略，以确保最佳效果：
* **Office 文档**：利用 **Docling** ，快速、精准地从文件中提取文本与格式信息。
* **复杂 PDF 文档**：引入 **MinerU 视觉大模型（VLM）**，该模型具备类似人类的视觉能力，能够深度理解文档排版，完美还原文档结构和准确提取页面上的公式和表格元素，并统一输出为标准的 Markdown 格式。

![原始文件](attachment:45b43827-6c38-49ed-9996-e08c037d6858:公式1_1原始文件.png)

原始文件

![解析后效果](attachment:90e6a6da-0ac6-4b43-bfcb-f0921960c5f1:公式1_3解析后预览.png)

解析后效果

### 2. 智能表格处理

针对文档中的表格数据，我们提供了两种灵活的转换方式，满足不同应用场景的需求：
* **键值对转换**：将表格行列关系转换为自然语义表达，便于大模型理解和问答。并且**支持合并单元格的处理**，避免生成关系错乱的表格数据。例如，将”产品名称 | 价格 | 库存”的表格转换为”产品名称：A；价格：100元，库存：50件；产品名称：B；价格：150元，库存：30件”等键值对形式。
* **Markdown表格转换**：保持表格的原始结构，转换为标准的Markdown表格格式，便于上层应用中可视化展示。

![原始表格](attachment:b02cd061-728f-4fd8-ac12-5298dd427cc3:表格1_1原始文件.png)

原始表格

![markdown格式表格](attachment:efaa0df1-96d6-4c9f-bbb4-80a561d10009:表格1_3解析后预览.png)

markdown格式表格

![key-value格式表格](attachment:a476994c-6d6b-4a74-9a55-b6907ce40bde:表格1_4切分后.png)

key-value格式表格

此外，**针对过长的表格，支持按照长度切分，在不丢失信息的前提下**适配向量数据库的存储，保证后续的检索效果。

![原始表格](attachment:71523288-f21f-413e-a69b-e9fda14532e9:表格2_3解析后预览.png)

原始表格

![表格切分后](attachment:ffa664bb-5eb8-4820-a4fb-9968266e8fd2:表格2_4切分后.png)

表格切分后

### 3. 基于语义的智能切分

摒弃了传统机械的字数切分方式，我们引入了 **BGE Embedding 模型**。通过计算文本向量并利用相似度聚类算法，将含义紧密相关的段落自动聚合。这种方式确保了每一个数据切片都是一个完整的”语义单元”，大幅提升了后续检索的准确性。与此同时，模型规模为24M，资源占用率极小，对整体解析流程的影响微乎其微。

![语义切分前1](attachment:7e4e4c3c-126b-4c5b-8dc7-e4da93252a54:语义分析1_2解析后.png)

语义切分前1

![语义切分后2](attachment:3febe837-7722-4c59-8e40-25e9bc8dbbaa:语义分析1_4切分后.png)

语义切分后2

![语义切分前2](attachment:e17278be-b749-4618-8b51-793f83892caf:语义分析2_2解析后.png)

语义切分前2

![语义切分后2](attachment:18f40c6d-6c08-40e8-b369-927b33f09816:语义分析2_3切分后.png)

语义切分后2

### 4. 上下文感知的标题聚合

针对企业长文档层级复杂的特点，我们开发了**标题聚合功能**。系统会自动将多级标题信息聚合到对应的段落中。无论文档如何切分，每个切片都能保留”父级标题-子标题”的完整路径，有效解决了碎片化导致的上下文丢失问题，从而提升了后续检索的准确性和相关性。

![多级标题聚合](attachment:7dd756ee-b097-4b2c-b7ca-638915fd7b23:语义分析2_3切分后.png)

多级标题聚合

### 5. 关键信息自动提取

为了进一步提升检索效率，我们集成了中英文的 **NER（命名实体识别）模型**。在解析过程中，系统会自动抽取文档中的组织机构、人名、专有名词等关键实体，为文档打上”智能标签”，实现多维度的精准检索。

![image.png](attachment:62685770-d067-4df5-9415-a559b0e20658:image.png)

![image.png](attachment:4d3847f8-317d-4393-8e7c-bddb5f3c50f7:image.png)

### 6. 内容检索

![原始PDF页面](attachment:4ad6a991-7d23-494d-a653-95ec8ce24f75:image.png)

原始PDF页面

![搜索关键字“叶片描述”](attachment:cc341750-9fb0-4863-b758-d77194a783c1:image.png)

搜索关键字“叶片描述”

返回结果如下：

```json
{
    "status": "success",
    "message": "搜索成功",
    "data": {
        "result": [
            {
                "page_idx": 14,
                "bbox": [70, 586, 525, 614]
            }
        ]
    }
}
```

应用展示

![c17614a4ad447a787d3a1131825a2d22.png](attachment:0c3997f7-3107-43c9-8c90-b99b8283dbf7:c17614a4ad447a787d3a1131825a2d22.png)

### 7. 性能优化

通过将CPU密集型任务和GPU密集型任务切分为相互独立的进程，更好利用了高性能服务器的硬件资源。

| 解析样例 | 原生MinerU | 本方案 |
| --- | --- | --- |
| PDF测试1 | 79.26s | 79.661 |

## 核心组件

| 依赖名称 | 版本范围 | 功能描述 |
| --- | --- | --- |
| **fastapi** | `>=0.115.12` | **Web 框架**：提供高性能接口。 |
| **mineru** | `>=2.5.0` | **核心引擎**：提供 PDF 解析能力。 |

## 应用价值

- **提质**：将非结构化文档转化为高质量的 Markdown 数据。
- **增效**：通过自动化、并行化的处理流程提升效率。
"""

def test_semantic_chunking_with_full_sample_assertions(remote_embed_fn, sample_markdown):
    """
    针对完整原始长文档的深度逻辑断言。
    
    测试目标：
    1. 切分逻辑正常运行且生成一定数量的片段。
    2. 验证标题路径的正确级联与合并（如 '# 核心技术亮点|4. 上下文感知的标题聚合'）。
    3. 表格片段附带特有标识并保持结构的封闭。
    4. 代码块（JSON）在切分后依然完好，不破坏花括号等字符。
    5. 无相关性的段落（首部与尾部）实现了正确的语义隔离（防止过度合并）。
    6. 所有重要技术关键词未在处理过程中丢失。
    """
    parser_config = {"chunk_token_num": 500}
    chunks = semantic.chunk_markdown(
        sample_markdown,
        parser_config=parser_config,
        embed_fn=remote_embed_fn
    )

    # 1. 产出数量验证 (长文档预期切分出多个 Chunk)
    assert len(chunks) >= 8, f"切分数量不足，实际: {len(chunks)}"

    # 2. 验证深度嵌套路径聚合 (Level 2 -> Level 3)
    # 定位“标题聚合”章节，它属于“核心技术亮点”
    context_chunks = [c for c in chunks if "保留”父级标题-子标题”的完整路径" in c]
    assert context_chunks, "未找到 '上下文感知的标题聚合' 相关片段"
    header_line = context_chunks[0].split("\n")[0]
    assert "核心技术亮点" in header_line, "丢失二级标题父路径"
    assert "上下文感知的标题聚合" in header_line, "丢失三级标题自身路径"
    assert "|" in header_line, "路径分隔符缺失"

    # 3. 验证复杂表格处理
    # 查找“架构设计”中的表格内容
    arch_table_chunks = [c for c in chunks if "mineru-api" in c]
    assert arch_table_chunks, "架构设计表格丢失"
    # 验证是否带有 |Table 语义标记 (由 semantic.py 逻辑生成)
    assert any("|Table" in c.split("\n")[0] for c in arch_table_chunks), "表格片段未正确打标"
    assert "|" in arch_table_chunks[0].split("\n")[2], "表格 Markdown 结构不完整"

    # 4. 验证 JSON 代码块完整性
    json_chunks = [c for c in chunks if "bbox" in c]
    assert json_chunks, "JSON 代码块在切分中丢失"
    assert "page_idx" in json_chunks[0] and "70" in json_chunks[0], "JSON 内部字段或数值丢失"

    # 5. 验证语义隔离性 (预防聚类过度合并)
    # “痛点与挑战” 是开头的背景，不应与末尾的 “应用价值” 混在一起
    pain_chunk = [c for c in chunks if "痛点与挑战" in c][0]
    assert "应用价值" not in pain_chunk, "语义隔离失败：开头背景与结尾总结被错误合并"

    # 6. 核心技术名词留存验证
    full_text = "".join(chunks)
    for keyword in ["MinerU", "FastAPI", "Celery", "CUDA", "CANN", "Docling", "BGE Embedding"]:
        assert keyword in full_text, f"核心关键词 {keyword} 丢失"

    print(f"\n[全量长样本断言通过] 验证了 {len(chunks)} 个片段。所有路径继承、表格标记、代码块及隔离逻辑均符合预期。")

def test_remote_api_smoke_with_full_sample(remote_embed_fn, sample_markdown):
    """
    冒烟测试：验证远程 API 对全量长样本的处理能力。
    简单确认是否能无错跑通且输出包含关键内容，适合排查网络、权限等基础连接问题。
    """
    chunks = semantic.chunk_markdown(sample_markdown, embed_fn=remote_embed_fn)
    assert len(chunks) > 0
    # 简单验证关键词
    assert "技术方案" in "".join(chunks)
    print(f"\n[远程 API 验证成功] 成功处理了全量长样本。")

def test_demonstrate_semantic_chunking(remote_embed_fn, sample_markdown):
    """
    演示语义分块功能的执行结果并打印在标准输出中。
    在需要查看最终形态以进行人工校验时，通常使用 `pytest -s` 命令观察本用例的打印结果。
    """
    parser_config = {"chunk_token_num": 500}
    chunks = semantic.chunk_markdown(
        sample_markdown,
        parser_config=parser_config,
        embed_fn=remote_embed_fn
    )
    print(f"\n[语义分块演示] 成功切分了 {len(chunks)} 个片段。")
    # 可以解除注释打印查看，配合 pytest -s
    # print(chunks)
