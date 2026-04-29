"""
基于向量嵌入的语义句子切分核心工具。

当遇到没有任何 Markdown 结构的极长纯文本段落时，使用本模块进行处理。
工作原理：
1. 通过标点符号或 NLTK 规则将长文本拆分为句子。
2. 使用 Embedding 模型将句子转为向量。
3. 利用聚类算法（层次聚类）将语义上连贯的句子自动组合在一起。
"""
from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import nltk
from nltk.tokenize import sent_tokenize
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# 全局标记位，防止重复检查 NLTK 依赖
_punkt_checked = False


def _ensure_punkt_tab() -> None:
    """
    首次使用分句能力时检查 NLTK punkt_tab 资源是否就绪。
    用于英文/混合文本的精确断句。
    """
    global _punkt_checked
    if _punkt_checked:
        return

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError as e:
        raise RuntimeError(
            "缺少 NLTK 资源 punkt_tab。请先执行: python -m nltk.downloader punkt_tab"
        ) from e

    _punkt_checked = True


def split_sentences_chinese(text: str) -> list[str]:
    """
    使用正则表达式将中文文本分割成句子。
    匹配常见的中文断句标点，并考虑右引号/单引号等闭合情况，避免把引号内的完整句子错误拆断。
    """
    # 匹配规则：匹配。！？且后面跟着右引号的情况，或者匹配。！？且后面不跟着右引号的情况
    pattern = r'(?<=[。！？][”’"])|(?<=[。！？])(?![”’"])'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def split_mixed_sentences(text: str) -> list[str]:
    """
    处理中英文混合文本的分句逻辑。
    优先用段落换行粗切，再判断段内字符特征分别走 NLTK (英文) 或正则 (中文) 的分句处理。
    """
    _ensure_punkt_tab()

    chunks = re.split(r"(\n+)", text)
    sentences = []

    for ch in chunks:
        if not ch.strip():
            continue
        # 如果包含英文字母，使用 NLTK 英文分句引擎
        if re.search(r"[A-Za-z]", ch):
            parts = sent_tokenize(ch)
            sentences.extend([p.strip() for p in parts if p.strip()])
        else:
            # 纯中文分支
            sents = split_sentences_chinese(ch)
            if sents:
                sentences.extend([s.strip() for s in sents if s.strip()])
            else:
                # 极端保底（没有匹配到标准断句标点，但是句子长，尝试任意匹配结束符）
                parts = re.split(r"(?<=[。！？])", ch)
                sentences.extend([p.strip() for p in parts if p.strip()])
    return sentences


def find_best_num_clusters(embeddings: Any, min_clusters: int = 2, max_clusters: int = 10) -> int:
    """
    (废弃 / 实验性策略) 使用轮廓系数 (Silhouette Score) 选择最佳聚类数量。
    
    轮廓系数评估的是聚类的紧凑度和分离度，理论上能找到纯粹语义上的"最佳断点"。
    但在 RAG 场景中，该策略容易生成极其长或极其短的不均衡块，因此实际多采用固定大小预期来推导 K 值。
    """
    if len(embeddings) <= min_clusters:
        return len(embeddings)

    best_score = -1
    best_k = min_clusters

    limit_k = min(max_clusters, len(embeddings))
    for k in range(min_clusters, limit_k):
        # 注意：k 必须小于样本数，采用余弦相似度进行层次聚类
        labels = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(embeddings)
        if len(set(labels)) <= 1:
            continue
        score = silhouette_score(embeddings, labels, metric="cosine")
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def semantic_chunking_with_auto_clusters(
    text: str, 
    embed_fn: Callable[[list[str]], Any] | None, 
    token_count_fn: Callable[[str], int], 
    max_chunk_size: int = 512,
    strategy: str = "auto"
) -> list[str]:
    """
    对传入的无结构长文本进行语义切分。

    Args:
        text: 待切分的纯文本。
        embed_fn: 获取向量表示的函数客户端。
        token_count_fn: 长度估算函数。
        max_chunk_size: 分块长度上限限制。
        strategy: 
            "auto" - (默认推荐) 根据长度预计算聚类 K 值，工程表现最优，保证长度受控均匀。
            "silhouette" - 使用轮廓系数寻找数学上的最佳语义界限，不强制物理长度均衡。
            
    Returns:
        聚类合并后，依然符合长度阈值的句子组合列表。
    """
    # 1. 拆分成极细的句子颗粒度
    sentences = split_mixed_sentences(text)
    if len(sentences) < 2:
        return [text.strip()]

    sentence_token_counts = [token_count_fn(s) for s in sentences]
    total_tokens = sum(sentence_token_counts)

    # 简单退化路径：如果未提供向量函数，或者按照 "auto" 策略算下来总长度本来就没超标
    if embed_fn is None or (strategy == "auto" and total_tokens <= max_chunk_size):
        chunks = []
        current_chunk = ""
        current_chunk_tokens = 0
        for s, cnt in zip(sentences, sentence_token_counts):
            if current_chunk_tokens + cnt > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = s
                current_chunk_tokens = cnt
            else:
                current_chunk += s
                current_chunk_tokens += cnt
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    # 2. 调用向量模型拿到所有句子的表示
    embeddings = embed_fn(sentences)

    # 3. 确定要聚成多少簇 (K)
    if strategy == "silhouette":
        # 使用轮廓系数寻找最佳聚类数，范围根据句子数动态调整
        best_k = find_best_num_clusters(embeddings, min_clusters=2, max_clusters=min(15, len(sentences)))
    else:
        # 默认 auto 策略：基于目标长度上限，直接计算最理想的切分数
        # 假设均匀分布，需要切的块数 = 总长度 / 最大长度（向上取整）
        best_k = (total_tokens + max_chunk_size - 1) // max_chunk_size
        best_k = min(best_k, len(sentences))

    # 4. 执行凝聚层次聚类算法 (将语义相似且相邻的句子标记为同一类)
    labels = AgglomerativeClustering(n_clusters=best_k, metric="cosine", linkage="average").fit_predict(embeddings)

    chunks = []
    current_chunk = ""
    current_chunk_tokens = 0
    current_label = labels[0]

    # 5. 根据聚类标签组装最终的分块
    for sentence, label, token_count in zip(sentences, labels, sentence_token_counts):
        # 发生切断的条件：聚类标签发生变化，或者长度达到了硬性截断上限
        if label != current_label or (strategy == "auto" and current_chunk_tokens + token_count > max_chunk_size):
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_chunk_tokens = token_count
            current_label = label
        else:
            current_chunk += sentence
            current_chunk_tokens += token_count

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks
