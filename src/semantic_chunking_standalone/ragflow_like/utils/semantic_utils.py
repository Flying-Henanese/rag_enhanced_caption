from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import nltk
from nltk.tokenize import sent_tokenize
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

_punkt_checked = False


def _ensure_punkt_tab() -> None:
    """首次使用分句能力时检查 NLTK punkt_tab 资源。"""
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
    """使用正则表达式将中文文本分割成句子。"""
    pattern = r'(?<=[。！？][”’"])|(?<=[。！？])(?![”’"])'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def split_mixed_sentences(text: str) -> list[str]:
    """处理中英文混合文本的分句逻辑。"""
    _ensure_punkt_tab()

    chunks = re.split(r"(\n+)", text)
    sentences = []

    for ch in chunks:
        if not ch.strip():
            continue
        if re.search(r"[A-Za-z]", ch):
            parts = sent_tokenize(ch)
            sentences.extend([p.strip() for p in parts if p.strip()])
        else:
            sents = split_sentences_chinese(ch)
            if sents:
                sentences.extend([s.strip() for s in sents if s.strip()])
            else:
                parts = re.split(r"(?<=[。！？])", ch)
                sentences.extend([p.strip() for p in parts if p.strip()])
    return sentences


def find_best_num_clusters(embeddings: Any, min_clusters: int = 2, max_clusters: int = 10) -> int:
    """
    使用轮廓系数选择最佳聚类数量。
    旨在让每个分段语义集中，且分段之间界限分明。
    """
    if len(embeddings) <= min_clusters:
        return len(embeddings)

    best_score = -1
    best_k = min_clusters

    limit_k = min(max_clusters, len(embeddings))
    for k in range(min_clusters, limit_k):
        # 注意：k 必须小于样本数
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
    对传入的文本进行语义切分。

    Args:
        strategy: 
            "auto" - (默认) 根据长度预计算 K 值，工程最优，保证长度受控。
            "silhouette" - 使用轮廓系数寻找数学上的最佳语义界限，不强制物理长度。
    """
    sentences = split_mixed_sentences(text)
    if len(sentences) < 2:
        return [text.strip()]

    sentence_token_counts = [token_count_fn(s) for s in sentences]
    total_tokens = sum(sentence_token_counts)

    # 简单路径：无向量函数或未超长
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

    embeddings = embed_fn(sentences)

    if strategy == "silhouette":
        # 使用轮廓系数寻找最佳聚类数，范围根据句子数动态调整
        best_k = find_best_num_clusters(embeddings, min_clusters=2, max_clusters=min(15, len(sentences)))
    else:
        # 默认 auto 策略：基于长度预计算
        best_k = (total_tokens + max_chunk_size - 1) // max_chunk_size
        best_k = min(best_k, len(sentences))

    labels = AgglomerativeClustering(n_clusters=best_k, metric="cosine", linkage="average").fit_predict(embeddings)

    chunks = []
    current_chunk = ""
    current_chunk_tokens = 0
    current_label = labels[0]

    for sentence, label, token_count in zip(sentences, labels, sentence_token_counts):
        # 即使在 silhouette 策略下，如果单块由于语义太聚拢导致超过 2 倍 max_chunk_size，
        # 我们依然保留长度截断保护，防止下游处理崩溃。
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
