from rag_enhanced_caption.lexical_search.builder import build_searchable_objects


def test_builds_searchable_objects_from_markdown_table() -> None:
    chunk = {
        "id": "doc_chunk_7",
        "content": (
            "| 模型 | 发布时间 | 联系邮箱 |\n"
            "| --- | --- | --- |\n"
            "| MiniRAG | 2026-06-19 | owner@example.com |\n"
            "| LightRAG | 2025-12-01 | team@example.com |"
        ),
        "metadata": {"element_type": "Table"},
    }

    objects = build_searchable_objects(chunk)

    headers = [obj for obj in objects if obj.field_type == "table_header"]
    rows = [obj for obj in objects if obj.field_type == "table_row"]
    assert [obj.searchable_text for obj in headers] == ["模型 发布时间 联系邮箱"]
    assert len(rows) == 2
    assert "模型：MiniRAG" in rows[0].searchable_text
    assert "发布时间：2026-06-19" in rows[0].searchable_text
    assert all(obj.owner_node_id == "doc_chunk_7" for obj in objects)


def test_extracts_fixed_format_values_without_llm() -> None:
    chunk = {
        "id": "doc_chunk_8",
        "content": "发布日期为 2026-06-19，联系人是 owner@example.com。",
        "metadata": {"element_type": "text"},
    }

    objects = build_searchable_objects(chunk)

    assert {(obj.field_type, obj.searchable_text) for obj in objects} == {
        ("date", "2026-06-19"),
        ("email", "owner@example.com"),
    }


def test_searchable_object_ids_are_stable_and_duplicates_are_removed() -> None:
    chunk = {
        "id": "doc_chunk_9",
        "content": "owner@example.com 与 owner@example.com",
        "metadata": {"element_type": "text"},
    }

    first = build_searchable_objects(chunk)
    second = build_searchable_objects(chunk)

    assert len(first) == 1
    assert first == second
