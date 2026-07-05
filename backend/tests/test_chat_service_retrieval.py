"""Characterization tests for chat retrieval helpers.

These lock in the data contracts used by find_similar_chunks's query-
decomposition and query-expansion branches, which previously fed into a
non-existent `_perform_semantic_search` method (fixed in the same change).
The full find_similar_chunks() requires a live pgvector DB and is therefore
exercised via integration tests; here we cover the pure fusion/dedup logic
that the (now inline-SQL) sub-query search feeds into.
"""
import pytest

from app.services.chat_service import ChatService


@pytest.fixture
def svc() -> ChatService:
    return ChatService()


def _chunk(cid, similarity, content="c", page=1):
    return {
        "id": cid,
        "content": content,
        "pageNumber": page,
        "documentId": "doc",
        "positionData": {},
        "chunk_type": "text",
        "similarity": similarity,
    }


def test_combine_results_with_rrf_fuses_multiple_rankings(svc):
    """RRF must merge two ranked lists, scoring by reciprocal rank.

    The query-decomposition branch (chat_service.py ~L700) calls this with
    one ranked list per sub-query; this test pins the fusion contract that
    the previously-broken _perform_semantic_search path now feeds.
    """
    # Two sub-query result sets, each ranked by similarity.
    set_a = [_chunk("c1", 0.9), _chunk("c2", 0.7), _chunk("c3", 0.5)]
    set_b = [_chunk("c2", 0.95), _chunk("c1", 0.8), _chunk("c4", 0.4)]

    combined = svc._combine_results_with_rrf([set_a, set_b], rrf_k=60)

    # All 4 unique chunk ids appear
    ids = [c["id"] for c in combined]
    assert set(ids) == {"c1", "c2", "c3", "c4"}
    # c1 and c2 appear in both rankings, so they outrank c3/c4 (single ranking)
    assert ids.index("c1") < ids.index("c3")
    assert ids.index("c2") < ids.index("c4")
    # RRF score is stored on each chunk
    assert all("_rrf_score" in c for c in combined)


def test_combine_results_with_rrf_empty_inputs(svc):
    assert svc._combine_results_with_rrf([], rrf_k=60) == []
    assert svc._combine_results_with_rrf([[], []], rrf_k=60) == []


def test_combine_results_with_rrf_single_set_passes_through(svc):
    """One result set is returned directly (no fusion needed)."""
    single = [_chunk("c1", 0.9), _chunk("c2", 0.5)]
    combined = svc._combine_results_with_rrf([single], rrf_k=60)
    assert [c["id"] for c in combined] == ["c1", "c2"]


def test_merge_chunks_and_sentences_dedupes_by_content(svc):
    """Sentence results merge with chunk results, deduping on content prefix."""
    chunks = [_chunk("c1", 0.8, content="The mitochondrion produces ATP via respiration.")]
    sentences = [
        {
            **_chunk("s1", 0.7, content="The mitochondrion produces ATP via respiration."),
            "is_sentence": True,
        },
        {**_chunk("s2", 0.6, content="Glycolysis splits glucose into pyruvate."), "is_sentence": True},
    ]

    merged = svc._merge_chunks_and_sentences(chunks, sentences, boost_factor=1.2)

    # The sentence overlapping the chunk content is deduped (kept once),
    # the unique sentence is added.
    contents = [m["content"] for m in merged]
    assert len(merged) == 2  # c1/s1 collapsed, s2 kept
    assert any("produces ATP" in c for c in contents)
    assert any("Glycolysis" in c for c in contents)
    # Sorted by similarity descending
    assert merged[0]["similarity"] >= merged[-1]["similarity"]


def test_calculate_adaptive_weights_returns_two_values(svc):
    """Adaptive weights always returns (semantic, keyword) in 0..1-ish range."""
    for query in ["definition of photosynthesis", "explain how ATP is produced", "code"]:
        sem, kw = svc._calculate_adaptive_weights(query)
        assert isinstance(sem, (int, float))
        assert isinstance(kw, (int, float))
        assert 0.0 <= sem <= 1.5  # boost values can push above 1.0
        assert 0.0 <= kw <= 1.5
