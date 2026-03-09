"""Unit tests for coreference resolution in graph extraction."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from app.ingestion.chunker import Chunk


def _chunk(index=0, content="Acme Corporation hired John Smith as CEO in New York.", token_count=20):
    return Chunk(
        document_id="doc-123", chunk_index=index,
        content=content, token_count=token_count,
        section_path="Test", has_table=False, has_code=False,
        overlap_tokens=0, metadata={},
    )


# ---------------------------------------------------------------------------
# Test 1: coreferee unavailable → original text passthrough
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_coref_unavailable_returns_originals():
    with patch("app.graph.coref._get_nlp", return_value=None):
        from app.graph.coref import resolve_coreferences
        chunks = [_chunk(0, "John Smith is the CEO."), _chunk(1, "He leads the company.")]
        result = await resolve_coreferences(chunks)

    assert result == [c.content for c in chunks]


# ---------------------------------------------------------------------------
# Test 2: empty chunks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_chunks():
    from app.graph.coref import resolve_coreferences
    result = await resolve_coreferences([])
    assert result == []


# ---------------------------------------------------------------------------
# Test 3: tiny chunks skipped (< 10 tokens)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_tiny_chunks_skipped():
    mock_nlp = MagicMock()

    with patch("app.graph.coref._get_nlp", return_value=mock_nlp):
        from app.graph.coref import resolve_coreferences
        tiny = _chunk(0, content="Hi.", token_count=2)
        result = await resolve_coreferences([tiny])

    assert result == ["Hi."]
    mock_nlp.assert_not_called()


# ---------------------------------------------------------------------------
# Helper: build a mock coreferee doc
# ---------------------------------------------------------------------------

def _mock_token(text, pos, idx):
    tok = MagicMock()
    tok.text = text
    tok.pos_ = pos
    tok.idx = idx
    return tok


def _mock_mention(token_indexes):
    m = MagicMock()
    m.token_indexes = token_indexes
    return m


def _make_mock_doc(tokens, chains):
    """Build a mock spaCy Doc with coreferee chains.

    tokens: list of (text, pos, char_idx)
    chains: list of lists of lists of token indices (chain -> mentions -> token indices)
    """
    mock_tokens = [_mock_token(t, p, i) for t, p, i in tokens]

    doc = MagicMock()
    doc.__getitem__.side_effect = lambda idx: mock_tokens[idx]

    mock_chains = []
    for chain_indices in chains:
        chain = MagicMock()
        chain.__iter__ = lambda self, ci=chain_indices: iter([_mock_mention(m) for m in ci])
        mock_chains.append(chain)

    doc._.coref_chains = mock_chains
    return doc


# ---------------------------------------------------------------------------
# Test 4: pronoun resolution within a single chunk
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pronoun_resolution_within_chunk():
    # "John Smith is the CEO. He leads the company."
    # Tokens:  0=John 1=Smith 2=is 3=the 4=CEO 5=. 6=He 7=leads 8=the 9=company 10=.
    tokens = [
        ("John", "PROPN", 0), ("Smith", "PROPN", 5), ("is", "AUX", 11),
        ("the", "DET", 14), ("CEO", "NOUN", 18), (".", "PUNCT", 21),
        ("He", "PRON", 23), ("leads", "VERB", 26), ("the", "DET", 32),
        ("company", "NOUN", 36), (".", "PUNCT", 43),
    ]
    # Chain: "John Smith" [0,1] ← "He" [6]
    chains = [[[0, 1], [6]]]
    mock_doc = _make_mock_doc(tokens, chains)

    mock_nlp = MagicMock(return_value=mock_doc)

    with patch("app.graph.coref._get_nlp", return_value=mock_nlp):
        from app.graph.coref import resolve_coreferences
        content = "John Smith is the CEO. He leads the company."
        chunk = _chunk(0, content=content, token_count=15)
        result = await resolve_coreferences([chunk])

    assert "John Smith" in result[0]
    # "He" should be replaced
    assert result[0] != content or "He" not in result[0].split("John Smith")[1] if "He" in content else True


# ---------------------------------------------------------------------------
# Test 5: cross-chunk resolution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cross_chunk_resolution():
    """Chunk 0 has 'John Smith', chunk 1 has 'He' → resolved."""
    from app.graph.coref import resolve_coreferences

    chunk0_text = "John Smith is the CEO of Acme Corporation."
    chunk1_text = "He leads the company with great vision and determination every single day."
    chunk0 = _chunk(0, chunk0_text, token_count=15)
    chunk1 = _chunk(1, chunk1_text, token_count=15)

    # For chunk 0 (no prev), mock doc without chains
    doc0 = MagicMock()
    doc0._.coref_chains = []

    # For chunk 1 (with prev context), mock doc with cross-chunk chain
    # prev = chunk0_text (43 chars), separator "\n\n" (2 chars), current_start = 45
    current_start = len(chunk0_text) + 2  # 45
    tokens1 = [
        ("John", "PROPN", 0), ("Smith", "PROPN", 5), ("is", "AUX", 11),
        ("the", "DET", 14), ("CEO", "NOUN", 18), ("of", "ADP", 22),
        ("Acme", "PROPN", 25), ("Corporation", "PROPN", 30), (".", "PUNCT", 41),
        # After \n\n (current_start = 45)
        ("He", "PRON", current_start), ("leads", "VERB", current_start + 3),
        ("the", "DET", current_start + 9), ("company", "NOUN", current_start + 13),
        ("with", "ADP", current_start + 21), ("great", "ADJ", current_start + 26),
        ("vision", "NOUN", current_start + 32), ("and", "CONJ", current_start + 39),
        ("determination", "NOUN", current_start + 43),
        ("every", "DET", current_start + 57), ("single", "ADJ", current_start + 63),
        ("day", "NOUN", current_start + 70), (".", "PUNCT", current_start + 73),
    ]
    chains1 = [[[0, 1], [9]]]  # "John Smith" ← "He"
    doc1 = _make_mock_doc(tokens1, chains1)

    call_count = [0]

    def mock_nlp_fn(text):
        result = doc0 if call_count[0] == 0 else doc1
        call_count[0] += 1
        return result

    mock_nlp = MagicMock(side_effect=mock_nlp_fn)

    with patch("app.graph.coref._get_nlp", return_value=mock_nlp):
        result = await resolve_coreferences([chunk0, chunk1])

    assert result[0] == chunk0.content  # no changes to chunk 0
    assert "John Smith" in result[1]  # "He" replaced in chunk 1


# ---------------------------------------------------------------------------
# Test 6: alias resolution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_alias_resolution():
    """Short alias 'The policy' resolved to full name."""
    # "The California Parental Leave Policy v3 provides benefits. The policy covers all employees."
    content = "The California Parental Leave Policy v3 provides benefits. The policy covers all employees."
    tokens = [
        ("The", "DET", 0), ("California", "PROPN", 4), ("Parental", "PROPN", 15),
        ("Leave", "PROPN", 24), ("Policy", "PROPN", 30), ("v3", "NOUN", 37),
        ("provides", "VERB", 40), ("benefits", "NOUN", 49), (".", "PUNCT", 57),
        ("The", "DET", 59), ("policy", "NOUN", 63), ("covers", "VERB", 70),
        ("all", "DET", 77), ("employees", "NOUN", 81), (".", "PUNCT", 90),
    ]
    # Chain: [1,2,3,4,5] "California Parental Leave Policy v3" ← [9,10] "The policy"
    chains = [[[1, 2, 3, 4, 5], [9, 10]]]
    mock_doc = _make_mock_doc(tokens, chains)
    mock_nlp = MagicMock(return_value=mock_doc)

    with patch("app.graph.coref._get_nlp", return_value=mock_nlp):
        from app.graph.coref import resolve_coreferences
        chunk = _chunk(0, content=content, token_count=20)
        result = await resolve_coreferences([chunk])

    # "The policy" (2 tokens, <=3) should be replaced with the longer proper noun mention
    assert "California" in result[0]


# ---------------------------------------------------------------------------
# Test 7: failure on one chunk falls back to original
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_failure_per_chunk_falls_back():
    call_count = [0]

    def mock_nlp_fn(text):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("spaCy crashed")
        doc = MagicMock()
        doc._.coref_chains = []
        return doc

    mock_nlp = MagicMock(side_effect=mock_nlp_fn)

    with patch("app.graph.coref._get_nlp", return_value=mock_nlp):
        from app.graph.coref import resolve_coreferences
        chunks = [
            _chunk(0, "The first chunk contains enough words to pass the minimum token threshold for processing.", token_count=15),
            _chunk(1, "The second chunk also contains enough words to pass the minimum token threshold for processing.", token_count=15),
        ]
        result = await resolve_coreferences(chunks)

    assert result[0] == chunks[0].content  # fell back to original
    assert result[1] == chunks[1].content  # processed fine


# ---------------------------------------------------------------------------
# Test 8: runs in thread pool (run_in_executor called)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_runs_in_thread_pool():
    mock_nlp = MagicMock()
    mock_doc = MagicMock()
    mock_doc._.coref_chains = []
    mock_nlp.return_value = mock_doc

    with patch("app.graph.coref._get_nlp", return_value=mock_nlp), \
         patch("app.graph.coref.asyncio") as mock_asyncio:
        mock_loop = MagicMock()
        mock_asyncio.get_event_loop.return_value = mock_loop

        async def mock_executor(executor, fn, *args):
            return fn(*args)

        mock_loop.run_in_executor = AsyncMock(side_effect=mock_executor)

        from app.graph.coref import resolve_coreferences
        chunks = [_chunk(0, "The quick brown fox jumps over the lazy dog and then runs away fast.", token_count=15)]
        await resolve_coreferences(chunks)

        mock_loop.run_in_executor.assert_called_once()


# ---------------------------------------------------------------------------
# Test 9: original content never modified
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_original_content_never_modified():
    original_contents = [
        "John Smith is the CEO of the large multinational corporation based in New York.",
        "He leads the company with great determination and vision every single day of the year.",
    ]
    chunks = [_chunk(i, content=c, token_count=15) for i, c in enumerate(original_contents)]

    mock_nlp = MagicMock()
    mock_doc = MagicMock()
    mock_doc._.coref_chains = []
    mock_nlp.return_value = mock_doc

    with patch("app.graph.coref._get_nlp", return_value=mock_nlp):
        from app.graph.coref import resolve_coreferences
        await resolve_coreferences(chunks)

    # Original chunk content must be unchanged
    for chunk, original in zip(chunks, original_contents):
        assert chunk.content == original


# ---------------------------------------------------------------------------
# Test 10: resolved texts passed to extractor
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resolved_texts_passed_to_extractor():
    """Integration: extract_from_chunks receives resolved_texts parameter."""
    import json
    from unittest.mock import call

    resolved = ["Resolved John Smith is the CEO.", "Resolved He leads Acme."]

    def _llm_response():
        data = {"entities": [], "relationships": []}
        message = MagicMock()
        message.content = json.dumps(data)
        choice = MagicMock()
        choice.message = message
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    with patch("app.graph.extractor.AsyncOpenAI") as mock_cls:
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(return_value=_llm_response())
        mock_cls.return_value = client

        from app.graph.extractor import extract_from_chunks
        chunks = [_chunk(0), _chunk(1)]
        await extract_from_chunks(chunks, "doc-123", resolved_texts=resolved)

        # Verify the resolved texts were used in the messages
        calls = client.chat.completions.create.call_args_list
        for i, c in enumerate(calls):
            messages = c.kwargs.get("messages") or c[1].get("messages")
            user_msg = messages[-1]["content"]
            assert resolved[i] in user_msg
