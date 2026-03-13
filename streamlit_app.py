"""Streamlit chat UI for testing the RAG retrieval pipeline end-to-end."""

import asyncio
import threading

import streamlit as st

from app.retrieval.models import QueryResult

# ── Persistent event loop (survives Streamlit reruns) ────────────────────────


@st.cache_resource
def _get_loop() -> asyncio.AbstractEventLoop:
    """Create a background event loop that stays alive across reruns.

    asyncio.run() would create+destroy a loop each time, killing asyncpg
    connection pools and orphaning fire-and-forget tasks (cache, logging).
    """
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return loop


def run_async(coro):
    """Submit a coroutine to the persistent loop and block until done."""
    loop = _get_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="RAG Chat", page_icon="📚", layout="wide")
st.title("RAG Chat")
st.caption("Agentic + Graph RAG retrieval over the RERA Act")

# ── Session state ────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Metadata expanders ───────────────────────────────────────────────────────

def _render_metadata(result: QueryResult) -> None:
    """Render expandable sections for query metadata."""

    # Query info
    with st.expander("Query Info"):
        cols = st.columns(3)
        cols[0].metric("Query Type", result.query_type.value)
        cols[1].metric("Cached", "Yes" if result.cached else "No")
        total = result.step_timings.get("total", "—")
        cols[2].metric("Total Time", f"{total}s" if isinstance(total, (int, float)) else total)

        if result.step_timings:
            st.json({k: f"{v:.3f}s" if isinstance(v, float) else v for k, v in result.step_timings.items()})

        if result.error:
            st.error(result.error)

    # Chunks used
    if result.chunks_used:
        with st.expander(f"Chunks Used ({len(result.chunks_used)})"):
            for i, chunk in enumerate(result.chunks_used, 1):
                st.markdown(f"**Chunk {i}** — score `{chunk.score:.4f}` · source `{chunk.source}` · section `{chunk.section_path}`")
                st.text(chunk.content[:500] + ("…" if len(chunk.content) > 500 else ""))
                st.divider()

    # Graph paths
    if result.graph_paths:
        with st.expander(f"Graph Paths ({len(result.graph_paths)})"):
            for i, path in enumerate(result.graph_paths, 1):
                # Interleave entities and relationships: E -[R]-> E -[R]-> E
                parts = []
                for j, entity in enumerate(path.entities):
                    parts.append(f"**{entity}**")
                    if j < len(path.relationships):
                        parts.append(f" —[{path.relationships[j]}]→ ")
                st.markdown(f"Path {i} (confidence {path.confidence:.2f}): {''.join(parts)}")

    # Conflicts
    if result.conflicts:
        with st.expander(f"Conflicts ({len(result.conflicts)})"):
            for c in result.conflicts:
                st.warning(f"**Claim A:** {c.claim_a}\n\n**Claim B:** {c.claim_b}")
                st.info(f"**Resolution:** {c.resolution}\n\n**Reason:** {c.reason}")
                st.divider()


# ── Render chat history ─────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("metadata"):
            _render_metadata(msg["metadata"])

# ── Chat input ───────────────────────────────────────────────────────────────

if user_input := st.chat_input("Ask a question about the RERA Act…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run retrieval on persistent loop
    with st.chat_message("assistant"):
        with st.spinner("Retrieving…"):
            from app.retrieval import query
            result: QueryResult = run_async(query(user_input))

        # Display answer
        st.markdown(result.answer)
        _render_metadata(result)

    # Persist to session
    st.session_state.messages.append({
        "role": "assistant",
        "content": result.answer,
        "metadata": result,
    })
