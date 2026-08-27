"""Garud AI-1.0 — Streamlit front end.

Wraps core/engine.py (chat, tool-use, JSON mode) and rag/retriever.py
(retrieval-augmented generation) in a single demoable web UI.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os

import streamlit as st

from core.engine import DEFAULT_MODEL, GarudEngine
from rag.retriever import Retriever

st.set_page_config(page_title="Garud AI-1.0", page_icon="🦅", layout="wide")


# --- Cached resources ------------------------------------------------------
# st.cache_resource keeps the model loaded across reruns/user interactions
# instead of reloading it on every Streamlit script rerun.

@st.cache_resource(show_spinner=False)
def get_engine(model_name: str, load_in_4bit: bool, max_new_tokens: int, temperature: float) -> GarudEngine:
    engine = GarudEngine(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    engine.load()
    return engine


@st.cache_resource(show_spinner=False)
def get_retriever(index_dir: str) -> Retriever | None:
    try:
        return Retriever(index_dir)
    except FileNotFoundError:
        return None


# --- Sidebar: configuration -------------------------------------------------

with st.sidebar:
    st.title("🦅 Garud AI-1.0")
    st.caption("Local LLM copilot — chat, tools, JSON, and RAG.")

    model_name = st.text_input("Model", value=DEFAULT_MODEL)
    load_in_4bit = st.checkbox("Load in 4-bit (CUDA only)", value=False)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    max_new_tokens = st.slider("Max new tokens", 64, 1024, 512, 32)

    mode = st.radio("Mode", ["chat", "tools", "json", "rag"], index=0)

    index_dir = st.text_input("RAG index directory", value="rag/index")

    st.divider()
    if st.button("🔄 Reset conversation", use_container_width=True):
        st.session_state.pop("messages", None)
        st.session_state.pop("mode_used", None)
        st.rerun()

    st.divider()
    with st.expander("ℹ️ Mode guide"):
        st.markdown(
            "- **chat** — plain conversation\n"
            "- **tools** — model can call calculator / file_search / read_file\n"
            "- **json** — forces structured JSON replies\n"
            "- **rag** — answers grounded in your ingested documents "
            "(build an index first with `py rag/ingest.py`)"
        )


# --- Load engine (spinner shown on first load / config change) -------------

with st.spinner(f"Loading {model_name}..."):
    engine = get_engine(model_name, load_in_4bit, max_new_tokens, temperature)
    # Config sliders can change without reloading the model; keep the live
    # engine's generation params in sync with the sidebar on every rerun.
    engine.max_new_tokens = max_new_tokens
    engine.temperature = temperature

retriever = get_retriever(index_dir) if mode == "rag" else None
if mode == "rag" and retriever is None:
    st.warning(
        f"No RAG index found at '{index_dir}'. Run `py rag/ingest.py --source rag/data --index {index_dir}` "
        "first, then reload this page."
    )


# --- Session state -----------------------------------------------------------

if "messages" not in st.session_state or st.session_state.get("mode_used") != mode:
    st.session_state.messages = [{"role": "system", "content": GarudEngine.system_prompt_for(mode)}]
    st.session_state.mode_used = mode

st.title("Garud AI-1.0")
st.caption(f"Mode: **{mode}** · Model: `{model_name}` · Device: `{engine.device}`")


# --- Render chat history (skip the system message) --------------------------

for message in st.session_state.messages[1:]:
    if message["role"] in ("user", "assistant"):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# --- Chat input --------------------------------------------------------------

user_input = st.chat_input("Message Garud AI...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            messages = GarudEngine.trim_history(st.session_state.messages)

            if mode == "tools":
                reply, scratch, trace = engine.generate_with_tools(messages)
                st.session_state.messages.extend(scratch)
                if trace:
                    with st.expander(f"🔧 {len(trace)} tool call(s)"):
                        for name, arg, observation in trace:
                            st.code(f"{name}({arg}) -> {observation}", language="text")

            elif mode == "rag":
                if retriever is None:
                    reply = "No RAG index is loaded — build one with rag/ingest.py first."
                else:
                    chunks = retriever.retrieve(user_input, top_k=4)
                    context = Retriever.format_context(chunks)
                    rag_system = {"role": "system", "content": GarudEngine.system_prompt_for("rag", context)}
                    reply = engine.generate_once([rag_system] + messages[1:])
                    if chunks:
                        with st.expander(f"📚 {len(chunks)} retrieved source(s)"):
                            for chunk in chunks:
                                st.markdown(f"**{chunk.source}** · score {chunk.score:.3f}")
                                st.text(chunk.text[:400])

            else:
                reply = engine.generate_once(messages)
                if mode == "json":
                    reply = engine.try_parse_json(reply)

        if mode == "json":
            st.code(reply, language="json")
        else:
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state.messages = GarudEngine.trim_history(st.session_state.messages)
