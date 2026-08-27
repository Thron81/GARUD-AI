# Garud AI-1.0 — Usage Notes

Garud AI-1.0 is a local terminal and web copilot built on Qwen 2.5.
It supports four modes: chat, tools, json, and rag.

## Chat mode
Plain conversation. This is the default mode and requires no setup.

## Tools mode
The model can call three tools: calculator, file_search, and read_file.
Enable it with /mode tools in the terminal client, or select "tools" in
the Streamlit sidebar.

## JSON mode
Forces the model to reply with a single valid JSON object. Useful for
structured extraction tasks.

## RAG mode
Answers are grounded in documents you ingest with rag/ingest.py. Build
the index first, then select "rag" mode in the Streamlit sidebar to ask
questions about your own documents with source citations.

## Resetting a conversation
Use /reset in the terminal client, or the "Reset conversation" button
in the Streamlit sidebar. Resetting clears the message history but keeps
your selected mode and model settings.
