from __future__ import annotations

import argparse

from core.engine import GarudEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Garud AI locally in a terminal.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="Hugging Face model ID or local model directory.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load with bitsandbytes NF4 4-bit quantization (CUDA only).")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = GarudEngine(
        model_name=args.model,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    engine.load(progress_callback=print)
    print("Garud AI-1.0 ready. Type /help for commands.")

    mode = "chat"
    messages = [{"role": "system", "content": GarudEngine.system_prompt_for(mode)}]

    def reset_messages() -> list[dict[str, str]]:
        return [{"role": "system", "content": GarudEngine.system_prompt_for(mode)}]

    while True:
        try:
            user_text = input(f"\n[{mode}] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_text:
            continue
        low = user_text.lower()
        if low in {"/exit", "/quit"}:
            print("Goodbye.")
            break
        if low == "/reset":
            messages = reset_messages()
            print("Conversation reset.")
            continue
        if low == "/help":
            print(
                "Commands:\n"
                "  /reset        clear history\n"
                "  /mode chat    plain conversation (default)\n"
                "  /mode tools   enable calculator / file_search / read_file tool use\n"
                "  /mode json    force structured JSON replies\n"
                "  /exit, /quit  quit\n"
                "(For RAG mode, use the Streamlit app: streamlit run streamlit_app.py)\n"
            )
            continue
        if low.startswith("/mode"):
            parts = low.split()
            if len(parts) == 2 and parts[1] in {"chat", "tools", "json"}:
                mode = parts[1]
                messages = reset_messages()
                print(f"Mode set to '{mode}' (history reset).")
            else:
                print("Usage: /mode chat|tools|json")
            continue

        messages.append({"role": "user", "content": user_text})
        messages = GarudEngine.trim_history(messages)

        try:
            if mode == "tools":
                reply, scratch, trace = engine.generate_with_tools(messages)
                messages.extend(scratch)
            else:
                reply = engine.generate_once(messages)
                if mode == "json":
                    reply = engine.try_parse_json(reply)
        except RuntimeError as error:
            messages.pop()
            print(f"\nGeneration error: {error}")
            print("Try a shorter message or lower --max-new-tokens.")
            continue

        messages.append({"role": "assistant", "content": reply})
        messages = GarudEngine.trim_history(messages)
        print(f"\nGarud: {reply}")


if __name__ == "__main__":
    main()
