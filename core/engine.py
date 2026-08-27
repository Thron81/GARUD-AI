"""Core Garud AI engine: model loading, generation, tool-use loop, JSON mode.

This module has no I/O (no input()/print()) so it can be imported cleanly
by both the terminal client (garud_ai.py) and the Streamlit app
(streamlit_app.py) without duplicating logic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.tools import TOOL_DESCRIPTIONS, run_tool

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

BASE_SYSTEM_PROMPT = (
    "You are Garud AI-1.0, a helpful AI teammate. "
    "Give clear, practical answers for coding help, situational conversation, "
    "and idea generation. Be honest when you are uncertain."
)

TOOL_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + (
    "\n\nYou have access to these tools:\n"
    f"{TOOL_DESCRIPTIONS}\n"
    "When you need a tool, respond with ONLY a single line in this exact "
    "format and nothing else:\n"
    "TOOL: tool_name(argument)\n\n"
    "You will then be given a line starting with 'OBSERVATION:' containing "
    "the tool's result. Use it to continue reasoning. When you have enough "
    "information, respond normally in plain text with your final answer — "
    "do not use the TOOL: format for your final answer."
)

JSON_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + (
    "\n\nRespond ONLY with a single valid JSON object and nothing else "
    "(no markdown fences, no commentary). Choose reasonable keys that fit "
    "the user's request (for example: {\"answer\": ...} or task-specific "
    "fields)."
)

RAG_SYSTEM_PROMPT_TEMPLATE = BASE_SYSTEM_PROMPT + (
    "\n\nUse the following retrieved context to answer the user's question. "
    "Cite the source file for any fact you use, like [source: filename]. "
    "If the context does not contain the answer, say so honestly instead "
    "of guessing.\n\nCONTEXT:\n{context}"
)

TOOL_CALL_RE = re.compile(r"^\s*TOOL:\s*(\w+)\((.*)\)\s*$", re.DOTALL)
MAX_TOOL_STEPS = 4
MAX_HISTORY_TURNS = 12


@dataclass
class GarudEngine:
    """Loads a model once and exposes generation methods for chat/tools/json/RAG modes."""

    model_name: str = DEFAULT_MODEL
    load_in_4bit: bool = False
    max_new_tokens: int = 512
    temperature: float = 0.7

    tokenizer: Any = field(init=False, default=None)
    model: Any = field(init=False, default=None)
    device: torch.device = field(init=False, default=None)

    def load(self, progress_callback: Callable[[str], None] | None = None) -> None:
        """Load tokenizer + model. Call once before generating."""
        if progress_callback:
            progress_callback(f"Loading {self.model_name}{' (4-bit)' if self.load_in_4bit else ''}...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.load_in_4bit:
            if self.device.type != "cuda":
                raise RuntimeError("load_in_4bit requires a CUDA GPU.")
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                device_map="auto",
            )
        else:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto" if self.device.type == "cuda" else None,
            )
            if self.device.type == "cpu":
                self.model.to(self.device)

        self.model.eval()
        if progress_callback:
            progress_callback(f"Ready on {self.device}.")

    @staticmethod
    def system_prompt_for(mode: str, rag_context: str | None = None) -> str:
        if mode == "tools":
            return TOOL_SYSTEM_PROMPT
        if mode == "json":
            return JSON_SYSTEM_PROMPT
        if mode == "rag":
            return RAG_SYSTEM_PROMPT_TEMPLATE.format(context=rag_context or "(no context retrieved)")
        return BASE_SYSTEM_PROMPT

    @staticmethod
    def trim_history(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Keep the system prompt plus the most recent MAX_HISTORY_TURNS turns."""
        system = messages[0]
        rest = messages[1:]
        max_messages = MAX_HISTORY_TURNS * 2
        if len(rest) > max_messages:
            rest = rest[-max_messages:]
        return [system] + rest

    def generate_once(self, messages: list[dict[str, str]]) -> str:
        """Run a single forward generation pass over the given messages."""
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else 1.0,
                top_p=0.9 if self.temperature > 0 else 1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs.input_ids.shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_with_tools(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, list[dict[str, str]], list[tuple[str, str, str]]]:
        """ReAct-style loop: let the model call tools before giving a final answer.

        Returns (final_reply, scratch_messages, tool_trace) where tool_trace
        is a list of (tool_name, arg, observation) for UI display.
        """
        scratch: list[dict[str, str]] = []
        trace: list[tuple[str, str, str]] = []
        working = messages + scratch

        for _ in range(MAX_TOOL_STEPS):
            reply = self.generate_once(working)
            match = TOOL_CALL_RE.match(reply)
            if not match:
                return reply, scratch, trace

            tool_name, arg = match.group(1), match.group(2).strip().strip("'\"")
            observation = run_tool(tool_name, arg)
            trace.append((tool_name, arg, observation))

            scratch.append({"role": "assistant", "content": reply})
            scratch.append({"role": "user", "content": f"OBSERVATION: {observation}"})
            working = messages + scratch

        working = messages + scratch + [
            {"role": "user", "content": "Give your best final answer now, in plain text, without calling any more tools."}
        ]
        reply = self.generate_once(working)
        return reply, scratch, trace

    @staticmethod
    def try_parse_json(text: str) -> str:
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            return json.dumps({"error": "model did not return valid JSON", "raw": text})
