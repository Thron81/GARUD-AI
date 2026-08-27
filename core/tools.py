"""Tool implementations for Garud AI's ReAct-style tool-use loop.

Each tool is a plain Python function that takes a single string argument
(parsed from the model's tool-call text) and returns a string observation
that gets fed back into the conversation. Keeping tools simple and
string-in/string-out makes them easy to parse, log, and extend.
"""

from __future__ import annotations

import ast
import math
import operator
import os

# --- Calculator -------------------------------------------------------
# A restricted arithmetic evaluator (no eval()) so the model can only ever
# compute numeric expressions, not run arbitrary code.

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_NAMES = {"pi": math.pi, "e": math.e}
_ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "abs": abs,
    "round": round,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _ALLOWED_NAMES:
        return _ALLOWED_NAMES[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS:
        args = [_safe_eval(a) for a in node.args]
        return _ALLOWED_FUNCS[node.func.id](*args)
    raise ValueError(f"Disallowed expression: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """Evaluate a restricted arithmetic expression safely."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as error:  # noqa: BLE001 - surface any parse/eval error to the model
        return f"Error: could not evaluate '{expression}': {error}"


# --- File search --------------------------------------------------------
# Restricted to a single working directory so the model can't read
# arbitrary paths on the host machine.

WORKDIR = os.path.abspath(os.environ.get("GARUD_WORKDIR", "."))


def _safe_join(relative_path: str) -> str | None:
    """Resolve a path inside WORKDIR, rejecting any attempt to escape it."""
    candidate = os.path.abspath(os.path.join(WORKDIR, relative_path))
    if os.path.commonpath([candidate, WORKDIR]) != WORKDIR:
        return None
    return candidate


def file_search(query: str) -> str:
    """List files under WORKDIR whose name contains the query substring."""
    matches = []
    for root, _dirs, files in os.walk(WORKDIR):
        for name in files:
            if query.lower() in name.lower():
                rel = os.path.relpath(os.path.join(root, name), WORKDIR)
                matches.append(rel)
    if not matches:
        return f"No files matching '{query}' found under {WORKDIR}."
    return "Found:\n" + "\n".join(matches[:20])


def read_file(relative_path: str) -> str:
    """Read a text file inside WORKDIR (first 4000 characters)."""
    path = _safe_join(relative_path)
    if path is None:
        return "Error: path escapes the allowed working directory."
    if not os.path.isfile(path):
        return f"Error: '{relative_path}' is not a file."
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            content = handle.read(4000)
        return content if content else "(file is empty)"
    except Exception as error:  # noqa: BLE001
        return f"Error reading '{relative_path}': {error}"


# --- Registry -------------------------------------------------------------

TOOLS = {
    "calculator": calculator,
    "file_search": file_search,
    "read_file": read_file,
}

TOOL_DESCRIPTIONS = """\
calculator(expression) - evaluate an arithmetic expression, e.g. calculator(2*3.5+sqrt(9))
file_search(query) - list files under the working directory whose name contains query
read_file(relative_path) - read up to 4000 characters of a text file under the working directory
"""


def run_tool(name: str, arg: str) -> str:
    fn = TOOLS.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'. Available tools: {', '.join(TOOLS)}"
    return fn(arg)
