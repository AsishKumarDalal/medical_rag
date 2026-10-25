"""Live H1 check: one real turn against the configured model, using the echo tool.

Not a pytest test: it needs network access and a working endpoint.

    .venv/bin/python test/smoke_h1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run from anywhere: put the repo root (not test/) on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.config import Config
from harness.llm import LLMClient
from harness.loop import TurnResult, run_turn
from harness.messages import Message, system, user
from harness.tools import EchoTool, ToolRegistry

SYSTEM_PROMPT = (
    "You are a tool-use test harness. When asked to echo a phrase, call the echo "
    "tool exactly once with that phrase, then reply with the exact text the tool "
    "returned and nothing else."
)


def render(result: TurnResult) -> str:
    lines: list[str] = []
    for message in result.messages:
        label = message.role
        if message.tool_calls:
            calls = ", ".join(
                f"{call.name}({call.raw_arguments or call.arguments})"
                for call in message.tool_calls
            )
            lines.append(f"[{label}] tool_call {calls}")
        if message.content:
            lines.append(f"[{label}] {message.content}")
        if message.role == "tool":
            lines[-1] = f"[tool:{message.name}] {message.content}"
    lines.append(
        f"--- stop_reason={result.stop_reason} iterations={result.iterations}"
        + (f" error={result.error}" if result.error else "")
    )
    return "\n".join(lines)


def main() -> int:
    config = Config.from_env()
    client = LLMClient(config)
    tools = ToolRegistry([EchoTool()])
    conversation: list[Message] = [
        system(SYSTEM_PROMPT),
        user("Echo the phrase 'loop is alive', then tell me exactly what came back."),
    ]

    print(f"model={config.model} base_url={config.base_url}")
    result = run_turn(
        conversation,
        client=client,
        tools=tools,
        max_iterations=config.max_iterations,
    )
    print(render(result))
    return 0 if result.answered else 1


if __name__ == "__main__":
    sys.exit(main())
