"""H2: tools validate their arguments, cap their output, and fail as data."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from harness.messages import ToolCall
from harness.tools import (
    EchoTool,
    SearchMedicalLiteratureTool,
    Tool,
    ToolRegistry,
    ToolResult,
)


def call(name: str, **arguments) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments=arguments)


def test_valid_call_runs():
    result = ToolRegistry([EchoTool()]).execute(call("echo", text="hi"))

    assert result.ok
    assert result.content == "hi"


def test_missing_required_argument_is_reported_to_the_model():
    result = ToolRegistry([EchoTool()]).execute(call("echo"))

    assert not result.ok
    assert "invalid arguments" in result.content
    assert "arguments.text is required" in result.content


def test_wrong_argument_type_is_reported():
    result = ToolRegistry([EchoTool()]).execute(call("echo", text=42))

    assert not result.ok
    assert "arguments.text must be a string" in result.content


def test_unexpected_argument_is_rejected():
    result = ToolRegistry([EchoTool()]).execute(call("echo", text="hi", verbose=True))

    assert not result.ok
    assert "arguments.verbose is not an allowed argument" in result.content
    assert "text" in result.content


def test_enum_and_range_are_enforced():
    registry = ToolRegistry([SearchMedicalLiteratureTool()])

    bad_mode = registry.execute(call("search_medical_literature", question="asthma", mode="fast"))
    bad_k = registry.execute(call("search_medical_literature", question="asthma", k=50))
    too_short = registry.execute(call("search_medical_literature", question="ab"))

    assert "must be one of 'auto', 'local', 'global'" in bad_mode.content
    assert "arguments.k must be <= 20" in bad_k.content
    assert "at least 3 characters" in too_short.content


def test_unknown_tool_lists_available_tools():
    result = ToolRegistry([EchoTool()]).execute(call("shell", command="ls"))

    assert not result.ok
    assert result.content.startswith("unknown tool 'shell'")
    assert "echo" in result.content


def test_result_is_truncated_with_a_marker():
    registry = ToolRegistry([EchoTool()], max_result_chars=10)

    result = registry.execute(call("echo", text="x" * 100))

    assert result.ok
    assert len(result.content) > 10
    assert result.content.startswith("x" * 10)
    assert "[truncated: 90 of 100 characters omitted from echo" in result.content


def test_result_type_is_checked():
    class BadTool(Tool):
        name = "bad"
        description = "returns the wrong thing"

        def run(self, arguments):
            return "not a ToolResult"

    result = ToolRegistry([BadTool()]).execute(call("bad"))

    assert not result.ok
    assert "expected ToolResult" in result.content


def test_duplicate_and_nameless_tools_are_rejected():
    class Nameless(Tool):
        description = "no name"

    with pytest.raises(ValueError, match="no name"):
        ToolRegistry([Nameless()])
    with pytest.raises(ValueError, match="duplicate tool name"):
        ToolRegistry([EchoTool(), EchoTool()])


def test_specs_expose_a_function_schema():
    specs = ToolRegistry([EchoTool()]).specs()

    assert specs[0]["type"] == "function"
    assert specs[0]["function"]["name"] == "echo"
    assert specs[0]["function"]["parameters"]["required"] == ["text"]


def test_rag_stub_says_the_corpus_is_not_ready():
    result = ToolRegistry([SearchMedicalLiteratureTool()]).execute(
        call("search_medical_literature", question="What is asthma?")
    )

    assert not result.ok
    assert "not wired" in result.content


@dataclass
class FakeAnswer:
    answer: str
    citations: list[str]
    mode: str = "local"


class FakeBackend:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def answer(self, question: str, *, mode: str = "auto", k: int = 8):
        self.requests.append({"question": question, "mode": mode, "k": k})
        return FakeAnswer("Asthma is a chronic airway disease.", ["asthma:2"], mode="local")


def test_rag_tool_renders_answer_and_citations():
    backend = FakeBackend()
    result = ToolRegistry([SearchMedicalLiteratureTool(backend)]).execute(
        call("search_medical_literature", question="What is asthma?", mode="local", k=3)
    )

    assert result.ok
    assert "Asthma is a chronic airway disease." in result.content
    assert "Citations: asthma:2" in result.content
    assert backend.requests == [{"question": "What is asthma?", "mode": "local", "k": 3}]
    assert result.data["citations"] == ["asthma:2"]


def test_rag_backend_failure_becomes_a_tool_error():
    class BrokenBackend:
        def answer(self, question: str, *, mode: str = "auto", k: int = 8):
            raise RuntimeError("index missing")

    result = ToolRegistry([SearchMedicalLiteratureTool(BrokenBackend())]).execute(
        call("search_medical_literature", question="What is asthma?")
    )

    assert not result.ok
    assert "search_medical_literature failed: RuntimeError: index missing" in result.content
