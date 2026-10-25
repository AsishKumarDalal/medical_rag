"""A terminal you can talk to — deliberately the smallest thing that works.

    python -m harness                      # interactive
    python -m harness --once "question"     # one turn, then exit
    python -m harness --list                # sessions on disk
    python -m harness --session <id>        # resume

Commands inside the REPL: /help, /new, /sessions, /resume <id>, /usage, /exit.
The RAG tool is registered whether or not the pipeline exists; until G3 it answers
with a clear "not wired yet" error.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from typing import Callable, Sequence

from .config import Config, ConfigError
from .context import ContextManager, LLMSummarizer
from .llm import LLMClient
from .loop import TurnResult, run_turn
from .messages import system, user
from .security import UNTRUSTED_NOTICE, SecurityPolicy, redact
from .session import SessionStore
from .tools import EchoTool, RagBackend, SearchMedicalLiteratureTool, ToolRegistry

SYSTEM_PROMPT = (
    "You are a medical literature assistant. Answer from the indexed corpus, not "
    "from memory.\n"
    "- Call search_medical_literature before answering a medical question.\n"
    "- Cite the chunk ids the tool returns, in the form [article:chunk].\n"
    "- If the evidence does not contain the answer, say so plainly. Never guess, "
    "and never give personal medical advice.\n"
    "- " + UNTRUSTED_NOTICE
)

COMPACT_CHARS = 160


def _compact(value: str, limit: int = COMPACT_CHARS) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class Chat:
    """One terminal conversation: a session, a client and the tools."""

    def __init__(
        self,
        *,
        config: Config,
        client: object,
        store: SessionStore,
        tools: ToolRegistry,
        context: ContextManager | None,
        session_id: str | None = None,
        out: Callable[[str], None] = print,
    ) -> None:
        self.config = config
        self.client = client
        self.store = store
        self.tools = tools
        self.context = context
        self.out = out
        self.session_id = ""
        self.turn = 0
        if session_id:
            self.resume(session_id)
        else:
            self.new_session()

    # -- sessions ---------------------------------------------------------

    def new_session(self) -> str:
        session_id = self.store.create(title="new session", model=self.config.model)
        self.session_id = session_id
        self.turn = 0
        self.store.append_message(session_id, system(SYSTEM_PROMPT), turn=0)
        return session_id

    def resume(self, session_id: str) -> None:
        if self.store.get(session_id) is None:
            raise ValueError(f"unknown session {session_id!r}")
        self.session_id = session_id
        self.turn = self.store.next_turn(session_id) - 1

    # -- one turn ---------------------------------------------------------

    def ask(self, text: str) -> TurnResult:
        turn = self.store.next_turn(self.session_id)
        self.turn = turn
        conversation = self.store.load_messages(self.session_id)
        message = user(text)
        self.store.append_message(self.session_id, message, turn=turn)
        conversation.append(message)

        result = run_turn(
            conversation,
            client=self.client,  # type: ignore[arg-type]
            tools=self.tools,
            max_iterations=self.config.max_iterations,
            context=self.context,
            recorder=self.store.recorder(self.session_id, turn=turn),
        )
        self._report(result, conversation)
        self._maybe_title(text)
        return result

    def _report(self, result: TurnResult, before: Sequence[object]) -> None:
        for message in result.messages[len(before) :]:
            if message.tool_calls:
                for call in message.tool_calls:
                    arguments = call.raw_arguments or json.dumps(call.arguments)
                    self.out(f"  -> {call.name}({_compact(arguments, 120)})")
            elif message.role == "tool":
                self.out(f"  <- {message.name or 'tool'}: {_compact(message.content)}")

        for event in result.context_events:
            self.out(
                f"[context] {event.kind}: {event.detail} "
                f"({event.before_tokens} -> {event.after_tokens} tokens)"
            )

        if result.stop_reason == "answer":
            self.out("\n" + (result.final.strip() or "(empty answer)"))
        elif result.stop_reason == "max_iterations":
            self.out(f"\n[stopped] {result.error}")
        else:
            self.out(f"\n[error] {result.error}")

        usage = self.store.usage_totals(self.session_id)
        if usage.get("calls"):
            self.out(
                f"[usage] prompt={usage['prompt_tokens']} "
                f"completion={usage['completion_tokens']} total={usage['total_tokens']} "
                f"calls={usage['calls']}"
            )

    def _maybe_title(self, text: str) -> None:
        info = self.store.get(self.session_id)
        if info is not None and info.title == "new session":
            self.store.rename(self.session_id, _compact(text, 60) or "untitled")

    # -- commands ---------------------------------------------------------

    def handle(self, line: str) -> bool:
        """Handle one line. Returns False when the session should end."""
        text = line.strip()
        if not text:
            return True
        if text.startswith("/"):
            return self._command(text)
        self.ask(text)
        return True

    def _command(self, text: str) -> bool:
        name, _, argument = text.partition(" ")
        argument = argument.strip()

        if name in ("/exit", "/quit"):
            return False
        if name == "/help":
            self.out(
                "/new              start a fresh session\n"
                "/sessions         list recent sessions\n"
                "/resume <id>      continue a session\n"
                "/usage            tokens used in this session\n"
                "/exit             leave"
            )
            return True
        if name == "/new":
            self.out(f"[session] {self.new_session()}")
            return True
        if name == "/sessions":
            sessions = self.store.sessions()
            if not sessions:
                self.out("no sessions yet")
            for info in sessions:
                marker = "*" if info.id == self.session_id else " "
                self.out(
                    f"{marker} {info.id}  {info.messages:>3} messages  "
                    f"{info.updated_at}  {info.title or ''}"
                )
            return True
        if name == "/resume":
            if not argument:
                self.out("usage: /resume <id>")
                return True
            try:
                self.resume(argument)
            except ValueError as exc:
                self.out(f"[error] {exc}")
                return True
            self.out(f"[session] resumed {argument}")
            return True
        if name == "/usage":
            usage = self.store.usage_totals(self.session_id)
            if not usage.get("calls"):
                self.out("no usage recorded in this session")
            else:
                self.out(
                    f"prompt={usage['prompt_tokens']} completion={usage['completion_tokens']} "
                    f"total={usage['total_tokens']} calls={usage['calls']}"
                )
            return True

        self.out(f"unknown command {name}; try /help")
        return True


def _rag_backend() -> RagBackend | None:
    """Wire the pipeline if it is importable and configured; otherwise stay honest."""
    try:
        from graphrag import GraphRagBackend  # type: ignore[import-not-found]

        return GraphRagBackend.from_env()
    except Exception:
        return None


def build_tools(config: Config) -> ToolRegistry:
    policy = SecurityPolicy.from_config(config)
    return ToolRegistry(
        [EchoTool(), SearchMedicalLiteratureTool(_rag_backend())],
        policy=policy,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    client: object | None = None,
    store: SessionStore | None = None,
    tools: ToolRegistry | None = None,
    context: ContextManager | None = None,
    out: Callable[[str], None] = print,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m harness",
        description="A small medical-literature agent with tools, context and sessions.",
    )
    parser.add_argument("--once", metavar="QUESTION", help="ask one question, then exit")
    parser.add_argument("--session", metavar="ID", help="resume this session")
    parser.add_argument("--db", metavar="PATH", help="session database (default: sessions.db)")
    parser.add_argument("--list", action="store_true", help="list sessions and exit")
    args = parser.parse_args(argv)

    try:
        config = Config.from_env()
    except ConfigError as exc:
        out(f"[config] {exc}")
        return 2
    if args.db:
        config = replace(config, db_path=args.db)

    owned_store = store is None
    session_store = store or SessionStore(config.db_path, sanitize=redact)
    try:
        if args.list:
            sessions = session_store.sessions()
            if not sessions:
                out("no sessions yet")
            for info in sessions:
                out(
                    f"{info.id}  {info.messages:>3} messages  {info.updated_at}  "
                    f"{info.title or ''}"
                )
            return 0

        model_client = client or LLMClient(config)
        tool_registry = tools or build_tools(config)
        manager = context
        if manager is None:
            manager = ContextManager.from_config(
                config, summarizer=LLMSummarizer(model_client)  # type: ignore[arg-type]
            )
        try:
            chat = Chat(
                config=config,
                client=model_client,
                store=session_store,
                tools=tool_registry,
                context=manager,
                session_id=args.session,
                out=out,
            )
        except ValueError as exc:
            out(f"[error] {exc}")
            return 2

        if args.once:
            chat.ask(args.once)
            return 0

        out(f"medical harness — session {chat.session_id} — /help for commands")
        while True:
            try:
                line = input("you> ")
            except (EOFError, KeyboardInterrupt):
                out("")
                return 0
            try:
                if not chat.handle(line):
                    return 0
            except KeyboardInterrupt:
                out("\n[interrupted — the turn was cancelled]")
    finally:
        if owned_store:
            session_store.close()


if __name__ == "__main__":
    sys.exit(main())
