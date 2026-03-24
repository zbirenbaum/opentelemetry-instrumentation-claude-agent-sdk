"""Tests for invoke_agent span production via query() wrapper (T010)."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING, Any

import pytest
from opentelemetry.sdk.metrics import MeterProvider as SDKMeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from opentelemetry.instrumentation.claude_agent_sdk._constants import (
    ERROR_TYPE,
    GEN_AI_CONVERSATION_ID,
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_OPERATION_NAME,
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_SYSTEM_INSTRUCTIONS,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    OPERATION_INVOKE_AGENT,
    SYSTEM_ANTHROPIC,
)
from opentelemetry.instrumentation.claude_agent_sdk._instrumentor import ClaudeAgentSdkInstrumentor
from tests.unit.conftest import make_usage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _create_mock_sdk(messages: list[Any] | None = None) -> ModuleType:
    """Create mock claude_agent_sdk that yields given messages."""
    mock_module = ModuleType("claude_agent_sdk")

    @dataclass
    class AssistantMessage:
        model: str = "claude-sonnet-4-20250514"
        content: list = field(default_factory=lambda: [{"type": "text", "text": "Hello, I'm Claude!"}])

    @dataclass
    class ResultMessage:
        usage: dict[str, int] | None = field(default_factory=make_usage)
        session_id: str = "test-session"
        subtype: str = "success"
        is_error: bool = False

    if messages is None:
        messages = [
            AssistantMessage(),
            ResultMessage(),
        ]

    @dataclass
    class ClaudeAgentOptions:
        model: str | None = None
        hooks: dict[str, list[Any]] = field(default_factory=dict)
        system_prompt: str | None = None

    async def mock_query(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        for msg in messages:
            yield msg

    class ClaudeSDKClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.options = kwargs.get("options", ClaudeAgentOptions())

        async def query(self, *args: Any, **kwargs: Any) -> Any:
            pass

        async def receive_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            for msg in messages:
                yield msg

    mock_module.query = mock_query  # type: ignore[attr-defined]
    mock_module.ClaudeSDKClient = ClaudeSDKClient  # type: ignore[attr-defined]
    mock_module.ClaudeAgentOptions = ClaudeAgentOptions  # type: ignore[attr-defined]
    mock_module.AssistantMessage = AssistantMessage  # type: ignore[attr-defined]
    mock_module.ResultMessage = ResultMessage  # type: ignore[attr-defined]

    return mock_module


@pytest.fixture()
def otel_setup():
    """Set up OTel tracer and meter providers for testing."""
    exporter = InMemorySpanExporter()
    tp = SDKTracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exporter))

    reader = InMemoryMetricReader()
    mp = SDKMeterProvider(metric_readers=[reader])

    return tp, mp, exporter, reader


@pytest.fixture()
def mock_sdk():
    """Install a mock claude_agent_sdk module."""
    mock_module = _create_mock_sdk()
    original = sys.modules.get("claude_agent_sdk")
    sys.modules["claude_agent_sdk"] = mock_module
    yield mock_module
    if original is not None:
        sys.modules["claude_agent_sdk"] = original
    else:
        sys.modules.pop("claude_agent_sdk", None)


class TestInvokeAgentSpan:
    async def test_query_produces_span(self, mock_sdk, otel_setup):
        tp, mp, exporter, _reader = otel_setup
        instrumentor = ClaudeAgentSdkInstrumentor()
        instrumentor.instrument(tracer_provider=tp, meter_provider=mp)

        try:
            import claude_agent_sdk

            async for _ in claude_agent_sdk.query(prompt="test prompt"):
                pass

            spans = exporter.get_finished_spans()
            assert len(spans) == 1
            span = spans[0]
            assert span.name == OPERATION_INVOKE_AGENT
            assert span.kind == SpanKind.CLIENT

            attrs = dict(span.attributes or {})
            assert attrs[GEN_AI_OPERATION_NAME] == OPERATION_INVOKE_AGENT
            assert attrs[GEN_AI_SYSTEM] == SYSTEM_ANTHROPIC
        finally:
            instrumentor.uninstrument()

    async def test_span_with_agent_name(self, mock_sdk, otel_setup):
        tp, mp, exporter, _reader = otel_setup
        instrumentor = ClaudeAgentSdkInstrumentor()
        instrumentor.instrument(tracer_provider=tp, meter_provider=mp, agent_name="my-agent")

        try:
            import claude_agent_sdk

            async for _ in claude_agent_sdk.query(prompt="test"):
                pass

            spans = exporter.get_finished_spans()
            assert spans[0].name == "invoke_agent my-agent"
        finally:
            instrumentor.uninstrument()

    async def test_model_extraction_from_assistant_message(self, mock_sdk, otel_setup):
        tp, mp, exporter, _reader = otel_setup
        instrumentor = ClaudeAgentSdkInstrumentor()
        instrumentor.instrument(tracer_provider=tp, meter_provider=mp)

        try:
            import claude_agent_sdk

            async for _ in claude_agent_sdk.query(prompt="test"):
                pass

            spans = exporter.get_finished_spans()
            attrs = dict(spans[0].attributes or {})
            assert attrs[GEN_AI_RESPONSE_MODEL] == "claude-sonnet-4-20250514"
        finally:
            instrumentor.uninstrument()

    async def test_token_usage_attributes(self, mock_sdk, otel_setup):
        tp, mp, exporter, _reader = otel_setup
        instrumentor = ClaudeAgentSdkInstrumentor()
        instrumentor.instrument(tracer_provider=tp, meter_provider=mp)

        try:
            import claude_agent_sdk

            async for _ in claude_agent_sdk.query(prompt="test"):
                pass

            spans = exporter.get_finished_spans()
            attrs = dict(spans[0].attributes or {})
            assert GEN_AI_USAGE_INPUT_TOKENS in attrs
            assert GEN_AI_USAGE_OUTPUT_TOKENS in attrs
        finally:
            instrumentor.uninstrument()

    async def test_conversation_id(self, mock_sdk, otel_setup):
        tp, mp, exporter, _reader = otel_setup
        instrumentor = ClaudeAgentSdkInstrumentor()
        instrumentor.instrument(tracer_provider=tp, meter_provider=mp)

        try:
            import claude_agent_sdk

            async for _ in claude_agent_sdk.query(prompt="test"):
                pass

            spans = exporter.get_finished_spans()
            attrs = dict(spans[0].attributes or {})
            assert attrs[GEN_AI_CONVERSATION_ID] == "test-session"
        finally:
            instrumentor.uninstrument()

    async def test_finish_reason_mapping(self, mock_sdk, otel_setup):
        tp, mp, exporter, _reader = otel_setup
        instrumentor = ClaudeAgentSdkInstrumentor()
        instrumentor.instrument(tracer_provider=tp, meter_provider=mp)

        try:
            import claude_agent_sdk

            async for _ in claude_agent_sdk.query(prompt="test"):
                pass

            spans = exporter.get_finished_spans()
            attrs = dict(spans[0].attributes or {})
            assert attrs[GEN_AI_RESPONSE_FINISH_REASONS] == ("end_turn",)
        finally:
            instrumentor.uninstrument()

    async def test_error_handling(self, otel_setup):
        """query() that raises should produce span with error attributes."""
        mock_module = _create_mock_sdk()

        # Get the AssistantMessage class from the mock module
        assistant_msg_cls = mock_module.AssistantMessage  # type: ignore[attr-defined]

        async def error_query(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            yield assistant_msg_cls()
            raise RuntimeError("SDK error")

        mock_module.query = error_query  # type: ignore[attr-defined]

        original = sys.modules.get("claude_agent_sdk")
        sys.modules["claude_agent_sdk"] = mock_module

        tp, mp, exporter, _reader = otel_setup
        instrumentor = ClaudeAgentSdkInstrumentor()
        instrumentor.instrument(tracer_provider=tp, meter_provider=mp)

        try:
            import claude_agent_sdk

            with pytest.raises(RuntimeError, match="SDK error"):
                async for _ in claude_agent_sdk.query(prompt="test"):
                    pass

            spans = exporter.get_finished_spans()
            assert len(spans) == 1
            assert spans[0].status.status_code == StatusCode.ERROR
            attrs = dict(spans[0].attributes or {})
            assert attrs[ERROR_TYPE] == "RuntimeError"
        finally:
            instrumentor.uninstrument()
            if original is not None:
                sys.modules["claude_agent_sdk"] = original
            else:
                sys.modules.pop("claude_agent_sdk", None)

    async def test_parent_span_nesting(self, mock_sdk, otel_setup):
        """invoke_agent should nest under an existing parent span."""

        tp, mp, exporter, _reader = otel_setup
        instrumentor = ClaudeAgentSdkInstrumentor()
        instrumentor.instrument(tracer_provider=tp, meter_provider=mp)

        try:
            import claude_agent_sdk

            tracer = tp.get_tracer("test")
            with tracer.start_as_current_span("parent-operation"):
                async for _ in claude_agent_sdk.query(prompt="test"):
                    pass

            spans = exporter.get_finished_spans()
            invoke_spans = [s for s in spans if s.name.startswith("invoke_agent")]
            parent_spans = [s for s in spans if s.name == "parent-operation"]

            assert len(invoke_spans) == 1
            assert len(parent_spans) == 1

            # invoke_agent span should have parent-operation as parent
            assert invoke_spans[0].parent is not None
            assert invoke_spans[0].parent.span_id == parent_spans[0].context.span_id
        finally:
            instrumentor.uninstrument()

    async def test_root_span_when_no_parent(self, mock_sdk, otel_setup):
        """invoke_agent should be a root span when no parent exists."""
        tp, mp, exporter, _reader = otel_setup
        instrumentor = ClaudeAgentSdkInstrumentor()
        instrumentor.instrument(tracer_provider=tp, meter_provider=mp)

        try:
            import claude_agent_sdk

            async for _ in claude_agent_sdk.query(prompt="test"):
                pass

            spans = exporter.get_finished_spans()
            assert len(spans) == 1
            assert spans[0].parent is None
        finally:
            instrumentor.uninstrument()


class TestContentCapture:
    """Tests for opt-in prompt and response content capture (gen_ai semantic conventions)."""

    async def test_input_messages_captured_when_enabled(self, mock_sdk, otel_setup):
        """gen_ai.input.messages should be set when capture_content=True."""
        tp, mp, exporter, _reader = otel_setup
        instrumentor = ClaudeAgentSdkInstrumentor()
        instrumentor.instrument(tracer_provider=tp, meter_provider=mp, capture_content=True)

        try:
            import claude_agent_sdk

            async for _ in claude_agent_sdk.query(prompt="What is the capital of France?"):
                pass

            spans = exporter.get_finished_spans()
            attrs = dict(spans[0].attributes or {})
            assert GEN_AI_INPUT_MESSAGES in attrs
            import json

            messages = json.loads(attrs[GEN_AI_INPUT_MESSAGES])
            assert messages == [{"role": "user", "content": "What is the capital of France?"}]
        finally:
            instrumentor.uninstrument()

    async def test_output_messages_captured_when_enabled(self, mock_sdk, otel_setup):
        """gen_ai.output.messages should be set from AssistantMessage.content when capture_content=True."""
        tp, mp, exporter, _reader = otel_setup
        instrumentor = ClaudeAgentSdkInstrumentor()
        instrumentor.instrument(tracer_provider=tp, meter_provider=mp, capture_content=True)

        try:
            import claude_agent_sdk

            async for _ in claude_agent_sdk.query(prompt="test"):
                pass

            spans = exporter.get_finished_spans()
            attrs = dict(spans[0].attributes or {})
            assert GEN_AI_OUTPUT_MESSAGES in attrs
            import json

            messages = json.loads(attrs[GEN_AI_OUTPUT_MESSAGES])
            assert messages[0]["role"] == "assistant"
            assert messages[0]["content"] == [{"type": "text", "text": "Hello, I'm Claude!"}]
        finally:
            instrumentor.uninstrument()

    async def test_system_instructions_captured_when_enabled(self, otel_setup):
        """gen_ai.system_instructions should be set from options.system_prompt when capture_content=True."""
        mock_module = _create_mock_sdk()
        original = sys.modules.get("claude_agent_sdk")
        sys.modules["claude_agent_sdk"] = mock_module

        tp, mp, exporter, _reader = otel_setup
        instrumentor = ClaudeAgentSdkInstrumentor()
        instrumentor.instrument(tracer_provider=tp, meter_provider=mp, capture_content=True)

        try:
            import claude_agent_sdk

            options = claude_agent_sdk.ClaudeAgentOptions(system_prompt="You are a helpful assistant.")
            async for _ in claude_agent_sdk.query(prompt="test", options=options):
                pass

            spans = exporter.get_finished_spans()
            attrs = dict(spans[0].attributes or {})
            assert attrs[GEN_AI_SYSTEM_INSTRUCTIONS] == "You are a helpful assistant."
        finally:
            instrumentor.uninstrument()
            if original is not None:
                sys.modules["claude_agent_sdk"] = original
            else:
                sys.modules.pop("claude_agent_sdk", None)

    async def test_content_not_captured_by_default(self, mock_sdk, otel_setup):
        """No content attributes should appear when capture_content is not set (default=False)."""
        tp, mp, exporter, _reader = otel_setup
        instrumentor = ClaudeAgentSdkInstrumentor()
        instrumentor.instrument(tracer_provider=tp, meter_provider=mp)  # capture_content defaults to False

        try:
            import claude_agent_sdk

            async for _ in claude_agent_sdk.query(prompt="What is the capital of France?"):
                pass

            spans = exporter.get_finished_spans()
            attrs = dict(spans[0].attributes or {})
            assert GEN_AI_INPUT_MESSAGES not in attrs
            assert GEN_AI_OUTPUT_MESSAGES not in attrs
            assert GEN_AI_SYSTEM_INSTRUCTIONS not in attrs
        finally:
            instrumentor.uninstrument()
