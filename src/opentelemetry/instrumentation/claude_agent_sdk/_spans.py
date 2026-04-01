"""Span creation and attribute helpers for GenAI semantic conventions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import SpanKind, StatusCode, Tracer

from opentelemetry.instrumentation.claude_agent_sdk._constants import (
    ERROR_TYPE,
    FINISH_REASON_MAP,
    GEN_AI_AGENT_DESCRIPTION,
    GEN_AI_AGENT_ID,
    GEN_AI_AGENT_NAME,
    GEN_AI_CONVERSATION_ID,
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_OPERATION_NAME,
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_SYSTEM_INSTRUCTIONS,
    GEN_AI_TOOL_CALL_ID,
    GEN_AI_TOOL_DEFINITIONS,
    GEN_AI_TOOL_NAME,
    GEN_AI_TOOL_TYPE,
    GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    MCP_TOOL_PREFIX,
    OPERATION_EXECUTE_TOOL,
    OPERATION_INVOKE_AGENT,
    SYSTEM_ANTHROPIC,
    TOOL_TYPE_EXTENSION,
    TOOL_TYPE_FUNCTION,
)

if TYPE_CHECKING:
    from opentelemetry.context import Context
    from opentelemetry.trace import Span


def create_invoke_agent_span(
    tracer: Tracer,
    agent_name: str | None = None,
    agent_id: str | None = None,
    agent_description: str | None = None,
    request_model: str | None = None,
    options: Any = None,
) -> Span:
    """Create an invoke_agent CLIENT span with GenAI semantic convention attributes.

    Args:
        tracer: OTel tracer instance.
        agent_name: Optional agent name for span name and attribute.
        agent_id: Optional agent ID.
        agent_description: Optional agent description.
        request_model: Optional model name for gen_ai.request.model.
        options: Optional ClaudeAgentOptions (used to extract model if request_model not set).

    Returns:
        A started span (must be ended by caller).
    """
    span_name = f"{OPERATION_INVOKE_AGENT} {agent_name}" if agent_name else OPERATION_INVOKE_AGENT

    attributes: dict[str, str | int | list[str]] = {
        GEN_AI_OPERATION_NAME: OPERATION_INVOKE_AGENT,
        GEN_AI_PROVIDER_NAME: SYSTEM_ANTHROPIC,
    }

    if agent_name:
        attributes[GEN_AI_AGENT_NAME] = agent_name
    if agent_id:
        attributes[GEN_AI_AGENT_ID] = agent_id
    if agent_description:
        attributes[GEN_AI_AGENT_DESCRIPTION] = agent_description

    # Resolve model: explicit param > options.model
    model = request_model
    if model is None and options is not None:
        model = getattr(options, "model", None)
    if model is not None:
        attributes[GEN_AI_REQUEST_MODEL] = model

    return tracer.start_span(name=span_name, kind=SpanKind.CLIENT, attributes=attributes)


def set_result_attributes(span: Span, result_message: Any) -> None:
    """Set token usage, finish reason, and conversation.id from a ResultMessage.

    Args:
        span: The invoke_agent span to annotate.
        result_message: SDK ResultMessage with usage, session_id, subtype.
    """
    usage = getattr(result_message, "usage", None)
    if usage is not None:
        input_tokens = usage.get("input_tokens", 0) or 0
        cache_creation = usage.get("cache_creation_input_tokens", 0) or 0
        cache_read = usage.get("cache_read_input_tokens", 0) or 0
        output_tokens = usage.get("output_tokens", 0) or 0

        total_input = input_tokens + cache_creation + cache_read
        span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, total_input)
        span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)

        if cache_creation > 0:
            span.set_attribute(GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS, cache_creation)
        if cache_read > 0:
            span.set_attribute(GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, cache_read)

    # Finish reason
    subtype = getattr(result_message, "subtype", None)
    if subtype is not None:
        finish_reason = FINISH_REASON_MAP.get(subtype, subtype)
        span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [finish_reason])

    # Conversation ID from session_id
    session_id = getattr(result_message, "session_id", None)
    if session_id is not None:
        span.set_attribute(GEN_AI_CONVERSATION_ID, session_id)


def set_response_model(span: Span, model: str) -> None:
    """Set the response model attribute on a span."""
    span.set_attribute(GEN_AI_RESPONSE_MODEL, model)


def set_error_attributes(span: Span, exception: BaseException) -> None:
    """Set error.type and ERROR status on a span.

    Args:
        span: The span to annotate with error info.
        exception: The exception that occurred.
    """
    error_type = type(exception).__qualname__
    span.set_attribute(ERROR_TYPE, error_type)
    span.set_status(StatusCode.ERROR, str(exception))


# --- Serialization helpers ---


def _to_serializable(obj: Any) -> Any:
    """Recursively convert Pydantic models and other non-serializable objects to plain dicts/lists."""
    if hasattr(obj, "model_dump"):
        return _to_serializable(obj.model_dump())
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(item) for item in obj]
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return _to_serializable(vars(obj))
    return obj


# --- Semconv message format helpers ---


def _content_block_to_part(block: Any) -> dict[str, Any]:
    """Convert a Claude SDK content block to a semconv message part.

    Handles TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock,
    and falls back to a generic part for unknown types.
    """
    block_type = type(block).__name__

    if block_type == "TextBlock":
        return {"type": "text", "content": getattr(block, "text", str(block))}

    if block_type == "ThinkingBlock":
        return {"type": "reasoning", "content": getattr(block, "thinking", str(block))}

    if block_type == "ToolUseBlock":
        part: dict[str, Any] = {
            "type": "tool_call",
            "name": getattr(block, "name", "unknown"),
        }
        tool_id = getattr(block, "id", None)
        if tool_id:
            part["id"] = tool_id
        tool_input = getattr(block, "input", None)
        if tool_input is not None:
            part["arguments"] = _to_serializable(tool_input)
        return part

    if block_type == "ToolResultBlock":
        part = {"type": "tool_call_response"}
        tool_use_id = getattr(block, "tool_use_id", None)
        if tool_use_id:
            part["id"] = tool_use_id
        content = getattr(block, "content", None)
        part["response"] = _to_serializable(content) if content is not None else ""
        return part

    # Fallback: serialize as generic part
    serialized = _to_serializable(block)
    if isinstance(serialized, dict):
        if "type" not in serialized:
            serialized["type"] = block_type.lower()
        return serialized
    return {"type": block_type.lower(), "content": str(block)}


def _content_blocks_to_parts(content: Any) -> list[dict[str, Any]]:
    """Convert SDK content (str or list of content blocks) to semconv parts list."""
    if isinstance(content, str):
        return [{"type": "text", "content": content}]
    if isinstance(content, list):
        return [_content_block_to_part(block) for block in content]
    return [{"type": "text", "content": str(content)}]


def content_to_semconv_input_message(role: str, content: Any) -> dict[str, Any]:
    """Convert content to a semconv input message with role and parts.

    Args:
        role: Message role (user, assistant, tool, system).
        content: Raw content (str, list of content blocks, or dict).

    Returns:
        A dict matching the gen_ai.input.messages JSON schema: {role, parts: [...]}.
    """
    return {"role": role, "parts": _content_blocks_to_parts(content)}


def assistant_content_to_semconv_output(content: Any, finish_reason: str | None = None) -> dict[str, Any]:
    """Convert assistant content to a semconv output message.

    Args:
        content: AssistantMessage.content (list of content blocks or str).
        finish_reason: Optional finish reason (stop, tool_call, etc.).

    Returns:
        A dict matching the gen_ai.output.messages JSON schema: {role, parts: [...], finish_reason}.
    """
    msg: dict[str, Any] = {
        "role": "assistant",
        "parts": _content_blocks_to_parts(content),
    }
    if finish_reason:
        msg["finish_reason"] = finish_reason
    return msg


def tool_result_to_semconv_message(tool_use_id: str, response: Any) -> dict[str, Any]:
    """Create a semconv tool message from a tool result.

    Args:
        tool_use_id: The tool call ID this result corresponds to.
        response: The tool's response data.

    Returns:
        A dict matching the gen_ai.input.messages JSON schema for role=tool.
    """
    return {
        "role": "tool",
        "parts": [
            {
                "type": "tool_call_response",
                "id": tool_use_id,
                "response": _to_serializable(response) if response is not None else "",
            }
        ],
    }


# --- Content capture helpers (opt-in) ---


def set_prompt_attributes(
    span: Span,
    prompt: Any = None,
    system_prompt: str | None = None,
    tool_definitions: Any = None,
) -> None:
    """Set opt-in prompt content attributes on an invoke_agent span.

    Formats messages according to the GenAI semantic conventions JSON schemas.

    Args:
        span: The invoke_agent span to annotate.
        prompt: The user prompt (str or structured message list).
        system_prompt: The system instructions string.
        tool_definitions: Tool definitions list from ClaudeAgentOptions.
    """
    if prompt is not None:
        if isinstance(prompt, list):
            # Already structured messages — convert each to semconv format
            messages: list[dict[str, Any]] = []
            for msg in prompt:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", msg.get("parts", ""))
                    messages.append(content_to_semconv_input_message(role, content))
                else:
                    messages.append(content_to_semconv_input_message("user", msg))
        elif isinstance(prompt, str):
            messages = [content_to_semconv_input_message("user", prompt)]
        else:
            messages = [content_to_semconv_input_message("user", str(prompt))]
        try:
            span.set_attribute(GEN_AI_INPUT_MESSAGES, json.dumps(messages))
        except (TypeError, ValueError):
            span.set_attribute(GEN_AI_INPUT_MESSAGES, str(messages))

    if system_prompt is not None:
        # Semconv system instructions format: [{type: "text", content: "..."}]
        instructions = [{"type": "text", "content": system_prompt}]
        try:
            span.set_attribute(GEN_AI_SYSTEM_INSTRUCTIONS, json.dumps(instructions))
        except (TypeError, ValueError):
            span.set_attribute(GEN_AI_SYSTEM_INSTRUCTIONS, str(instructions))

    if tool_definitions is not None:
        try:
            span.set_attribute(GEN_AI_TOOL_DEFINITIONS, json.dumps(_to_serializable(tool_definitions)))
        except (TypeError, ValueError):
            span.set_attribute(GEN_AI_TOOL_DEFINITIONS, str(tool_definitions))


def set_response_content(span: Span, content: Any, finish_reason: str | None = None) -> None:
    """Set gen_ai.output.messages on an invoke_agent span from AssistantMessage content.

    Formats output according to the GenAI output messages JSON schema.

    Args:
        span: The invoke_agent span to annotate.
        content: AssistantMessage.content (str or list of content blocks).
        finish_reason: Optional finish reason.
    """
    if content is None:
        return

    # Determine finish_reason from content: if any tool_call parts, it's "tool_call"
    if finish_reason is None and isinstance(content, list):
        for block in content:
            if type(block).__name__ == "ToolUseBlock":
                finish_reason = "tool_call"
                break

    messages = [assistant_content_to_semconv_output(content, finish_reason)]
    try:
        span.set_attribute(GEN_AI_OUTPUT_MESSAGES, json.dumps(messages))
    except (TypeError, ValueError):
        span.set_attribute(GEN_AI_OUTPUT_MESSAGES, str(messages))


def set_conversation_history(span: Span, history: list[dict[str, Any]]) -> None:
    """Set gen_ai.input.messages to the full conversation history.

    Args:
        span: The invoke_agent span to annotate.
        history: Full conversation history in semconv format.
    """
    if not history:
        return
    try:
        span.set_attribute(GEN_AI_INPUT_MESSAGES, json.dumps(history))
    except (TypeError, ValueError):
        span.set_attribute(GEN_AI_INPUT_MESSAGES, str(history))


# --- Tool span helpers ---


def derive_tool_type(tool_name: str) -> str:
    """Derive tool type from tool name.

    'mcp__*' tools are 'extension' (MCP tools), all others are 'function'.
    """
    if tool_name.startswith(MCP_TOOL_PREFIX):
        return TOOL_TYPE_EXTENSION
    return TOOL_TYPE_FUNCTION


def create_execute_tool_span(
    tracer: Tracer,
    tool_name: str,
    tool_use_id: str,
    parent_context: Context | None = None,
) -> Span:
    """Create an execute_tool INTERNAL span with tool attributes.

    Args:
        tracer: OTel tracer instance.
        tool_name: The tool name (e.g., 'Bash', 'mcp__server__action').
        tool_use_id: Unique tool call ID for correlation.
        parent_context: Explicit parent context.  When provided the tool span
            becomes a child of the span stored in *parent_context*.  When
            ``None`` OTel auto-parents under the current span (works in unit
            tests but not in real SDK hook callbacks which run in a separate
            async context).

    Returns:
        A started span (must be ended by caller).
    """
    span_name = f"{OPERATION_EXECUTE_TOOL} {tool_name}"
    tool_type = derive_tool_type(tool_name)

    attributes: dict[str, str] = {
        GEN_AI_OPERATION_NAME: OPERATION_EXECUTE_TOOL,
        GEN_AI_PROVIDER_NAME: SYSTEM_ANTHROPIC,
        GEN_AI_TOOL_NAME: tool_name,
        GEN_AI_TOOL_CALL_ID: tool_use_id,
        GEN_AI_TOOL_TYPE: tool_type,
    }

    return tracer.start_span(
        name=span_name,
        kind=SpanKind.INTERNAL,
        attributes=attributes,
        context=parent_context,
    )


def set_tool_error_attributes(span: Span, error_message: str) -> None:
    """Set error.type and ERROR status on a tool span.

    Unlike set_error_attributes (which takes an exception), this takes a raw error
    string from the SDK hook, since tool errors are strings, not exceptions.

    Args:
        span: The tool span to annotate.
        error_message: Raw error string from PostToolUseFailure.
    """
    span.set_attribute(ERROR_TYPE, error_message)
    span.set_status(StatusCode.ERROR, error_message)
