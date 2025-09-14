"""Request/response transformers for OpenAI <-> Claude API compatibility."""

import json
import uuid
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import time

from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatMessage,
    ClaudeRequest,
    ClaudeResponse,
    ClaudeMessage,
    ClaudeTool,
    ChatCompletionResponseChoice,
    ChatCompletionStreamChoice,
    Usage,
    ClaudeUsage,
    ToolCall,
    FunctionCall
)


class RequestTransformer:
    """Transforms OpenAI-compatible requests to Claude API format."""

    @staticmethod
    def openai_to_claude(openai_request: ChatCompletionRequest) -> ClaudeRequest:
        """Convert OpenAI chat completion request to Claude format."""

        # Extract system message if present
        system_content = None
        claude_messages = []

        for message in openai_request.messages:
            if message.role == "system":
                system_content = message.content
            elif message.role in ["user", "assistant"]:
                # Convert content to Claude format
                if isinstance(message.content, str):
                    content = message.content
                elif isinstance(message.content, list):
                    # Handle multi-modal content
                    content = RequestTransformer._convert_multimodal_content(message.content)
                else:
                    content = str(message.content) if message.content else ""

                # Handle tool calls in assistant messages
                if message.role == "assistant" and message.tool_calls:
                    content = [{"type": "text", "text": content or ""}]
                    for tool_call in message.tool_calls:
                        content.append({
                            "type": "tool_use",
                            "id": tool_call.id,
                            "name": tool_call.function.name,
                            "input": json.loads(tool_call.function.arguments)
                        })

                # Handle tool responses
                elif message.role == "tool":
                    # Convert tool response to user message format
                    claude_messages.append(ClaudeMessage(
                        role="user",
                        content=[{
                            "type": "tool_result",
                            "tool_use_id": message.tool_call_id,
                            "content": message.content
                        }]
                    ))
                    continue

                claude_messages.append(ClaudeMessage(
                    role=message.role,
                    content=content
                ))

        # Convert tools to Claude format
        claude_tools = None
        if openai_request.tools:
            claude_tools = []
            for tool in openai_request.tools:
                if tool.type == "function":
                    claude_tools.append(ClaudeTool(
                        name=tool.function.name,
                        description=tool.function.description or "",
                        input_schema=tool.function.parameters or {}
                    ))

        # Handle legacy functions parameter
        elif openai_request.functions:
            claude_tools = []
            for func in openai_request.functions:
                claude_tools.append(ClaudeTool(
                    name=func.name,
                    description=func.description or "",
                    input_schema=func.parameters or {}
                ))

        # Convert tool_choice
        tool_choice = None
        if openai_request.tool_choice:
            if isinstance(openai_request.tool_choice, str):
                if openai_request.tool_choice == "auto":
                    tool_choice = {"type": "auto"}
                elif openai_request.tool_choice == "none":
                    tool_choice = None
            elif isinstance(openai_request.tool_choice, dict):
                if "function" in openai_request.tool_choice:
                    tool_choice = {
                        "type": "tool",
                        "name": openai_request.tool_choice["function"]["name"]
                    }

        return ClaudeRequest(
            model=openai_request.model,
            max_tokens=openai_request.max_tokens or 4096,
            messages=claude_messages,
            system=system_content,
            temperature=openai_request.temperature,
            top_p=openai_request.top_p,
            tools=claude_tools,
            tool_choice=tool_choice,
            stream=openai_request.stream or False
        )

    @staticmethod
    def _convert_multimodal_content(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI multimodal content to Claude format."""
        claude_content = []

        for item in content:
            if item.get("type") == "text":
                claude_content.append({
                    "type": "text",
                    "text": item.get("text", "")
                })
            elif item.get("type") == "image_url":
                image_url = item.get("image_url", {})
                claude_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64" if image_url.get("url", "").startswith("data:") else "url",
                        "media_type": "image/jpeg",  # Default, should be detected
                        "data": image_url.get("url", "")
                    }
                })

        return claude_content


class ResponseTransformer:
    """Transforms Claude API responses to OpenAI-compatible format."""

    @staticmethod
    def claude_to_openai(
        claude_response: ClaudeResponse,
        original_request: ChatCompletionRequest,
        request_id: Optional[str] = None
    ) -> ChatCompletionResponse:
        """Convert Claude response to OpenAI chat completion format."""

        if not request_id:
            request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        # Convert content
        message_content = ""
        tool_calls = []

        for content_item in claude_response.content:
            if content_item.get("type") == "text":
                message_content += content_item.get("text", "")
            elif content_item.get("type") == "tool_use":
                tool_calls.append(ToolCall(
                    id=content_item.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    type="function",
                    function=FunctionCall(
                        name=content_item.get("name", ""),
                        arguments=json.dumps(content_item.get("input", {}))
                    )
                ))

        # Create response message
        response_message = ChatMessage(
            role="assistant",
            content=message_content or None,
            tool_calls=tool_calls if tool_calls else None
        )

        # Determine finish reason
        finish_reason = None
        if claude_response.stop_reason == "end_turn":
            finish_reason = "stop"
        elif claude_response.stop_reason == "max_tokens":
            finish_reason = "length"
        elif claude_response.stop_reason == "tool_use":
            finish_reason = "tool_calls"
        elif claude_response.stop_reason == "stop_sequence":
            finish_reason = "stop"

        # Create usage information
        usage = Usage(
            prompt_tokens=claude_response.usage.input_tokens,
            completion_tokens=claude_response.usage.output_tokens,
            total_tokens=claude_response.usage.input_tokens + claude_response.usage.output_tokens
        )

        return ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=original_request.model,
            choices=[ChatCompletionResponseChoice(
                index=0,
                message=response_message,
                finish_reason=finish_reason
            )],
            usage=usage
        )

    @staticmethod
    def claude_stream_to_openai(
        claude_chunk: Dict[str, Any],
        original_request: ChatCompletionRequest,
        request_id: Optional[str] = None
    ) -> Optional[ChatCompletionStreamResponse]:
        """Convert Claude streaming chunk to OpenAI streaming format."""

        if not request_id:
            request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        chunk_type = claude_chunk.get("type")

        if chunk_type == "message_start":
            # Initial chunk
            delta = ChatMessage(role="assistant", content="")
            return ChatCompletionStreamResponse(
                id=request_id,
                created=int(time.time()),
                model=original_request.model,
                choices=[ChatCompletionStreamChoice(
                    index=0,
                    delta=delta,
                    finish_reason=None
                )]
            )

        elif chunk_type == "content_block_start":
            # Start of content block - no delta needed
            return None

        elif chunk_type == "content_block_delta":
            # Content delta
            delta_data = claude_chunk.get("delta", {})
            delta_type = delta_data.get("type")

            if delta_type == "text_delta":
                delta = ChatMessage(role="assistant", content=delta_data.get("text", ""))
                return ChatCompletionStreamResponse(
                    id=request_id,
                    created=int(time.time()),
                    model=original_request.model,
                    choices=[ChatCompletionStreamChoice(
                        index=0,
                        delta=delta,
                        finish_reason=None
                    )]
                )

        elif chunk_type == "content_block_stop":
            # End of content block
            return None

        elif chunk_type == "message_delta":
            # Message metadata delta (like usage)
            return None

        elif chunk_type == "message_stop":
            # Final chunk
            delta = ChatMessage(role="assistant")
            finish_reason = "stop"  # Default finish reason

            return ChatCompletionStreamResponse(
                id=request_id,
                created=int(time.time()),
                model=original_request.model,
                choices=[ChatCompletionStreamChoice(
                    index=0,
                    delta=delta,
                    finish_reason=finish_reason
                )]
            )

        return None


class ModelMapper:
    """Maps between OpenAI and Claude model names."""

    @staticmethod
    def map_openai_to_claude(openai_model: str, model_mapping: Dict[str, str]) -> str:
        """Map OpenAI model name to Claude model name."""
        return model_mapping.get(openai_model, openai_model)

    @staticmethod
    def map_claude_to_openai(claude_model: str, model_mapping: Dict[str, str]) -> str:
        """Map Claude model name back to requested OpenAI model name."""
        # Reverse lookup in the mapping
        for openai_model, mapped_claude in model_mapping.items():
            if mapped_claude == claude_model:
                return openai_model
        return claude_model