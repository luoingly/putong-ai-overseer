import json
import logging
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


# --- Models ---


@dataclass
class Message:
    role: str  # system / user / assistant / tool
    content: str | None = None
    tool_calls: list["ToolCall"] | None = None
    tool_call_id: str | None = None
    reasoning_content: str | None = None


@dataclass
class ToolCallFunction:
    name: str
    arguments: dict[str, Any]  # parsed JSON, not raw string


@dataclass
class ToolCall:
    id: str
    type: str = "function"
    function: ToolCallFunction | None = field(default=None)


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: dict[str, int] | None = None
    completion_tokens_details: dict[str, int] | None = None

    @staticmethod
    def _merge_details(
        left: dict[str, int] | None, right: dict[str, int] | None
    ) -> dict[str, int] | None:
        if left is None and right is None:
            return None
        merged: dict[str, int] = {}
        for details in (left, right):
            if details is None:
                continue
            for key, value in details.items():
                merged[key] = merged.get(key, 0) + (value or 0)
        return merged

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            prompt_tokens_details=self._merge_details(
                self.prompt_tokens_details, other.prompt_tokens_details
            ),
            completion_tokens_details=self._merge_details(
                self.completion_tokens_details, other.completion_tokens_details
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "prompt_tokens_details": self.prompt_tokens_details,
            "completion_tokens_details": self.completion_tokens_details,
        }


@dataclass
class AIResponse:
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: Usage | None = None
    finish_reason: str | None = None
    model: str | None = None
    reasoning_content: str | None = None


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    temperature: float = 0.2
    max_tokens: int = 4096
    thinking: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProviderConfig":
        return cls(
            name=data["name"],
            base_url=data["base_url"],
            api_key=data["api_key"],
            temperature=data.get("temperature", 0.2),
            max_tokens=data.get("max_tokens", 4096),
            thinking=data.get("thinking", False),
        )


# --- Provider ---


def _message_to_openai(msg: Message) -> dict[str, Any]:
    d: dict[str, Any] = {"role": msg.role}
    if msg.content is not None:
        d["content"] = msg.content
    if msg.reasoning_content is not None:
        d["reasoning_content"] = msg.reasoning_content
    if msg.tool_calls is not None:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": json.dumps(tc.function.arguments, ensure_ascii=False),
                },
            }
            for tc in msg.tool_calls
        ]
    if msg.tool_call_id is not None:
        d["tool_call_id"] = msg.tool_call_id
    return d


def _tooldef_to_openai(td: ToolDef) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": td.name,
            "description": td.description,
            "parameters": td.parameters,
        },
    }


def _parse_response(resp: Any) -> AIResponse:
    choice = resp.choices[0]
    message = choice.message

    tool_calls = None
    if message.tool_calls:
        tool_calls = []
        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {"_raw": tc.function.arguments}
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    type=tc.type,
                    function=ToolCallFunction(name=tc.function.name, arguments=args),
                )
            )

    usage = None
    if resp.usage:
        usage = Usage(
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            total_tokens=resp.usage.total_tokens,
            prompt_tokens_details=(
                resp.usage.prompt_tokens_details.model_dump()
                if resp.usage.prompt_tokens_details
                else None
            ),
            completion_tokens_details=(
                resp.usage.completion_tokens_details.model_dump()
                if resp.usage.completion_tokens_details
                else None
            ),
        )

    reasoning_content = getattr(message, "reasoning_content", None)

    return AIResponse(
        content=message.content,
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=choice.finish_reason,
        model=resp.model,
        reasoning_content=reasoning_content,
    )


class AIProvider:
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
    ) -> AIResponse:
        openai_messages = [_message_to_openai(m) for m in messages]
        kwargs: dict[str, Any] = {
            "model": self.config.name,
            "messages": openai_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "extra_body": {"thinking": {"type": "enabled" if self.config.thinking else "disabled"}},
        }
        if tools:
            kwargs["tools"] = [_tooldef_to_openai(t) for t in tools]

        logger.debug(
            "Sending request to %s (model=%s, messages=%d)",
            self.config.base_url,
            self.config.name,
            len(messages),
        )
        resp = await self.client.chat.completions.create(**kwargs)
        result = _parse_response(resp)
        logger.debug(
            "Response: finish_reason=%s, tool_calls=%d, usage=%s",
            result.finish_reason,
            len(result.tool_calls or []),
            result.usage,
        )
        return result
