import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from overseer.problems.models import Problem
from overseer.provider import AIResponse, Message, Usage
from overseer.tools.definitions import ALL_TOOLS
from overseer.tools.executor import ToolExecutor

logger = logging.getLogger(__name__)

# Callback type: called after each turn with (turn_index, conversation, usage, last_code)
TurnCallback = Callable[[int, list[dict[str, Any]], Usage, str | None], None]


class AgentStatus(StrEnum):
    Completed = "completed"
    Failed = "failed"
    Timeout = "timeout"


@dataclass
class AgentResult:
    status: AgentStatus
    code: str | None = None
    language: str | None = None
    token_usage: Usage | None = None
    turn_count: int = 0
    conversation: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


SIMPLE_SYSTEM_PROMPT = """\
你是一名竞赛选手。请解答给定的算法题目。

将你的解答放在带有语言标识符的代码块中输出，例如：
```python
# 你的代码
```

只输出一个代码块，不要在代码块之后输出其他内容。
"""

TOOL_SYSTEM_PROMPT = """\
你是一名竞赛选手，正在解答算法题目。\
你可以使用工具来阅读题目、运行代码和提交解答。
"""


def _extract_code(text: str | None, language_hint: str | None = None) -> str | None:
    if not text:
        return None

    if language_hint:
        pattern = rf"```{language_hint}\s*\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    pattern = r"```(?:\w+)?\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def _message_to_record(msg: Message) -> dict[str, Any]:
    record: dict[str, Any] = {"role": msg.role}
    if msg.content is not None:
        record["content"] = msg.content
    if msg.reasoning_content is not None:
        record["reasoning_content"] = msg.reasoning_content
    return record


def _response_to_record(resp: AIResponse, model: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {"role": "assistant", "model": model}
    if resp.content:
        record["content"] = resp.content
    if resp.reasoning_content:
        record["reasoning_content"] = resp.reasoning_content
    if resp.tool_calls:
        record["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in resp.tool_calls
        ]
    if resp.usage:
        record["usage"] = resp.usage.to_dict()
    return record


class SimpleAgent:
    def __init__(
        self,
        language_name: str,
        max_turns: int = 5,
        on_turn_complete: TurnCallback | None = None,
    ):
        self.language_name = language_name
        self.max_turns = max_turns
        self.on_turn_complete = on_turn_complete

    async def solve(self, problem: Problem, provider: Any) -> AgentResult:
        statement = problem.read_statement()
        user_content = (
            f"请使用 {self.language_name} 解答以下题目。\n\n"
            f"{statement}\n\n"
            f"请将解答放在 ```{self.language_name} 代码块中。"
        )
        messages = [
            Message(role="system", content=SIMPLE_SYSTEM_PROMPT),
            Message(role="user", content=user_content),
        ]
        conversation = [_message_to_record(m) for m in messages]

        total_usage = Usage()
        response: AIResponse | None = None

        for turn in range(self.max_turns):
            try:
                response = await provider.complete(messages)
            except Exception as e:
                logger.exception("SimpleAgent: provider call failed at turn %d", turn + 1)
                return AgentResult(
                    status=AgentStatus.Failed,
                    error=str(e),
                    conversation=conversation,
                    token_usage=total_usage,
                    turn_count=turn + 1,
                )

            if response.usage:
                total_usage = total_usage + response.usage

            conversation.append(_response_to_record(response, provider.config.name))

            assistant_msg = Message(
                role="assistant",
                content=response.content,
                reasoning_content=response.reasoning_content,
            )
            messages.append(assistant_msg)

            # 检查是否被截断
            if response.finish_reason == "length":
                logger.warning(
                    "SimpleAgent: response truncated (max tokens) at turn %d, continuing...",
                    turn + 1,
                )
                # 让模型继续生成
                continue

            # 尝试提取代码
            code = _extract_code(response.content, self.language_name) or _extract_code(
                response.content
            )

            if code is not None:
                if self.on_turn_complete:
                    self.on_turn_complete(turn, conversation, total_usage, code)
                return AgentResult(
                    status=AgentStatus.Completed,
                    code=code,
                    language=self.language_name,
                    token_usage=total_usage,
                    turn_count=turn + 1,
                    conversation=conversation,
                )

            # 如果没有找到代码块，且模型没有截断，可能是格式错误
            if response.finish_reason != "length":
                logger.warning("SimpleAgent: failed to extract code at turn %d", turn + 1)
                # 让模型重新输出
                messages.append(
                    Message(
                        role="user",
                        content=f"请在 ```{self.language_name} 代码块中输出完整代码。",
                    )
                )
                conversation.append(
                    {
                        "role": "user",
                        "content": f"请在 ```{self.language_name} 代码块中输出完整代码。",
                    }
                )

        # 达到最大轮次或最后一次响应被截断
        final_code = None
        if response and response.content:
            final_code = _extract_code(response.content, self.language_name) or _extract_code(
                response.content
            )

        if response and response.finish_reason == "length":
            logger.warning("SimpleAgent: last response truncated, may have incomplete output")

        if self.on_turn_complete:
            self.on_turn_complete(self.max_turns - 1, conversation, total_usage, final_code)

        return AgentResult(
            status=AgentStatus.Completed if final_code else AgentStatus.Failed,
            code=final_code,
            language=self.language_name,
            token_usage=total_usage,
            turn_count=self.max_turns,
            conversation=conversation,
            error=None if final_code else "Failed to extract code after multiple attempts",
        )


class ToolAgent:
    def __init__(
        self,
        language_name: str,
        max_turns: int = 10,
        tool_executor: ToolExecutor | None = None,
        on_turn_complete: TurnCallback | None = None,
    ):
        self.language_name = language_name
        self.max_turns = max_turns
        self.tool_executor = tool_executor
        self.on_turn_complete = on_turn_complete

    async def solve(
        self,
        problem: Problem,
        provider: Any,  # AIProvider
    ) -> AgentResult:
        if not self.tool_executor:
            return AgentResult(
                status=AgentStatus.Failed,
                error="ToolAgent requires a tool_executor",
                conversation=[],
            )

        system_prompt = TOOL_SYSTEM_PROMPT.format(language=self.language_name)
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content="请开始，阅读题目并解答。"),
        ]
        conversation = [_message_to_record(m) for m in messages]

        total_usage = Usage()
        last_code: str | None = None
        response: AIResponse | None = None

        for turn in range(self.max_turns):
            try:
                response = await provider.complete(messages, tools=ALL_TOOLS)
            except Exception as e:
                logger.exception("ToolAgent: provider call failed at turn %d", turn + 1)
                return AgentResult(
                    status=AgentStatus.Failed,
                    code=last_code,
                    language=self.language_name,
                    token_usage=total_usage,
                    turn_count=turn + 1,
                    conversation=conversation,
                    error=str(e),
                )

            if response.usage:
                total_usage = total_usage + response.usage

            conversation.append(_response_to_record(response, provider.config.name))

            assistant_msg = Message(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls,
                reasoning_content=response.reasoning_content,
            )
            messages.append(assistant_msg)

            # 检查是否被截断（达到最大输出 Token 限制）
            if response.finish_reason == "length":
                logger.warning(
                    "ToolAgent: response truncated (max tokens) at turn %d, continuing...",
                    turn + 1,
                )
                # 不执行不完整的工具调用，直接继续对话让模型接着生成
                if self.on_turn_complete:
                    self.on_turn_complete(turn, conversation, total_usage, last_code)
                continue

            if not response.tool_calls:
                logger.debug("ToolAgent: no tool calls at turn %d, finishing", turn + 1)
                if self.on_turn_complete:
                    self.on_turn_complete(turn, conversation, total_usage, last_code)
                break

            for tc in response.tool_calls:
                args = tc.function.arguments
                if not isinstance(args, dict):
                    args = {}

                logger.info("ToolAgent: executing tool '%s' with args: %s", tc.function.name, args)
                result_str = await self.tool_executor.execute(tc.function.name, args)

                conversation.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "tool_name": tc.function.name,
                        "content": result_str,
                    }
                )

                tool_msg = Message(
                    role="tool",
                    content=result_str,
                    tool_call_id=tc.id,
                )
                messages.append(tool_msg)

                if tc.function.name == "submit_code" and "code" in args:
                    last_code = args["code"]
                    if "评测结果：Accepted" in result_str:
                        logger.info("ToolAgent: Accepted, stopping early")
                        if self.on_turn_complete:
                            self.on_turn_complete(turn, conversation, total_usage, last_code)
                        return AgentResult(
                            status=AgentStatus.Completed,
                            code=last_code,
                            language=self.language_name,
                            token_usage=total_usage,
                            turn_count=turn + 1,
                            conversation=conversation,
                        )

            # 每轮结束后调用回调（保存中间状态）
            if self.on_turn_complete:
                self.on_turn_complete(turn, conversation, total_usage, last_code)

        # 检查最后一次响应是否被截断
        if response and response.finish_reason == "length":
            logger.warning(
                "ToolAgent: last response truncated (max tokens), may have incomplete output"
            )

        final_code = last_code
        if not final_code and response and response.content:
            final_code = _extract_code(response.content, self.language_name) or _extract_code(
                response.content
            )

        turn_count = len([m for m in conversation if m.get("role") == "assistant"])

        return AgentResult(
            status=AgentStatus.Completed if final_code else AgentStatus.Failed,
            code=final_code,
            language=self.language_name,
            token_usage=total_usage,
            turn_count=turn_count,
            conversation=conversation,
        )
