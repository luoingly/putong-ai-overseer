import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

from judger.client import SandboxClient
from judger.models import Language
from overseer.agent import AgentResult, AgentStatus, SimpleAgent, ToolAgent
from overseer.problems.registry import ProblemRegistry
from overseer.provider import AIProvider, ProviderConfig
from overseer.recorder import Recorder
from overseer.tools.executor import ToolExecutor

console = Console()

DEFAULT_SANDBOX = "http://localhost:5050"
DEFAULT_PROBLEMS_DIR = "data/problems"
DEFAULT_CONFIG = "data/models.yaml"
DEFAULT_MAX_TURNS = 100

LANGUAGE_MAP = {
    "c": Language.C,
    "cpp11": Language.Cpp11,
    "cpp17": Language.Cpp17,
    "java": Language.Java,
    "python": Language.Python,
    "pypy": Language.PyPy,
}


def parse_args():
    parser = argparse.ArgumentParser(description="AI Algorithm Problem Evaluation Tool")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model name (from models config). Can specify multiple times.",
    )
    parser.add_argument(
        "--problem",
        action="append",
        required=True,
        help="Problem ID. Can specify multiple times.",
    )
    parser.add_argument(
        "--language",
        required=True,
        choices=list(LANGUAGE_MAP.keys()),
        help="Programming language for the solution.",
    )
    parser.add_argument(
        "--agent",
        default="tool",
        choices=["simple", "tool"],
        help="Agent type: simple (single-turn) or tool (multi-turn, default: tool).",
    )
    parser.add_argument(
        "--sandbox",
        default=DEFAULT_SANDBOX,
        help=f"Sandbox endpoint (default: {DEFAULT_SANDBOX}).",
    )
    parser.add_argument(
        "--problems-dir",
        default=DEFAULT_PROBLEMS_DIR,
        help=f"Problems directory (default: {DEFAULT_PROBLEMS_DIR}).",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Models config file (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help=f"Max turns for tool agent (default: {DEFAULT_MAX_TURNS}).",
    )
    return parser.parse_args()


def load_model_configs(config_path: str) -> dict[str, ProviderConfig]:
    path = Path(config_path)
    if not path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not data.get("models"):
        console.print(f"[red]No models found in {config_path}[/red]")
        sys.exit(1)
    configs = {}
    for m in data.get("models", []):
        configs[m["name"]] = ProviderConfig.from_dict(m)
    return configs


class RunContext:
    def __init__(
        self,
        start_time: float,
        recorder: Recorder,
        model_name: str,
        problem_id: str,
        language_name: str,
    ):
        self.start_time = start_time
        self.recorder = recorder
        self.model_name = model_name
        self.problem_id = problem_id
        self.language_name = language_name

    def on_turn_complete(
        self,
        turn_index: int,
        conversation: list[dict[str, Any]],
        usage: Any,
        last_code: str | None,
    ) -> None:
        elapsed = time.time() - self.start_time
        intermediate = AgentResult(
            status=AgentStatus.Completed if last_code else AgentStatus.Failed,
            code=last_code,
            language=self.language_name,
            token_usage=usage,
            turn_count=turn_index + 1,
            conversation=conversation,
        )
        self.recorder.save_intermediate(
            model_name=self.model_name,
            problem_id=self.problem_id,
            language=self.language_name,
            agent_result=intermediate,
            elapsed_seconds=elapsed,
            turn_index=turn_index,
        )

        token_info = f" | Tokens: {usage.total_tokens:,}" if usage and usage.total_tokens else ""
        console.print(
            f"\n[bold cyan]--- Turn {turn_index + 1} complete ---[/bold cyan]{token_info}"
        )

        if conversation:
            last_msg = conversation[-1]
            if last_msg.get("role") == "assistant":
                content = last_msg.get("content") or ""
                reasoning = last_msg.get("reasoning_content") or ""
                if reasoning:
                    preview = reasoning[:200].replace("\n", " ")
                    console.print(f"  [dim]Reasoning: {preview}...[/dim]")
                if content:
                    preview = content[:200].replace("\n", " ")
                    console.print(f"  [dim]Content: {preview}...[/dim]")
                tool_calls = last_msg.get("tool_calls") or []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    console.print(f"  [yellow]Tool:[/yellow] {func.get('name', '?')}")


async def run_one(
    model_name: str,
    problem_id: str,
    language: Language,
    language_name: str,
    agent_type: str,
    max_turns: int,
    provider: AIProvider,
    sandbox_client: SandboxClient,
    problem_registry: ProblemRegistry,
    recorder: Recorder,
) -> None:
    console.print(f"\n[bold]{'=' * 60}[/bold]")
    console.print(
        f"[bold]Model:[/bold] {model_name} | "
        f"[bold]Problem:[/bold] {problem_id} | "
        f"[bold]Language:[/bold] {language_name}"
    )
    console.print(f"[bold]Agent:[/bold] {agent_type}")

    problem = problem_registry.get(problem_id)
    start_time = time.time()
    ctx = RunContext(start_time, recorder, model_name, problem_id, language_name)

    if agent_type == "simple":
        agent = SimpleAgent(
            language_name=language_name,
            max_turns=max_turns,
            on_turn_complete=ctx.on_turn_complete,
        )
    else:
        tool_executor = ToolExecutor(
            problem=problem,
            sandbox_client=sandbox_client,
            language=language,
        )
        agent = ToolAgent(
            language_name=language_name,
            max_turns=max_turns,
            tool_executor=tool_executor,
            on_turn_complete=ctx.on_turn_complete,
        )

    start = time.time()
    agent_result: AgentResult = await agent.solve(problem, provider)
    elapsed = time.time() - start

    # 清除 in_progress 标记
    recorder.finalize(model_name=model_name, problem_id=problem_id)

    judge_result = None
    if agent_result.status == AgentStatus.Completed and agent_result.code:
        from judger.judger import Judger
        from overseer.tools.executor import build_submission

        submission = build_submission(problem, agent_result.code, language)
        judger = Judger(client=sandbox_client, submission=submission)
        judge_result = await judger.get_result()

    filepath = await recorder.save(
        model_name=model_name,
        problem_id=problem_id,
        language=language_name,
        agent_result=agent_result,
        judge_result=judge_result,
        elapsed_seconds=elapsed,
    )

    _print_result(agent_result, judge_result, elapsed, filepath)


def _print_result(
    agent_result: AgentResult,
    judge_result,
    elapsed: float,
    filepath: Path,
) -> None:
    if agent_result.error and agent_result.status == AgentStatus.Failed:
        console.print(f"[red]Agent failed: {agent_result.error}[/red]")

    token_info = ""
    if agent_result.token_usage:
        token_info = f" | Tokens: {agent_result.token_usage.total_tokens:,}"

    console.print(f"Turns: {agent_result.turn_count}{token_info} | Time: {elapsed:.1f}s")

    if judge_result:
        status_color = "green" if judge_result.judge.name == "Accepted" else "red"
        console.print(f"\nResult: [{status_color}]{judge_result.judge.name}[/{status_color}]")
        if judge_result.error:
            console.print(f"[red]Error: {judge_result.error}[/red]")
        console.print(f"Time: {judge_result.time}ms | Memory: {judge_result.memory}KB\n")

        passed = sum(1 for tc in judge_result.testcases if tc.judge.name == "Accepted")
        total = len(judge_result.testcases)

        for tc in judge_result.testcases:
            if tc.judge.name == "Accepted":
                icon = "[green]✓[/green]"
            elif tc.judge.name == "Skipped":
                icon = "[dim]—[/dim]"
            else:
                icon = "[red]✗[/red]"
            console.print(f"  {tc.uuid}: {icon} {tc.judge.name} ({tc.time}ms, {tc.memory}KB)")

        console.print(f"\nTotal: {passed}/{total} passed")
    elif agent_result.code:
        console.print(
            "[yellow]Code generated but not judged (agent completed without submission)[/yellow]"
        )

    console.print(f"\n[dim]Output saved to: {filepath}[/dim]")


async def async_main():
    load_dotenv()
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    model_configs = load_model_configs(args.config)

    language = LANGUAGE_MAP[args.language]

    problem_registry = ProblemRegistry(Path(args.problems_dir))
    console.print(f"Loaded {len(problem_registry.ids)} problems: {problem_registry.ids}")

    recorder = Recorder()

    async with SandboxClient(args.sandbox) as sandbox_client:
        for model_name in args.model:
            if model_name not in model_configs:
                available = list(model_configs.keys())
                console.print(
                    f"[red]Model '{model_name}' not found in config. Available: {available}[/red]"
                )
                sys.exit(1)

            config = model_configs[model_name]
            provider = AIProvider(config)

            for problem_id in args.problem:
                await run_one(
                    model_name=model_name,
                    problem_id=problem_id,
                    language=language,
                    language_name=args.language,
                    agent_type=args.agent,
                    max_turns=args.max_turns,
                    provider=provider,
                    sandbox_client=sandbox_client,
                    problem_registry=problem_registry,
                    recorder=recorder,
                )

    console.print(f"\n[bold]All done.[/bold] Results in: {recorder.output_dir}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
