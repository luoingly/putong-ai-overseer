import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from judger.models import SubmissionResult
from overseer.agent import AgentResult

logger = logging.getLogger(__name__)


class Recorder:
    def __init__(self, output_dir: Path | None = None):
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("data/records") / timestamp
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_output_dir(self) -> Path:
        return self.output_dir

    def _build_base_record(
        self,
        model_name: str,
        problem_id: str,
        language: str,
        agent_result: AgentResult,
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "model": model_name,
            "problem": problem_id,
            "language": language,
            "status": agent_result.status.value if agent_result.status else None,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "turn_count": agent_result.turn_count,
            "conversation": agent_result.conversation,
        }

        if agent_result.token_usage:
            record["token_usage"] = agent_result.token_usage.to_dict()

        if agent_result.error:
            record["error"] = agent_result.error

        return record

    def save_intermediate(
        self,
        model_name: str,
        problem_id: str,
        language: str,
        agent_result: AgentResult,
        elapsed_seconds: float,
        turn_index: int | None = None,
    ) -> Path:
        """Save intermediate state after each turn. Can be called multiple times."""
        filename = f"{model_name}__{problem_id}.json"
        filepath = self.output_dir / filename

        record = self._build_base_record(
            model_name, problem_id, language, agent_result, elapsed_seconds
        )
        # Override status logic for intermediate
        record["status"] = agent_result.status.value if agent_result.status else "in_progress"
        record["in_progress"] = True

        if agent_result.code:
            record["extracted_code"] = (
                agent_result.code[:200] + "..."
                if len(agent_result.code) > 200
                else agent_result.code
            )

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        logger.debug("Intermediate record saved to %s (turn %s)", filepath, turn_index or "?")
        return filepath

    def finalize(
        self,
        model_name: str,
        problem_id: str,
    ) -> None:
        """Mark the record as complete (remove in_progress flag)."""
        filename = f"{model_name}__{problem_id}.json"
        filepath = self.output_dir / filename
        if not filepath.exists():
            return
        with open(filepath, encoding="utf-8") as f:
            record = json.load(f)
        record.pop("in_progress", None)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    async def save(
        self,
        model_name: str,
        problem_id: str,
        language: str,
        agent_result: AgentResult,
        judge_result: SubmissionResult | None,
        elapsed_seconds: float,
    ) -> Path:
        record = self._build_base_record(
            model_name, problem_id, language, agent_result, elapsed_seconds
        )

        if judge_result:
            record["judge_detail"] = {
                "status": judge_result.judge.name,
                "time": judge_result.time,
                "memory": judge_result.memory,
                "error": judge_result.error or None,
                "testcases": [
                    {
                        "uuid": tc.uuid,
                        "status": tc.judge.name,
                        "time": tc.time,
                        "memory": tc.memory,
                    }
                    for tc in judge_result.testcases
                ],
            }

        filename = f"{model_name}__{problem_id}.json"
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        logger.info("Record saved to %s", filepath)

        return filepath
