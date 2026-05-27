"""CADGenBench reference baseline, an iterative LLM agent.

The agent reads a task description, writes build123d Python in a loop,
gets a render + validity check of its output each turn, and signals
completion with ``[DONE]``.
"""

from cadgenbench.baseline.agent import run_agent
from cadgenbench.baseline.types import (
    AgentConfig,
    AgentResult,
    CodeExecution,
    TurnRecord,
    save_conversation,
)

__all__ = [
    "run_agent",
    "AgentConfig",
    "AgentResult",
    "CodeExecution",
    "TurnRecord",
    "save_conversation",
]
