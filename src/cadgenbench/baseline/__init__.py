# Copyright 2026 Hugging Face
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
