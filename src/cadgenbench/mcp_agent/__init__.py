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

"""MCP-agent strategy for CADGenBench.

Uses the build123d-mcp server's native tool-calling interface instead of
code-block extraction. The LLM calls execute(), render_view(), measure(),
export(), and signal_done() as tools; the benchmark connects to a running
build123d-mcp server via its stdio MCP protocol.
"""

__all__ = ["run_mcp_agent", "McpAgentConfig", "McpAgentResult"]


def __getattr__(name: str):  # noqa: N807
    if name == "run_mcp_agent":
        from cadgenbench.mcp_agent.agent import run_mcp_agent
        return run_mcp_agent
    if name == "McpAgentConfig":
        from cadgenbench.mcp_agent.types import McpAgentConfig
        return McpAgentConfig
    if name == "McpAgentResult":
        from cadgenbench.mcp_agent.types import McpAgentResult
        return McpAgentResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
