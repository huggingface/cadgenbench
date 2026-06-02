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

"""Default reference-baseline model trio + labels (dependency-light).

These live here rather than in ``baseline/compare_llms.py`` so callers that
lack the heavy ``[baseline]`` extras (litellm / python-dotenv) -- e.g. the HF
Jobs orchestrator environment -- can still import the canonical model list and
display labels. ``compare_llms`` re-exports these names, and the orchestrator's
``run_baselines`` wrapper imports them from here, so the trio never diverges
between the local comparison command and the fan-out tooling.

Current flagship from each of Anthropic, Google, OpenAI as of May 2026.
"""
from __future__ import annotations

DEFAULT_COMPARE_MODELS: tuple[str, ...] = (
    "anthropic/claude-opus-4-7",
    "gemini/gemini-3.1-pro-preview",
    "openai/gpt-5.5",
)
DEFAULT_COMPARE_LABELS: tuple[str, ...] = (
    "Claude Opus 4.7",
    "Gemini 3.1 Pro",
    "GPT-5.5",
)
