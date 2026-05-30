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

"""CADGenBench, a benchmark for AI-driven CAD generation and editing.

The benchmark itself is system-agnostic; the included reference
baseline happens to be an LLM agent.

Three top-level subpackages:

- :mod:`cadgenbench.common`  , shared helpers (renderer, validity, mesh).
- :mod:`cadgenbench.eval`    , benchmark scoring + report generation.
- :mod:`cadgenbench.baseline`, the reference LLM agent (build123d).
"""

__version__ = "0.1.0"
