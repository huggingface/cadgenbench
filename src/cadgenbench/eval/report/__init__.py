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

"""HTML report generators for cadgenbench result directories.

Two entry points, both wired into the ``cadgenbench report`` CLI:

- :mod:`.single_run`   -- inspect one run's per-fixture results.
- :mod:`.compare_runs` -- compare 2+ runs side by side.

Generated HTML files are self-contained: every image is embedded as a
base64 ``data:`` URI so the file can be opened anywhere, no asset
dependencies.
"""
