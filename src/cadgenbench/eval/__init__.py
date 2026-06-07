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

"""CADGenBench evaluation: metric categories and orchestrator.

The benchmark scores a candidate STEP against a ground-truth STEP along
three orthogonal axes, each with its own module:

- :mod:`.shape_similarity`: surface distance F1 and volume IoU against the
  ground truth.
- :mod:`.interface_match`:  keep-in / keep-out region matching against
  authored sub-volumes.
- :mod:`.topo_match`:       Betti-number agreement (b0, b1, b2)
  computed on the tessellated boundary.

Plus the validity gate in :mod:`cadgenbench.common.validity`; invalid
candidates short-circuit to zero on every metric.

:mod:`.evaluate` is the orchestrator: it runs every applicable metric
against a result directory and persists the merged scores to
``result.json``.

Supporting modules:

- :mod:`.alignment`:     rigid-pose alignment of candidate to GT.
- :mod:`.sampling`:      surface point sampling.
- :mod:`.booleans`:      mesh Boolean ops (``manifold3d`` kernel).
"""
