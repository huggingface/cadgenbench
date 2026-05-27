"""CADGenBench evaluation: metric categories and orchestrator.

The benchmark scores a candidate STEP against a ground-truth STEP along
three orthogonal axes, each with its own module:

- :mod:`.shape_similarity`: point-cloud F1, volume IoU, and
  feature-edge F1 against the ground truth.
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
- :mod:`.feature_edges`: dihedral feature-edge extraction + overlay.
- :mod:`.booleans`:      mesh Boolean ops (``manifold3d`` kernel).
"""
