"""CADGenBench, a benchmark for AI-driven CAD generation and editing.

The benchmark itself is system-agnostic; the included reference
baseline happens to be an LLM agent.

Three top-level subpackages:

- :mod:`cadgenbench.common`  , shared helpers (renderer, validity, mesh).
- :mod:`cadgenbench.eval`    , benchmark scoring + report generation.
- :mod:`cadgenbench.baseline`, the reference LLM agent (build123d).
"""

__version__ = "0.1.0"
