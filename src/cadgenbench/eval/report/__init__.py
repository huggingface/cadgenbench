"""HTML report generators for cadgenbench result directories.

Two entry points, both wired into the ``cadgenbench report`` CLI:

- :mod:`.single_run`   -- inspect one run's per-fixture results.
- :mod:`.compare_runs` -- compare 2+ runs side by side.

Generated HTML files are self-contained: every image is embedded as a
base64 ``data:`` URI so the file can be opened anywhere, no asset
dependencies.
"""
