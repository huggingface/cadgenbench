"""Verify the CLI boots when optional-extra deps are missing.

The ``[baseline]`` extra pulls in litellm + python-dotenv. An eval-only
install (e.g. the leaderboard Space, which only ever runs ``cadgenbench
evaluate``) doesn't install the extra; the CLI must still register
``evaluate`` and ``report`` and run them.

These tests simulate the missing-extra environment by inserting a
sentinel that makes the relevant imports raise ``ImportError``, then
re-import ``cadgenbench.cli`` from scratch.
"""
from __future__ import annotations

import builtins
import importlib
import sys

import pytest


def _purge_modules(prefix: str) -> None:
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            del sys.modules[name]


@pytest.fixture
def baseline_unimportable(monkeypatch):
    """Make ``import cadgenbench.baseline.*`` raise, then reload cli."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cadgenbench.baseline._cli" or name == "cadgenbench.baseline.package":
            raise ImportError(
                f"simulated missing optional dep while importing {name}"
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    _purge_modules("cadgenbench.cli")
    _purge_modules("cadgenbench.baseline")
    yield
    _purge_modules("cadgenbench.cli")
    _purge_modules("cadgenbench.baseline")


def test_help_works_without_baseline_extras(baseline_unimportable, capsys):
    """``cadgenbench --help`` succeeds even when the baseline subparser
    cannot be registered. It just doesn't list the baseline command."""
    cli = importlib.import_module("cadgenbench.cli")
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "evaluate" in out
    assert "report" in out
    assert "baseline" not in out


def test_evaluate_subcommand_parses_without_baseline_extras(baseline_unimportable):
    """``cadgenbench evaluate --help`` boots without baseline extras."""
    cli = importlib.import_module("cadgenbench.cli")
    with pytest.raises(SystemExit) as exc:
        cli.main(["evaluate", "--help"])
    assert exc.value.code == 0


def test_baseline_subcommand_absent_when_extras_missing(baseline_unimportable):
    """Calling the baseline subcommand falls back to ``--help`` (unknown
    command), with no traceback leaking the missing-dep import error."""
    cli = importlib.import_module("cadgenbench.cli")
    # argparse treats an unregistered subcommand as "invalid choice" and
    # exits with code 2. Either that or the no-handler help fallback is
    # acceptable; the important thing is no ImportError surfaces.
    with pytest.raises(SystemExit):
        cli.main(["baseline", "--help"])
