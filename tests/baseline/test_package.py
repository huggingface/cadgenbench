from __future__ import annotations

import argparse
import importlib.util
import json
import zipfile
from pathlib import Path

PACKAGE_PATH = Path(__file__).parents[2] / "src/cadgenbench/baseline/package.py"
SPEC = importlib.util.spec_from_file_location("baseline_package", PACKAGE_PATH)
assert SPEC is not None and SPEC.loader is not None
package = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package)


def _args(run_dir: Path, out: Path) -> argparse.Namespace:
    return argparse.Namespace(
        run_dir=run_dir,
        output=out,
        submitter="team-test",
        submission_name="Smoke",
        agent_url=None,
        notes=None,
        agree=True,
    )


def test_package_preserves_missing_fixture_dirs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "101").mkdir(parents=True)
    (run_dir / "102").mkdir()
    out = tmp_path / "submission.zip"

    assert package.run(_args(run_dir, out)) == 0

    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        assert "meta.json" in names
        assert "101/" in names
        assert "102/" in names
        assert "101/output.step" not in names
        assert "102/output.step" not in names
        meta = json.loads(zf.read("meta.json"))
        assert meta["agree_to_publish"] is True


def test_package_includes_present_steps_and_missing_dirs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "101").mkdir(parents=True)
    (run_dir / "101" / "output.step").write_text("ISO-10303-21;\n")
    (run_dir / "102").mkdir()
    out = tmp_path / "submission.zip"

    assert package.run(_args(run_dir, out)) == 0

    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        assert "101/" in names
        assert "101/output.step" in names
        assert "102/" in names
        assert "102/output.step" not in names
