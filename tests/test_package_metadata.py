from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_package_metadata_supports_python_39_to_313() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'requires = ["setuptools>=61", "wheel"]' in pyproject
    assert 'requires-python = ">=3.9"' in pyproject
    for version in ("3.9", "3.10", "3.11", "3.12", "3.13"):
        assert f'"Programming Language :: Python :: {version}"' in pyproject


def test_tooling_targets_python_39_syntax() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'target-version = ["py39"]' in pyproject
    assert 'target-version = "py39"' in pyproject
    assert 'python_version = "3.9"' in pyproject
