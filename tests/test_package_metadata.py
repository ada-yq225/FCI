from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_package_metadata_supports_python_39_to_313() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'requires = ["setuptools>=77", "wheel"]' in pyproject
    assert 'requires-python = ">=3.9"' in pyproject
    for version in ("3.9", "3.10", "3.11", "3.12", "3.13"):
        assert f'"Programming Language :: Python :: {version}"' in pyproject


def test_tooling_targets_python_39_syntax() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'target-version = ["py39"]' in pyproject
    assert 'target-version = "py39"' in pyproject
    assert 'python_version = "3.9"' in pyproject


def test_mypy_toolchain_remains_compatible_with_python_39() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    typed_marker = ROOT / "src" / "fci_engine" / "py.typed"

    assert '"mypy>=1.5,<2"' in pyproject
    assert '"pandas-stubs>=1.5"' in pyproject
    assert '"types-networkx>=3.1"' in pyproject
    assert "strict = true" in pyproject
    assert pyproject.count("ignore_missing_imports = true") == 1
    assert 'module = ["causallearn.*"]' in pyproject
    assert 'fci_engine = ["py.typed"]' in pyproject
    assert typed_marker.exists()


def test_ruff_policy_is_stable_across_minor_releases() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '"ruff>=0.16,<0.17"' in pyproject
    assert 'required-version = ">=0.16,<0.17"' in pyproject
    assert 'select = ["E4", "E7", "E9", "F"]' in pyproject


def test_dev_extra_includes_report_test_dependencies() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '"matplotlib>=3.6"' in pyproject


def test_package_uses_spdx_license_metadata() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'license = "MIT"' in pyproject
    assert 'license-files = ["LICENSE"]' in pyproject
    assert (ROOT / "LICENSE").exists()
