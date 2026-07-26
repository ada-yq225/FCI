import numpy as np
import pytest

from fci_engine import FCIConfig


@pytest.mark.parametrize("alpha", [np.nan, np.inf, -np.inf, True, None])
def test_config_rejects_invalid_alpha(alpha) -> None:
    with pytest.raises((TypeError, ValueError), match="alpha"):
        FCIConfig(alpha=alpha)


@pytest.mark.parametrize(
    "name",
    ["max_cond_set_size", "sparsity_bound", "max_path_length"],
)
@pytest.mark.parametrize("value", [1.5, True, "2"])
def test_search_bounds_must_be_integers_or_none(name, value) -> None:
    with pytest.raises(TypeError, match=name):
        FCIConfig(**{name: value})


@pytest.mark.parametrize(
    "name",
    [
        "do_pdsep",
        "skeleton_stable",
        "pdsep_stable",
        "conservative_colliders",
        "conservative_orientation",
        "verbose",
    ],
)
def test_boolean_configuration_rejects_truthy_non_booleans(name) -> None:
    with pytest.raises(TypeError, match=name):
        FCIConfig(**{name: "false"})


def test_config_rejects_objects_that_do_not_implement_ci_interface() -> None:
    with pytest.raises(TypeError, match="CITest"):
        FCIConfig(ci_test=object())
