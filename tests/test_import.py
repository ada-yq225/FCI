def test_import_fci_engine() -> None:
    import fci_engine

    assert fci_engine.__version__ == "0.1.0"


def test_common_ci_tests_are_public_exports() -> None:
    from fci_engine import (
        FCIPlusConfig,
        FisherZTest,
        KernelCITest,
        MissingValueFisherZTest,
    )

    assert FCIPlusConfig.__name__ == "FCIPlusConfig"
    assert FisherZTest.__name__ == "FisherZTest"
    assert KernelCITest.__name__ == "KernelCITest"
    assert MissingValueFisherZTest.allow_nan is True
