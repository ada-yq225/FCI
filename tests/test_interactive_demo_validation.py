from interactive_demo import generate_medical_data
from fci_engine import fci


def test_medical_interactive_demo_recovers_expected_reference_shape() -> None:
    data = generate_medical_data(8000)

    result = fci(data, alpha="auto", verbose=False)

    assert result.config.alpha == 0.001
    assert result.graph.edge_repr("X1", "A") == "X1 o-> A"
    assert result.graph.edge_repr("X2", "B") == "X2 o-> B"
    assert result.graph.edge_repr("A", "B") == "A <-> B"
    assert result.graph.edge_repr("A", "D") == "A --> D"
