from examples.basic_pag_demo import expected_pag_for_toy_data, format_demo


def test_basic_pag_demo_output_contains_expected_edges() -> None:
    output = format_demo()

    assert "service_load --> latency" in output
    assert "latency --> timeouts" in output
    assert "cpu_pressure <-> error_rate" in output


def test_basic_pag_demo_queries() -> None:
    pag = expected_pag_for_toy_data()

    assert pag.definite_causes("timeouts") == ["service_load", "latency"]
    assert pag.possible_causes("timeouts") == ["service_load", "latency"]
    assert pag.definite_causes("error_rate") == []
    assert pag.possible_causes("error_rate") == []
