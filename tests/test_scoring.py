from agent_debugger_evals.scoring import score_test_run


def test_score_passes_only_for_zero_exit_code() -> None:
    passing = score_test_run(0, "all tests passed")
    failing = score_test_run(1, "one test failed")

    assert passing.passed
    assert passing.output == "all tests passed"
    assert not failing.passed
    assert failing.exit_code == 1
