from benchmarks.gaia.harness import make_gaia_harness


def test_make_gaia_harness_has_no_evaluation_processor():
    cfg = make_gaia_harness()
    for p in getattr(cfg, "_rt_procs", None) or []:
        assert type(p).__name__ != "EvaluationProcessor", "EvaluationProcessor still wired in"


def test_make_gaia_harness_accepts_no_pipeline_evaluator_arg():
    # The signature should no longer accept `pipeline_evaluator`.
    import inspect

    sig = inspect.signature(make_gaia_harness)
    assert "pipeline_evaluator" not in sig.parameters
