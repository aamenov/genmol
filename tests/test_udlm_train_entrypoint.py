from scripts.train import checkpoint_startup_mode


def test_existing_training_checkpoint_takes_precedence_over_warm_start():
    assert checkpoint_startup_mode("step-100.ckpt", "mdlm.ckpt") == "resume"


def test_warm_start_is_used_only_without_a_resume_checkpoint():
    assert checkpoint_startup_mode(None, "mdlm.ckpt") == "warm_start"
    assert checkpoint_startup_mode(None, None) == "scratch"
