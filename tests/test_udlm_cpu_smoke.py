import pytest

from scripts.udlm.cpu_smoke import _validate_smoke_gate


def _diagnostics(before_loss=6.0, after_loss=2.0):
    return (
        {"0.5": {"loss": before_loss}},
        {"0.5": {"loss": after_loss}},
    )


def test_smoke_gate_accepts_falling_losses_and_a_strict_decode():
    before, after = _diagnostics()
    _validate_smoke_gate([6.0, 5.0, 2.0, 1.0], before, after, ["CCO"])


@pytest.mark.parametrize(
    ("losses", "before_after", "generated", "message"),
    [
        ([1.0, 2.0], _diagnostics(), ["CCO"], "window mean"),
        ([2.0, 1.0], _diagnostics(1.0, 2.0), ["CCO"], "diagnostic loss"),
        ([2.0, 1.0], _diagnostics(), [], "no strictly decodable"),
    ],
)
def test_smoke_gate_rejects_failed_evidence(losses, before_after, generated, message):
    before, after = before_after
    with pytest.raises(RuntimeError, match=message):
        _validate_smoke_gate(losses, before, after, generated)
