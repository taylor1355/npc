"""Deliberately failing test proving CI's pytest step gates the job (NPC-1029).

Added by CI negative test 2/3 and removed by 3/3 - must never reach main.
"""


def test_ci_negative_gate_fails():
    observed = 1
    assert observed == 2
