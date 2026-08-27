"""Tests for gait scheduler."""

import pytest

from quadruped.planners.gait_scheduler import GaitScheduler


class TestGaitScheduler:
    @pytest.fixture
    def trot(self):
        return GaitScheduler(
            frequency=2.0,
            swing_ratio=0.5,
            phase_offset={"FL": 0.0, "FR": 0.5, "RL": 0.5, "RR": 0.0},
        )

    def test_period(self, trot):
        assert trot.period == pytest.approx(0.5)

    def test_trot_diagonal_pairs(self, trot):
        # At t=0: FL/RR swing (phase 0), FR/RL stance (phase 0.5)
        assert trot.is_swing("FL", 0.0)
        assert trot.is_swing("RR", 0.0)
        assert trot.is_stance("FR", 0.0)
        assert trot.is_stance("RL", 0.0)

    def test_contact_state_keys(self, trot):
        contacts = trot.contact_state(0.0)
        assert set(contacts.keys()) == {"FL", "FR", "RL", "RR"}
        assert all(isinstance(v, bool) for v in contacts.values())

    def test_from_config(self):
        cfg = {
            "gait": {
                "frequency": 2.0,
                "swing_ratio": 0.5,
                "phase_offset": {"FL": 0.0, "FR": 0.5, "RL": 0.5, "RR": 0.0},
            }
        }
        gait = GaitScheduler.from_config(cfg)
        assert gait.frequency == 2.0
