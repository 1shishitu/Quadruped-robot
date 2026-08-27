"""MPC placeholder tests (Phase 4)."""

import numpy as np
import pytest

from quadruped.controllers.gait_controller import MPCController
from quadruped.config_loader import load_mpc_config, load_robot_config
from quadruped.models.srbm import SingleRigidBodyModel
from quadruped.types import TrunkRef


class TestMPCController:
    @pytest.fixture
    def controller(self):
        robot_cfg = load_robot_config()
        mpc_cfg = load_mpc_config()
        return MPCController.from_config(robot_cfg, mpc_cfg)

    def test_not_implemented(self, controller):
        state = np.zeros(SingleRigidBodyModel.STATE_DIM)
        ref = TrunkRef(
            p_com=np.zeros(3),
            v_com=np.zeros(3),
            a_com=np.zeros(3),
            rpy=np.zeros(3),
            omega=np.zeros(3),
        )
        contact = np.ones((controller.horizon, 4), dtype=bool)
        with pytest.raises(NotImplementedError):
            controller.compute(state, ref, contact)
