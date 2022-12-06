import os
import numpy as np
import pybullet as p
import pkg_resources

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary


class FlyThruGateAviary(BaseSingleAgentAviary):
    def __init__(
        self,
        drone_model: DroneModel=DroneModel.CF2X,
        initial_xyzs=None,
        initial_rpys=None,
        physics: Physics=Physics.PYB,
        freq: int=240,
        aggregate_phy_steps: int=1,
        gui=False,
        record=False, 
        obs: ObservationType=ObservationType.KIN,
        act: ActionType=ActionType.RPM
    ):
        super().__init__(
            drone_model=drone_model,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            freq=freq,
            aggregate_phy_steps=aggregate_phy_steps,
            gui=gui,
            record=record,
            obs=obs,
            act=act
        )
    
    def _addObstacles(self):
        super()._addObstacles()
        p.loadURDF(
            pkg_resources.resource_filename('gym_pybullet_drones', 'assets/architrave.urdf'),
            [0, 1, 0.55],
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.CLIENT
        )
        p.loadURDF(
            pkg_resources.resource_filename('gym_pybullet_drones', 'assets/architrave.urdf'),
            [0, 2, 0.55],
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.CLIENT
        )
        for i in range(10): 
            p.loadURDF(
                "cube_small.urdf",
                [-.5, 1, .02+i*0.05],
                p.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.CLIENT
            )
            p.loadURDF(
                "cube_small.urdf",
                [.5, 1, .02+i*0.05],
                p.getQuaternionFromEuler([0,0,0]),
                physicsClientId=self.CLIENT
            )
            p.loadURDF(
                "cube_small.urdf",
                [-.5, 2, .02+i*0.05],
                p.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.CLIENT
            )
            p.loadURDF(
                "cube_small.urdf",
                [.5, 2, .02+i*0.05],
                p.getQuaternionFromEuler([0,0,0]),
                physicsClientId=self.CLIENT
            )
    
    def _computeReward(self):
        state = self._getDroneStateVector(0)
        t = (self.step_counter/self.SIM_FREQ) / self.EPISODE_LEN_SEC
        return -5 * np.linalg.norm(np.array([0, 1 + t, 1]) - state[0:3])**2
    
    def _computeDone(self):
        if self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    def _computeInfo(self):
        return {"dummy": 1}
    
    def _clipAndNormalizeState(self,state):
        MAX_LIN_VEL_XY = 2 
        MAX_LIN_VEL_Z = 1
        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC
        MAX_PITCH_ROLL = np.pi/2

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([
            normalized_pos_xy,
            normalized_pos_z,
            state[3:7],
            normalized_rp,
            normalized_y,
            normalized_vel_xy,
            normalized_vel_z,
            normalized_ang_vel,
            state[16:20]
        ]).reshape(20,)

        return norm_and_clipped