"""
Script to use CtrlAviary to hover to a point
"""
import time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync

def run(
    gui=True,
    record_video=False,
    plot=True,
    user_debug_gui=False,
    obstacles=True,
    simulation_freq_hz=240,
    control_freq_hz=240,
    duration_sec=4,
    output_folder='output',
    colab=False
):

    PERIOD = 4
    COUNTER = 0
    AGGR_PHY_STEPS = int(simulation_freq_hz/control_freq_hz)
    INIT_XYZ = np.array([0, 0, 0.015]).reshape(1,3)
    TARGET_XYZ = np.array([0, 0, 1]).reshape(1,3)
    TARGET_POS = np.zeros((control_freq_hz*PERIOD,3))
    for i in range(control_freq_hz*PERIOD):
        TARGET_POS[i, :] = TARGET_XYZ[0, 0], TARGET_XYZ[0, 1], TARGET_XYZ[0, 2]

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=INIT_XYZ,
        physics=Physics.PYB_GND,
        neighbourhood_radius=10,
        freq=simulation_freq_hz,
        aggregate_phy_steps=AGGR_PHY_STEPS,
        gui=gui,
        record=record_video,
        obstacles=obstacles,
        user_debug_gui=user_debug_gui
    )
    PYB_CLIENT = env.getPyBulletClient()
    logger = Logger(
        logging_freq_hz=control_freq_hz,
        num_drones=1,
        output_folder=output_folder,
        colab=colab
    )

    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

    action = {"0": np.array([0,0,0,0])}
    X_STEPS = int(np.floor(env.SIM_FREQ/control_freq_hz))
    START = time.time()

    # SIM Loop
    for i in range(0, int(duration_sec*env.SIM_FREQ), AGGR_PHY_STEPS):
        obs, _, _, _ = env.step(action)

        if i%X_STEPS == 0:
            action["0"], _, _ = ctrl.computeControlFromState(
                control_timestep=X_STEPS*env.TIMESTEP,
                state=obs["0"]["state"],
                target_pos=TARGET_POS[COUNTER, :],
            )
            if COUNTER < ((control_freq_hz*PERIOD)-1):
                COUNTER += 1
            else:
                COUNTER = 0

        logger.log(
            drone=0,
            timestamp=i/env.SIM_FREQ,
            state=obs["0"]["state"],
            control=np.hstack([TARGET_POS[COUNTER, :], np.zeros(9)])
        )
        if i%env.SIM_FREQ == 0:
            env.render()
        if gui:
            sync(i, START, env.TIMESTEP)

    env.close()
    logger.save()
    if plot:
        logger.plot()

run()
