import time
import argparse
import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.utils import sync

DEFAULT_GUI = True
DEFAULT_PLOT = True
DEFAULT_OUTPUT_FOLDER = 'output'
AGGR_PHY_STEPS = 5

def run(
    exp, 
    gui=DEFAULT_GUI, 
    plot=DEFAULT_PLOT, 
    output_folder=DEFAULT_OUTPUT_FOLDER
):
    algo = exp.split("-")[2]
    path = exp+'/success_model.zip'
    if algo == 'ppo':
        model = PPO.load(path)
    if algo == 'sac':
        model = SAC.load(path)

    env_name = exp.split("-")[1]+"-aviary-v0"
    OBS = ObservationType.KIN

    action_name = exp.split("-")[4]
    ACT = [action for action in ActionType if action.value == action_name]
    ACT = ACT.pop()

    eval_env = gym.make(
        env_name,
        aggregate_phy_steps=AGGR_PHY_STEPS,
        obs=OBS,
        act=ACT
    )
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print("\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    # model
    test_env = gym.make(
        env_name,
        gui=gui,
        record=False,
        aggregate_phy_steps=AGGR_PHY_STEPS,
        obs=OBS,
        act=ACT
    )
    logger = Logger(
        logging_freq_hz=int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS),
        num_drones=1,
        output_folder=output_folder
    )
    obs = test_env.reset()
    start = time.time()
    for i in range(4*int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS)):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = test_env.step(action)
        test_env.render()
        logger.log(
            drone=0,
            timestamp=i/test_env.SIM_FREQ,
            state=np.hstack([obs[0:3], np.zeros(4), obs[3:15],  np.resize(action, (4))]),
            control=np.zeros(12)
        )
        sync(np.floor(i*test_env.AGGR_PHY_STEPS), start, test_env.TIMESTEP)
        # if done:
        #     break
    test_env.close()
    if plot:
        logger.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fly to point RL')
    parser.add_argument('--exp', type=str, help='T', metavar='')
    ARGS = parser.parse_args()

    run(exp=ARGS.exp)