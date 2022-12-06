import os
from datetime import datetime
import argparse
import numpy as np
import gym
import torch
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.policies import ActorCriticPolicy as A2CPPOPolicy
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType

DEFAULT_ALGO = 'ppo'

AGGR_PHY_STEPS = 5
EPISODE_REWARD_THRESHOLD = -0

def run(
    env='hover',
    algo=DEFAULT_ALGO,
    obs=ObservationType('kin'),
    act=ActionType('one_d_rpm'),
    cpu=1,
    steps=16000,
    output_folder='output'
):

    filename = os.path.join(output_folder, 'save-'+env+'-'+algo+'-'+obs.value+'-'+act.value+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    env_name = env+"-aviary-v0"
    sa_env_kwargs = dict(aggregate_phy_steps=AGGR_PHY_STEPS, obs=obs, act=act)
    # train_env = gym.make(env_name, aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS, obs=obs, act=act)    
    train_env = make_vec_env(
        HoverAviary,
        env_kwargs=sa_env_kwargs,
        n_envs=cpu,
        seed=0
    )
    print("[INFO] Action space:", train_env.action_space)
    print("[INFO] Observation space:", train_env.observation_space)
    
    args_on = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[512, 512, dict(vf=[256, 128], pi=[256, 128])]
    )
    args_off = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[512, 512, 256, 128]
    )
    if algo == 'ppo':
        model = PPO(
            A2CPPOPolicy,
            train_env,
            policy_kwargs=args_on,
            tensorboard_log=filename+'/tb/',
            verbose=1
        )
    if algo == 'sac':
        model = SAC(
            SACPolicy,
            train_env,
            policy_kwargs=args_off,
            tensorboard_log=filename+'/tb/',
            verbose=1
        )

    eval_env = gym.make(
        env_name,
        aggregate_phy_steps=AGGR_PHY_STEPS,
        obs=obs,
        act=act
    )

    # Train
    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=EPISODE_REWARD_THRESHOLD,
        verbose=1
    )
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        verbose=1,
        best_model_save_path=filename+'/',
        log_path=filename+'/',
        eval_freq=int(2000/cpu),
        deterministic=True,
        render=False
    )
    model.learn(
        total_timesteps=steps,
        callback=eval_callback,
        log_interval=100,
    )

    # Save model
    model.save(filename+'/success_model.zip')
    print(filename)

    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hover RL')
    parser.add_argument('--algo', default=DEFAULT_ALGO, type=str, choices=['ppo', 'sac'], help='RL agent (default: ppo)', metavar='')
    ARGS = parser.parse_args()

    run(algo=ARGS.algo)