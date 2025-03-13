import os
import json
import argparse
import pickle
import numpy as np
import torch
import gym
import pandas as pd
# from models.pg import PolicyGradient
# from models.ac import ActorCritic
#from models.gail import GAIL
from models.trpo import TRPO
import sys
sys.path.append('C:/Users/14487/python-book/follow_code/gail_yielding_trpo')

from path import test_path

def main(env_name, model_name, num_episodes):
    ckpt_path = "ckpts"
    ckpt_path = os.path.join(ckpt_path, env_name)

    with open(os.path.join(ckpt_path, "model_config.json")) as f:
        config = json.load(f)

    if env_name not in ['follow-v1']:
        print("The environment name is wrong!")
        return

    env = gym.make(env_name)
    # env.reset()
    state_dim =16
    discrete = False
    action_dim =2

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if model_name == "ppo":
        model = TRPO(
            state_dim, action_dim, discrete, config
        ).to(device)

    if hasattr(model, "pi"):
        model.pi.load_state_dict(
            torch.load(
                os.path.join(ckpt_path, "policy.ckpt"), map_location=device
            )
        )

    rwd_mean = []
    total_obs=[]
    for i in range(900):

        rwds = []

        done = False
        ob = env.reset()
        data_frame=env.unwrapped.save_time_step()
        i_d=env.unwrapped.i_d
        #print('obs',ob)
        init_ob=[np.append(ob,0)]
        #print('init_ob',init_ob)
        obs=[]
        while not done:
            act = model.act(ob)
            #print('act',act)
            ob, rwd, done, info = env.step(act)
            data_frame=pd.concat([data_frame,env.unwrapped.save_time_step()])
            rwds.append(rwd)
            obs.append(np.append(ob,act))
        data_frame.to_csv(test_path+'data_%d.csv' % (i_d))
        obs=init_ob+obs
        
        #print('obs',len(obs))
        
        total_obs.append(obs)
        
        rwd_sum = sum(rwds)
        #print("The total reward of the episode %i = %f" % (i, rwd_sum))
        rwd_mean.append(rwd_sum)

    env.close()
    
    rwd_std = np.std(rwd_mean)
    rwd_mean = np.mean(rwd_mean)
    #print("Mean = %f" % rwd_mean)
    #print("Standard Deviation = %f" % rwd_std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="follow-v1",
        help="Type the environment name to run. \
            The possible environments are \
                [CartPole-v1, Pendulum-v0, BipedalWalker-v3]"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ppo",
        help="Type the model name to train. \
            The possible models are [pg, ac, trpo, gae, ppo]"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1800,
        help="Type the number of episodes to run this agent"
    )

    args = parser.parse_args()
    main(args.env_name, args.model_name, args.num_episodes)
