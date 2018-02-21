#!/usr/bin/env python
# -- coding: utf-8 --
import os,sys
from ddpg_model import *
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import argparse
import numpy as np

from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios




if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='delayed_communication.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = True)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    #policies = [InteractivePolicy(env,i) for i in range(env.n)]#可能会有多个环境，所以可能有多个policy
    # execution loop
    #obs_n = env.reset()
    ddpg_model=DDPG(10,520,0.5)
    var=0.5
    for i in range(1000):

        s = env.reset()#env.reset()没有工作。
        env.render()
        ep_reward = 0

        for j in range(2000):

            # Add exploration noise
            a = ddpg_model.choose_action(np.reshape(s,[520]))
            a = np.clip(np.random.normal(a, var), -1, 1)  # add randomness to action selection for exploration
            s_, r, done, info = env.step(np.reshape(a,[10,2]))
            env.render()
            ddpg_model.store_transition(s, np.reshape(a,[10,2]), r, s_)

            if ddpg_model.pointer > MEMORY_CAPACITY:
                var *= .9999  # decay the action randomness
                ddpg_model.learn()

            s = s_
            ep_reward += np.mean(r)
            if j == 200 - 1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var,)
                # if ep_reward > -300:RENDER = True
                break

