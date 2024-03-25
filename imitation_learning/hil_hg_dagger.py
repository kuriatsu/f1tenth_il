#!/usr/bin/python3
# -*-coding:utf-8-*-

import torch
import pygame
import numpy as np

import utils.agent_utils as agent_utils
import utils.env_utils as env_utils
from dataset import Dataset
from pathlib import Path
from bc import bc

def get_expert_action(joystick):
    # if event == pygame.JOYAXISMOTION:
    steer = -0.2*joystick.get_axis(0)
    speed = -5.0*joystick.get_axis(4)
    pygame.event.pump()
    print(f"get action steer:{steer}, speed:{speed}")
    return np.array([[steer, speed]])
    # return None


def hil_hg_dagger(seed, agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode):
    algo_name = "HILHGDAgger"
    path = "logs/{}".format(algo_name)

    # hiper param
    max_traj_len = 3500
    eval_max_traj_len = 10000
    n_iter = 1000
    n_batch = 64
    n_round = 268 

    torch.manual_seed(seed)
    dataset = Dataset()
    lidar_noise = 0.25

    # variable of training
    best_model = agent
    num_of_saved_models = 0
    
    ## expert
    pygame.init()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    for round in range(n_round):

        if round > 0:
            mean_travelled_dist, stdev_travelled_dist, mean_reward, stdev_reward = agent_utils.eval(env, agent, start_pose, eval_max_traj_len, 1, observation_shape, downsampling_method, render, render_mode)

            
            model_path = Path(path + f'/{algo_name}_round_{str(round)}_dist_{int(mean_travelled_dist)}.pkl')
            model_path.parent.mkdir(parents=True, exist_ok=True) 
            torch.save(agent.state_dict(), model_path)

        
        if round == 0:
            agent, log, dataset = bc(seed, agent, expert, env, start_pose, observation_shape, downsampling_method, render, render_mode, purpose="bootstrap")
        else:
            observ, step_reward, cone, info = env.reset(start_pose) 
            if env.renderer is None: 
                env.render()

            traj = {
                "observs": [],
                "poses_x": [],
                "poses_y": [],
                "poses_theta": [],
                "scans": [],
                "actions": [],
                "reward": 0 
                }

            ## start collecting data
            for _ in range(max_traj_len):
                downsampled_scan = agent_utils.downsample_and_extract_lidar(observ, observation_shape, downsampling_method)
                downsampled_scan = agent_utils.add_noise_to_lidar(downsampled_scan, lidar_noise)
                
                raw_agent_action = agent.get_action(downsampled_scan)
                actions = np.expand_dims(raw_agent_action, axis=0)
                expert_action = get_expert_action(joystick)
                
                if  expert_action[0,0] < -0.01 or 0.01 < expert_action[0,0]:
                    actions[0,0] = expert_action[0,0]

                if  expert_action[0,1] < -0.01 or 0.01 < expert_action[0,1]:
                    actions[0,1] = expert_action[0,1]

                traj["observs"].append(observ)
                traj["poses_x"].append(observ["poses_x"][0])
                traj["poses_y"].append(observ["poses_y"][0])
                traj["poses_theta"].append(observ["poses_theta"][0])
                traj["scans"].append(downsampled_scan)
                traj["actions"].append(actions)
                traj["reward"]+=step_reward
                
                observ, reward, done, _ = env.step(actions)
                env.render(mode=render_mode)
                if done: break

            ## rearange dataset
            if len(traj["observs"]) > 0:
                traj["observs"]     = np.vstack(traj["observs"])
                traj["poses_x"]      = np.vstack(traj["poses_x"])
                traj["poses_y"]      = np.vstack(traj["poses_y"])
                traj["poses_theta"] = np.vstack(traj["poses_theta"])
                traj["scans"]       = np.vstack(traj["scans"])
                traj["actions"]     = np.vstack(traj["actions"])
                dataset.add(traj)

            ## train
            for _ in range(n_iter):
                train_batch = dataset.sample(n_batch)
                agent.train(train_batch["scans"], train_batch["actions"])
