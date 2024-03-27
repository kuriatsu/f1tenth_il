
import torch
import pygame
import numpy as np
import yaml
import argparse

import utils.agent_utils as agent_utils
import utils.env_utils as env_utils
from dataset import Dataset
from pathlib import Path
from bc import bc

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped
from ackermann_msgs.msg import AckermannDriveStamped

from policies.agents.agent_mlp import AgentPolicyMLP

class RosBridge(Node):
    def __init__(self):
        super().__init__("gym_ros_env")
        print("init_node")
        self.sub_lidar = self.create_subscription(
                LaserScan,
                "/scan",
                self.listener_callback,
                10)

        self.pub_reset = self.create_publisher(
                PoseWithCovarianceStamped,
                "/initialpose",
                10)

        self.pub_control = self.create_publisher(
                AckermannDriveStamped,
                "/drive",
                10)

        self.observ = {"scans":[]} 
        self.reward = None
        self.done = False
        self.render = False
        self.renderer = True
        self.is_collecting_data = True

        self.algo_name = "HILHGDAgger"
        self.path = "logs/{}".format(self.algo_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize the agent
        self.agent = AgentPolicyMLP(1080, 256, 2, 0.001, device)

        # hiper param
        self.max_traj_len = 3500
        self.eval_max_traj_len = 10000
        self.n_iter = 1000
        self.n_batch = 64
        self.n_round = 268 
        self.observation_shape = 1080

        self.current_round = 0
        self.current_step = 0

        torch.manual_seed(0)
        self.dataset = Dataset()
        self.lidar_noise = 0.25
        self.start_pose = np.array([[0.0, 0.0, 1.0]])
        self.traj = {
            "observs": [],
            "poses_x": [],
            "poses_y": [],
            "poses_theta": [],
            "scans": [],
            "actions": [],
            "reward": 0 
            }

        self.observ = {
            "poses_x": 0.0,
            "poses_y": 0.0,
            "poses_theta": 0.0,
            "scans": [],
            "reward": 0 
            }

        # variable of training
        self.best_model = self.agent
        self.num_of_saved_models = 0
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        ## expert
        pygame.init()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        self.reset()

    def listener_callback(self, msg):
        self.observ["scans"] = np.array([msg.ranges])

    def timer_callback(self):
        if len(self.observ["scans"]) == 0:
            return
        if self.is_collecting_data:
            self.step()

    def step(self):
        ## start collecting data
        if self.current_step < self.max_traj_len:
            ## observation
            downsampled_scan = agent_utils.downsample_and_extract_lidar(self.observ, self.observation_shape, "simple")
            downsampled_scan = agent_utils.add_noise_to_lidar(downsampled_scan, self.lidar_noise)
            ## agent action
            raw_agent_action = self.agent.get_action(downsampled_scan)
            actions = np.expand_dims(raw_agent_action, axis=0)

            ## expert intervention
            expert_steer, expert_speed = self.get_expert_action(self.joystick)
            if expert_steer is not None:
                actions[0,0] = expert_steer
            if expert_speed is not None:
                actions[0,1] = expert_speed

            if expert_steer is not None or expert_speed is not None:
                self.traj["observs"].append(self.observ)
                self.traj["poses_x"].append(0)
                self.traj["poses_y"].append(0)
                self.traj["poses_theta"].append(0)
                self.traj["scans"].append(downsampled_scan)
                self.traj["actions"].append(actions)
                self.traj["reward"]+=1
            
            out_control = AckermannDriveStamped()
            out_control.drive.speed = float(actions[0, 0])
            out_control.drive.steering_angle = float(actions[0, 1])
            self.pub_control.publish(out_control)
            self.current_step += 1

        else:
            ## rearange dataset
            self.is_collecting_data = False
            print(f"adding data self.traj observs is {len(self.traj['observs'])}")
            if len(self.traj["observs"]) > 0:
                self.traj["observs"]     = np.vstack(self.traj["observs"])
                self.traj["poses_x"]      = np.vstack(self.traj["poses_x"])
                self.traj["poses_y"]      = np.vstack(self.traj["poses_y"])
                self.traj["poses_theta"] = np.vstack(self.traj["poses_theta"])
                self.traj["scans"]       = np.vstack(self.traj["scans"])
                self.traj["actions"]     = np.vstack(self.traj["actions"])
                self.dataset.add(self.traj)

            ## train
            for _ in range(self.n_iter):
                print(f"start train at round {self.current_round}")
                train_batch = self.dataset.sample(self.n_batch)
                self.agent.train(train_batch["scans"], train_batch["actions"])

            if self.current_round > 0:
                mean_travelled_dist, stdev_travelled_dist, mean_reward, stdev_reward = agent_utils.eval(self, self.agent, self.start_pose, self.eval_max_traj_len, 1, 1080, "simple", False, "human_fast")
                model_path = Path(path + f'/{self.algo_name}_round_{str(self.round)}_dist_{int(mean_travelled_dist)}.pkl')
                model_path.parent.mkdir(parents=True, exist_ok=True) 
                torch.save(self.agent.state_dict(), model_path)
                self.is_collecting_data = True

            self.current_step = 0
            self.current_round += 1
            self.reset()

    def reset(self, *start_pose):
        out_pose = PoseWithCovarianceStamped()
        out_pose.pose.pose.position.x = self.start_pose[0,0] 
        out_pose.pose.pose.position.y = self.start_pose[0,1] 
        out_pose.pose.pose.orientation.x = 0.0
        out_pose.pose.pose.orientation.y = 0.0
        out_pose.pose.pose.orientation.z = 0.0
        out_pose.pose.pose.orientation.x = 1.0
        self.pub_reset.publish(out_pose)
        self.traj = {
            "observs": [],
            "poses_x": [],
            "poses_y": [],
            "poses_theta": [],
            "scans": [],
            "actions": [],
            "reward": 0 
            }
        return self.observ, self.reward, self.done, None

    def get_expert_action(self, joystick):
        # if event == pygame.JOYAXISMOTION:
        steer = -0.2*joystick.get_axis(4)
        speed = -10*joystick.get_axis(0)
        pygame.event.pump()

        if -0.01 < steer < 0.01:
            steer = None 
        if -0.1 < speed < 0.1:
            speed = None 

        print(f"get action steer:{steer}, speed:{speed}")
        return steer, speed
        # return None



if __name__=="__main__":
    rclpy.init()
    ros_bridge = RosBridge()

    with open('map/example_map/config_example_map.yaml') as file:
        map_conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    map_conf = argparse.Namespace(**map_conf_dict)
    start_pose = np.array([[map_conf.sx, map_conf.sy, map_conf.stheta]])


    rclpy.spin(ros_bridge) 
