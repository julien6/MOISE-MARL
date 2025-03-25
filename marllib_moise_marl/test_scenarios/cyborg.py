import math
import gym
import random
import numpy as np
from random import randint
from typing import Any, List
from marllib import marl
from mma_wrapper.label_manager import label_manager
from mma_wrapper.organizational_model import deontic_specification, organizational_model, structural_specifications, functional_specifications, deontic_specifications, time_constraint_type
from mma_wrapper.organizational_specification_logic import role_logic, goal_factory, role_factory, goal_logic
from mma_wrapper.utils import label, observation, action, trajectory
from marllib.envs.base_env.wmt import RLlibWMT
from collections import OrderedDict
from ipaddress import IPv4Address
from marllib.envs.base_env.cyborg import create_env


class cyborg_label_manager(label_manager):

    def __init__(self, ip_list, ip_host_map, msg_len, agent_host_map, action_space: gym.Space = None, observation_space: gym.Space = None):
        super().__init__(action_space, observation_space)

        self.ip_list = ip_list
        self.ip_host_map = ip_host_map
        self.msg_len = msg_len
        self.agent_host_map = agent_host_map

        self.action_encode = {
            "nothing": 4,
            "up": 0,
            "down": 1,
            "left": 3,
            "right": 2,
            "interact": 5
        }

        self.action_decode = {v: k for k, v in self.action_encode.items()}

    def one_hot_decode_observation(self, observation: observation, agent: str = None) -> Any:

        num_drones = len(self.ip_list)
        obs = {}

        idx = 0

        # === success (value was obs['success'].value - 1) ===
        success_val = observation[idx]
        success_enum = {0: "FAILURE", 1: "SUCCESS", 2: "UNKNOWN"}
        obs["success"] = success_enum.get(success_val + 1, "UNKNOWN")
        idx += 1

        own_drone = self.agent_host_map[agent]
        own_ip = next(ip for ip, host in self.ip_host_map.items()
                      if host == own_drone)

        obs[own_drone] = {
            "Interface": [{
                "Interface Name": "wlan0",
                "IP Address": str(own_ip)  # placeholder
            }],
            "System info": {},
        }

        # === Blocked IPs ===
        blocked_ips = []
        for i, ip in enumerate(self.ip_list):
            if observation[idx + i] == 1:
                blocked_ips.append(str(ip))
        idx += num_drones
        obs[own_drone]["Interface"][0]["blocked_ips"] = blocked_ips

        # === Flagged Processes ===
        has_processes = observation[idx] == 1
        if has_processes:
            obs[own_drone]["Processes"] = [{"PID": 1094, "Username": "root"}]
        idx += 1

        # === Network Connections ===
        net_conns = []
        for i, ip in enumerate(self.ip_list):
            if observation[idx + i] == 1:
                net_conns.append({
                    "remote_address": str(ip)
                })
        idx += num_drones
        if len(net_conns) > 0:
            obs[own_drone]["Interface"][0]["NetworkConnections"] = net_conns

        # === Own Position ===
        x, y = observation[idx], observation[idx + 1]
        obs[own_drone]["System info"]["position"] = [x, y]
        idx += 2

        # === Other drones (excluding own) ===
        obs.update({
            host: {
                "System info": {},
            } for ip, host in self.ip_host_map.items() if host != own_drone
        })

        for _ in range(num_drones - 1):
            drone_id = observation[idx]
            idx += 1
            pos_x = observation[idx]
            pos_y = observation[idx + 1]
            idx += 2
            has_session = observation[idx] == 1
            idx += 1

            ip = self.ip_list[drone_id]
            hostname = self.ip_host_map[ip]

            obs[hostname]["System info"]["position"] = [pos_x, pos_y]
            if has_session:
                obs[hostname]["Sessions"] = [{"Username": "root", "ID": 0}]

        # === Message parsing ===
        if self.msg_len > 0:
            message_part = observation[idx:idx+self.msg_len].tolist()
            idx += self.msg_len
            obs["message"] = message_part

        return obs

    def one_hot_encode_action(self, action: Any, agent: str = None) -> action:
        return self.action_encode[action]

    def one_hot_decode_action(self, action: action, agent: str = None) -> Any:
        return self.action_decode[action]


def primary_fun(trajectory: trajectory, observation: label, agent_name: str, label_manager: label_manager) -> label:
    data = label_manager.one_hot_decode_observation(
        observation=observation, agent=agent_name)
    from pprint import pprint
    pprint(data)
    print("\n\n")
    return None


_env = create_env()
_env.reset()

ip_list = _env.ip_addresses
ip_host_map = {ip: host for host, ip in _env.env.get_ip_map().items()}
msg_len = _env.msg_len
agent_host_map = _env.agent_host_map
agent_ids = _env._agent_ids
int_to_action = _env.int_to_action

cyborg_label_mngr = cyborg_label_manager(
    ip_list, ip_host_map, msg_len, agent_host_map)

cyborg_model = organizational_model(
    structural_specifications(
        roles={
            "role_primary": role_logic(label_manager=cyborg_label_mngr).registrer_script_rule(primary_fun),
            "role_secondary": role_logic(label_manager=cyborg_label_mngr).registrer_script_rule(primary_fun)},
        role_inheritance_relations={}, root_groups={}),
    functional_specifications=functional_specifications(
        goals={}, social_scheme={}, mission_preferences=[]),
    deontic_specifications=deontic_specifications(permissions=[], obligations=[
        deontic_specification(
            "role_primary", ["blue_agent_0", "blue_agent_2", "blue_agent_3", "blue_agent_4", "blue_agent_5", "blue_agent_6"], [], time_constraint_type.ANY),
        deontic_specification(
            "role_secondary", ["blue_agent_1"], [], time_constraint_type.ANY)
    ]))

# prepare env
env = marl.make_env(environment_name="cyborg",
                    map_name="cage3", organizational_model=cyborg_model)

# initialize algorithm with appointed hyper-parameters
# mappo = marl.algos.mappo(hyperparam_source="mpe")
mappo = marl.algos.mappo(hyperparam_source="test")

# build agent model based on env + algorithms + user preference
model = marl.build_model(
    env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

# # start training
# mappo.fit(env, model, stop={'episode_reward_mean': 6000, 'timesteps_total': 20000000}, local_mode=False,
#           num_gpus=0, num_gpus_per_worker=0, num_workers=1, share_policy='group', checkpoint_freq=20)

# rendering
mappo.render(env, model,
             restore_path={
                 'params_path': "./exp_results/mappo_mlp_cage3_copy/MAPPOTrainer_cyborg_cage3_313ec_00000_0_2025-03-23_17-30-54/params.json",
                 'model_path': "./exp_results/mappo_mlp_cage3_copy/MAPPOTrainer_cyborg_cage3_313ec_00000_0_2025-03-23_17-30-54/checkpoint_000020/checkpoint-20",
                 #  'record_env': True,
                 'render_env': True
             },
             local_mode=True,
             share_policy="group",
             stop_timesteps=1,
             timesteps_total=1,
             stop_reward=10,
             checkpoint_freq=1,
             stop_iters=1,
             checkpoint_end=True)
