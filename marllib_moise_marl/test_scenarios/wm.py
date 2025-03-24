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


class wm_label_manager(label_manager):

    def __init__(self, action_space: gym.Space = None, observation_space: gym.Space = None, view_scope=None):
        super().__init__(action_space, observation_space)

        self.view_scope = view_scope

        self.action_encode = {
            "nothing": 0,
            "up": 1,
            "down": 2,
            "left": 3,
            "right": 4,
            "pick": 5,
            "drop": 6
        }

        self.action_decode = {v: k for k, v in self.action_encode.items()}

        self.cell_encode = {
            "empty": 1,
            "obstacle": 0,
            "agent": 2,
            "agent_with_primary": 3,
            "agent_with_secondary": 4,
            "primary_object": 5,
            "secondary_object": 6,
            "empty_input": 7,
            "input_with_object": 8,
            "empty_input_craft": 9,
            "input_craft_with_object": 10,
            "empty_output_craft": 11,
            "output_craft_with_object": 12,
            "empty_output": 13,
            "output_with_object": 14
        }

        self.cell_decode = {v: k for k, v in self.cell_encode.items()}

    def one_hot_encode_observation(self, observation: Any, agent: str = None) -> 'observation':
        _obs = np.asarray(observation)
        if len(_obs.shape) > 1:
            _obs = _obs.reshape(np.prod(_obs.shape))
        _obs = np.asarray([self.cell_encode[_obs[x]]
                          for x in range(0, _obs.shape[0])])
        return _obs

    def one_hot_decode_observation(self, observation: observation, agent: str = None) -> Any:
        _obs = np.asarray([self.cell_decode[observation[x]]
                          for x in range(0, observation.shape[0])])
        if self.view_scope is not None:
            return _obs.reshape((self.view_scope * 2 + 1), (self.view_scope * 2 + 1))
        return _obs

    def one_hot_encode_action(self, action: Any, agent: str = None) -> action:
        return self.action_encode[action]

    def one_hot_decode_action(self, action: action, agent: str = None) -> Any:
        return self.action_decode[action]


def primary_fun(trajectory: trajectory, observation: label, agent_name: str, label_manager: label_manager) -> label:
    data = label_manager.one_hot_decode_observation(
        observation=observation, agent=agent_name)
    x_agent, y_agent = data.shape[0]//2, data.shape[1]//2

    if randint(0, 100) < 20:
        return randint(1, 4)

    if data[x_agent, y_agent] == "agent_with_primary":
        for x in range(0, data.shape[0]):
            for y in range(0, data.shape[1]):
                if "empty_input_craft" == data[x, y]:
                    dx, dy = x - x_agent, y - y_agent
                    # print(dx, " ", dy)
                    if abs(dx) == 0 and abs(dy) == 1:
                        # print(agent_name, ": drop")
                        return 6
                    if abs(dx) >= abs(dy):
                        if dx < 0:
                            # print(agent_name, ": down")
                            return 1
                        if dx > 0:
                            # print(agent_name, ": up")
                            return 2
                    else:
                        if dy < 0:
                            # print(agent_name, ": left")
                            return 3
                        if dy > 0:
                            # print(agent_name, ": right")
                            return 4
        return 3
    if data[x_agent, y_agent] == "agent":
        for x in range(0, data.shape[0]):
            for y in range(0, data.shape[1]):
                if "input_with_object" == data[x, y]:
                    dx, dy = x - x_agent, y - y_agent
                    # print(dx, " ", dy)
                    if abs(dx) == 0 and abs(dy) == 1:
                        # print(agent_name, ": pick")
                        return 5
                    if abs(dx) >= abs(dy):
                        if dx < 0:
                            # print(agent_name, ": down")
                            return 1
                        if dx > 0:
                            # print(agent_name, ": up")
                            return 2
                    else:
                        if dy < 0:
                            # print(agent_name, ": left")
                            return 3
                        if dy > 0:
                            # print(agent_name, ": right")
                            return 4
        return 3

    return 0


def secondary_fun(trajectory: trajectory, observation: label, agent_name: str, label_manager: label_manager) -> label:
    data = label_manager.one_hot_decode_observation(
        observation=observation, agent=agent_name)

    def block_around(x, y, grid, block):
        directions = [(1, 0), (0, 1), (0, -1), (-1, 0)]
        for x_dir, y_dir in directions:
            if (grid[x+x_dir][y+y_dir] == block):
                return True

    if randint(0, 100) < 40:
        return randint(1, 4)

    x_agent, y_agent = data.shape[0]//2, data.shape[1]//2
    if data[x_agent, y_agent] == "agent_with_secondary":
        for x in range(0, data.shape[0]):
            for y in range(0, data.shape[1]):
                if "empty_output" == data[x, y]:
                    dx, dy = x - x_agent, y - y_agent
                    # print(dx, " ", dy)
                    if abs(dx) == 0 and abs(dy) == 1:
                        # print(agent_name, ": drop")
                        return 6
                    if abs(dx) >= abs(dy):
                        if dx < 0:
                            # print(agent_name, ": down")
                            return 1
                        if dx > 0:
                            # print(agent_name, ": up")
                            return 2
                    else:
                        if dy < 0:
                            # print(agent_name, ": left")
                            return 3
                        if dy > 0:
                            # print(agent_name, ": right")
                            return 4
        return 4
    if data[x_agent, y_agent] == "agent":

        if block_around(x_agent, y_agent, data, "output_craft_with_object"):
            return 5

        for x in range(0, data.shape[0]):
            for y in range(0, data.shape[1]):
                if "output_craft_with_object" == data[x, y]:
                    dx, dy = x - x_agent, y - y_agent
                    # print(dx, " ", dy)
                    if abs(dx) >= abs(dy):
                        if dx < 0:
                            # print(agent_name, ": down")
                            return 1
                        if dx > 0:
                            # print(agent_name, ": up")
                            return 2
                    else:
                        if dy < 0:
                            # print(agent_name, ": left")
                            return 3
                        if dy > 0:
                            # print(agent_name, ": right")
                            return 4
        return random.choice([2, 4]) if randint(0, 100) < 70 else randint(1, 4)
    return 0


_env = RLlibWMT({"map_name": "warehouse_management"})
view_scope = _env.env.par_env.view_size
wm_label_mngr = wm_label_manager(view_scope=view_scope)

wm_model = organizational_model(
    structural_specifications(
        roles={
            "role_primary": role_logic(label_manager=wm_label_mngr).registrer_script_rule(primary_fun),
            "role_secondary": role_logic(label_manager=wm_label_mngr).registrer_script_rule(secondary_fun)},
        role_inheritance_relations={}, root_groups={}),
    functional_specifications=functional_specifications(
        goals={}, social_scheme={}, mission_preferences=[]),
    deontic_specifications=deontic_specifications(permissions=[], obligations=[
        deontic_specification(
            "role_primary", ["agent_0", "agent_1"], [], time_constraint_type.ANY),
        deontic_specification(
            "role_secondary", ["agent_2"], [], time_constraint_type.ANY)
    ]))


# prepare env
env = marl.make_env(environment_name="wmt",
                    map_name="warehouse_management", organizational_model=wm_model)

# initialize algorithm with appointed hyper-parameters
# mappo = marl.algos.mappo(hyperparam_source="mpe")
mappo = marl.algos.mappo(hyperparam_source="test")

# build agent model based on env + algorithms + user preference
model = marl.build_model(
    env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

# start training
# mappo.fit(env, model, stop={'episode_reward_mean': 6000, 'timesteps_total': 20000000}, local_mode=False, num_gpus=0, num_gpus_per_worker=0,
#           num_workers=1, share_policy='group', checkpoint_freq=20)

# rendering
mappo.render(env, model,
             restore_path={
                 'params_path': "./exp_results/mappo_mlp_warehouse_management_copy/MAPPOTrainer_wmt_warehouse_management_3846d_00000_0_2025-03-18_14-09-27/params.json",
                 'model_path': "./exp_results/mappo_mlp_warehouse_management_copy/MAPPOTrainer_wmt_warehouse_management_3846d_00000_0_2025-03-18_14-09-27/checkpoint_000020/checkpoint-20",
                #  'render': True,
                #  'record_env': True,
                 'render_env': True
             },
             local_mode=True,
             share_policy="group",
             stop_timesteps=100,
             timesteps_total=100,
             checkpoint_freq=1,
             stop_iters=1,
             checkpoint_end=True)
