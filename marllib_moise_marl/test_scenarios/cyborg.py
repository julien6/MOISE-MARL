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


class cyborg_label_manager(label_manager):

    def __init__(self, action_space: gym.Space = None, observation_space: gym.Space = None):
        super().__init__(action_space, observation_space)

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
        return

    def one_hot_encode_action(self, action: Any, agent: str = None) -> action:
        return self.action_encode[action]

    def one_hot_decode_action(self, action: action, agent: str = None) -> Any:
        return self.action_decode[action]


def primary_fun(trajectory: trajectory, observation: label, agent_name: str, label_manager: label_manager) -> label:
    data = label_manager.one_hot_decode_observation(
        observation=observation, agent=agent_name)
    return label_manager.one_hot_encode_action("nothing")


cyborg_label_mngr = cyborg_label_manager()

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
            "role_primary", ["agent_0"], [], time_constraint_type.ANY),
        deontic_specification(
            "role_secondary", ["agent_1"], [], time_constraint_type.ANY)
    ]))

# prepare env
env = marl.make_env(environment_name="cyborg", map_name="cage3", organizational_model=None) # cyborg_model

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
                 'render': True
             },
             local_mode=True,
             share_policy="group",
             stop_timesteps=1,
             timesteps_total=1,
             checkpoint_freq=1,
             stop_iters=1,
             checkpoint_end=True)
