import math
import gym

from typing import Any, List
from marllib import marl
from mma_wrapper.label_manager import label_manager
from mma_wrapper.organizational_model import deontic_specification, organizational_model, structural_specifications, functional_specifications, deontic_specifications, time_constraint_type
from mma_wrapper.organizational_specification_logic import role_logic, goal_factory, role_factory, goal_logic
from mma_wrapper.utils import label, observation, action, trajectory


class wm_label_manager(label_manager):

    def __init__(self, action_space: gym.Space = None, observation_space: gym.Space = None):
        super().__init__(action_space, observation_space)

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
        return [[self.cell_encode[cell] for cell in line] for line in observation]

    def one_hot_decode_observation(self, observation: observation, agent: str = None) -> Any:
        return [[self.cell_decode[cell] for cell in line] for line in observation]

    def one_hot_encode_action(self, action: Any, agent: str = None) -> action:
        return self.action_encode[action]

    def one_hot_decode_action(self, action: action, agent: str = None) -> Any:
        return self.action_decode[action]


def primary_fun(trajectory: trajectory, observation: label, agent_name: str, label_manager: label_manager) -> label:
    data = label_manager.one_hot_decode_observation(
        observation=observation, agent=agent_name)
    return 0


def secondary_fun(trajectory: trajectory, observation: label, agent_name: str, label_manager: label_manager) -> label:
    data = label_manager.one_hot_decode_observation(
        observation=observation, agent=agent_name)
    return 0


# prepare env
env = marl.make_env(environment_name="wmt", map_name="warehouse_management")

# initialize algorithm with appointed hyper-parameters
# mappo = marl.algos.mappo(hyperparam_source="mpe")
mappo = marl.algos.mappo(hyperparam_source="test")

# build agent model based on env + algorithms + user preference
model = marl.build_model(
    env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

# start training
mappo.fit(env, model, stop={'episode_reward_mean': 6000, 'timesteps_total': 20000000}, local_mode=False, num_gpus=0, num_gpus_per_worker=0,
          num_workers=1, share_policy='group', checkpoint_freq=20)

# mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 20000000}, local_mode=False, num_gpus=0,
#             num_workers=1, share_policy='individual', checkpoint_freq=20)
