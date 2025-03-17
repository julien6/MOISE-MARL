import math
import gym

from typing import Any, List
from marllib import marl
from mma_wrapper.label_manager import label_manager
from mma_wrapper.organizational_model import deontic_specification, organizational_model, structural_specifications, functional_specifications, deontic_specifications, time_constraint_type
from mma_wrapper.organizational_specification_logic import role_logic, goal_factory, role_factory, goal_logic
from mma_wrapper.utils import label, observation, action, trajectory


class mpe_label_manager(label_manager):

    def __init__(self, action_space: gym.Space = None, observation_space: gym.Space = None):
        super().__init__(action_space, observation_space)

        self.action_encode = {
            "no_action": 0,
            "move_left": 1,
            "move_right": 2,
            "move_down": 3,
            "move_up": 4
        }

        self.action_decode = {v: k for k, v in self.action_encode.items()}

        self.normal_leader_adversary_sizes = {
            'self_vel': 2,
            'self_pos': 2,
            'landmark_rel_positions': 10,
            'other_agent_rel_positions': 10,
            'other_agent_velocities': 4,
            'self_in_forest': 2,
            'leader_comm': 4
        }

        self.good_sizes = {
            'self_vel': 2,
            'self_pos': 2,
            'landmark_rel_positions': 10,
            'other_agent_rel_positions': 10,
            'other_agent_velocities': 2,
            'self_in_forest': 2
        }

    def one_hot_encode_observation(self, observation: Any, agent: str = None) -> 'observation':
        one_hot_encoded_observation = []
        for attr, val in observation.items():
            one_hot_encoded_observation = [val] + one_hot_encoded_observation
        return one_hot_encoded_observation

    def one_hot_decode_observation(self, observation: observation, agent: str = None) -> Any:
        sizes = None
        if agent in ['leadadversary_0', 'adversary_0', 'adversary_1', 'adversary_2']:
            sizes = self.normal_leader_adversary_sizes

        extracted_values = {}
        index = 0
        for key, size in sizes.items():
            extracted_values[key] = observation[index:index+size]
            index += size
        return extracted_values

    def one_hot_encode_action(self, action: Any, agent: str = None) -> action:
        return self.action_encode[action]

    def one_hot_decode_action(self, action: action, agent: str = None) -> Any:
        return self.action_decode[action]


def leader_adversary_fun(trajectory: trajectory, observation: label, agent_name: str, label_manager: label_manager) -> label:
    # print("Leader adversary")
    data = label_manager.one_hot_decode_observation(
        observation=observation, agent=agent_name)
    other_agent_rel_positions = data["other_agent_rel_positions"]
    # self_pos = (data["self_pos"][0], data["self_pos"][1])
    # other_agent_rel_positions = {agent: (other_agent_rel_positions[i*2] + self_pos[0], other_agent_rel_positions[i*2+1] + self_pos[1]) for i, agent in enumerate(
    #     ['adversary_0', 'adversary_1', 'adversary_2', 'agent_0', 'agent_1'])}
    other_agent_rel_positions = {agent: (other_agent_rel_positions[i*2], other_agent_rel_positions[i*2+1]) for i, agent in enumerate(
        ['adversary_0', 'adversary_1', 'adversary_2', 'agent_0', 'agent_1'])}
    # print(other_agent_rel_positions)

    for adversary in ["adversary_0", "adversary_1", "adversary_2"]:
        min_dist = 10000
        min_agent = None
        for good_agent in ["agent_0", "agent_1"]:
            d = math.sqrt(
                (other_agent_rel_positions[good_agent][0])**2 + (other_agent_rel_positions[good_agent][1])**2)
            if d < min_dist:
                min_dist = d
                min_agent = good_agent
        # target_agent_vec = np.array(other_agent_rel_positions[min_agent]) - np.array(self_pos)
        target_agent_vec = other_agent_rel_positions[min_agent]
        if abs(target_agent_vec[0]) / abs(target_agent_vec[1]) > 1:
            if target_agent_vec[0] > 0:
                return 2
            return 1
        else:
            if target_agent_vec[1] > 0:
                return 4
            return 3
    return 0


def normal_adversary_opc_fun(trajectory: trajectory, observation: label, agent_name: str, label_manager: label_manager) -> label:

    # print("Normal adversary")
    agent_names = ['leadadversary_0', 'adversary_0',
                   'adversary_1', 'adversary_2', 'agent_0', 'agent_1']
    agent_names.remove(agent_name)
    data = label_manager.one_hot_decode_observation(
        observation=observation, agent=agent_name)
    other_agent_rel_positions = data["other_agent_rel_positions"]
    other_agent_rel_positions = {agent: (
        other_agent_rel_positions[i*2], other_agent_rel_positions[i*2+1]) for i, agent in enumerate(agent_names)}

    min_dist = 10000
    min_agent = None
    for good_agent in ["agent_0", "agent_1"]:
        d = math.sqrt(
            (other_agent_rel_positions[good_agent][0])**2 + (other_agent_rel_positions[good_agent][1])**2)
        if d < min_dist:
            min_dist = d
            min_agent = good_agent
    target_agent_vec = other_agent_rel_positions[min_agent]
    if abs(target_agent_vec[1]) == 0:
        return 0
    if float(abs(target_agent_vec[0]) / abs(target_agent_vec[1])) >= 1.:
        if target_agent_vec[0] > 0:
            return 2
        return 1
    else:
        if target_agent_vec[1] > 0:
            return 4
        return 3
    return 0


def good_agent_fun(trajectory: trajectory, observation: label, agent_name: str, label_manager: label_manager) -> label:
    # print("Good agents")
    # pprint(extract_values(observation, good_sizes))
    # print()
    return None


mpe_model = organizational_model(
    structural_specifications(
        roles={
            "role_leader": role_logic(label_manager=mpe_label_manager).registrer_script_rule(leader_adversary_fun),
            "role_normal": role_logic(label_manager=mpe_label_manager).registrer_script_rule(normal_adversary_opc_fun),
            "role_good": role_logic(label_manager=mpe_label_manager).registrer_script_rule(normal_adversary_opc_fun)},
        role_inheritance_relations={}, root_groups={}),
    functional_specifications=functional_specifications(
        goals={}, social_scheme={}, mission_preferences=[]),
    deontic_specifications=deontic_specifications(permissions=[], obligations=[
        deontic_specification(
            "role_leader", ["leadadversary_0"], [], time_constraint_type.ANY),
        deontic_specification("role_normal", [
                              "adversary_0", "adversary_1", "adversary_2"], [], time_constraint_type.ANY)
    ]))

# prepare env
env = marl.make_env(environment_name="mpe",
                    map_name="simple_world_comm", organizational_model=mpe_model)

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
                 'params_path': "./exp_results/mappo_mlp_simple_world_comm/MAPPOTrainer_mpe_simple_world_comm_f5f5f_00000_0_2025-03-09_18-28-48/params.json",
                 'model_path': "./exp_results/mappo_mlp_simple_world_comm/MAPPOTrainer_mpe_simple_world_comm_f5f5f_00000_0_2025-03-09_18-28-48/checkpoint_000020/checkpoint-20",
                 'render': True
             },
             local_mode=True,
             share_policy="group",
             checkpoint_end=False)
