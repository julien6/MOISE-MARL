import math
import gym
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


def decompose_feature_vector(feature_vector, num_agents=2, num_pots=2):
    """
    Décompose un vecteur d'observation en un dictionnaire de sous-vecteurs associés aux noms des features,
    tout en conservant l'ordre d'apparition des features.

    Arguments :
        feature_vector (np.array) : Vecteur d'observation produit par OvercookedAI.
        num_agents (int) : Nombre d'agents dans l'environnement (par défaut 2).
        num_pots (int) : Nombre de pots pris en compte (par défaut 2).

    Retourne :
        OrderedDict : Dictionnaire ordonné contenant les sous-vecteurs associés aux noms des features.
    """
    index = 0
    features = OrderedDict()

    # Orientation (4 valeurs one-hot)
    features["orientation"] = feature_vector[index:index+4]
    index += 4

    # Objet tenu (4 valeurs one-hot)
    features["held_object"] = feature_vector[index:index+4]
    index += 4

    # Distances aux objets clés (6 paires dx, dy)
    obj_names = ["onion", "tomato", "dish", "soup", "serving", "empty_counter"]
    for obj in obj_names:
        features[f"dist_{obj}"] = feature_vector[index:index+2]
        index += 2

    # Nombre d'ingrédients dans la soupe la plus proche
    features["soup_num_onions"] = feature_vector[index]
    index += 1
    features["soup_num_tomatoes"] = feature_vector[index]
    index += 1

    # Caractéristiques des casseroles proches
    for pot_idx in range(num_pots):
        features[f"pot_{pot_idx}_exists"] = feature_vector[index]
        index += 1
        features[f"pot_{pot_idx}_is_empty"] = feature_vector[index]
        index += 1
        features[f"pot_{pot_idx}_is_full"] = feature_vector[index]
        index += 1
        features[f"pot_{pot_idx}_is_cooking"] = feature_vector[index]
        index += 1
        features[f"pot_{pot_idx}_is_ready"] = feature_vector[index]
        index += 1
        features[f"pot_{pot_idx}_num_onions"] = feature_vector[index]
        index += 1
        features[f"pot_{pot_idx}_num_tomatoes"] = feature_vector[index]
        index += 1
        features[f"pot_{pot_idx}_cook_time"] = feature_vector[index]
        index += 1
        features[f"pot_{pot_idx}_dist"] = feature_vector[index:index+2]
        index += 2

    # Murs adjacents (4 valeurs booléennes)
    for direction in range(4):
        features[f"wall_{direction}"] = feature_vector[index]
        index += 1

    # Caractéristiques des autres joueurs
    other_players_features_length = (num_agents - 1) * (num_pots * 10 + 26)
    features["other_player_features"] = feature_vector[index:index +
                                                       other_players_features_length]
    index += other_players_features_length

    # Distances relatives aux autres joueurs
    features["relative_distances_to_others"] = feature_vector[index:index +
                                                              2*(num_agents-1)]
    index += 2*(num_agents-1)

    # Position absolue
    features["absolute_position"] = feature_vector[index:index+2]
    index += 2

    return dict((k, v) for k, v in features.items())


class overcooked_label_manager(label_manager):

    def __init__(self, action_space: gym.Space = None, observation_space: gym.Space = None, view_scope=None):
        super().__init__(action_space, observation_space)

        self.view_scope = view_scope

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
        return decompose_feature_vector(observation)

    def one_hot_encode_action(self, action: Any, agent: str = None) -> action:
        return self.action_encode[action]

    def one_hot_decode_action(self, action: action, agent: str = None) -> Any:
        return self.action_decode[action]


def go_towards(dx, dy):
    if abs(dx) >= abs(dy):
        if dx < 0:
            return "left"
        if dx > 0:
            return "right"
    else:
        if dy < 0:
            return "up"
        if dy > 0:
            return "down"


ori = {
    "up": np.asarray([1, 0, 0, 0]),
    "down": np.asarray([0, 1, 0, 0]),
    "right": np.asarray([0, 0, 1, 0]),
    "left": np.asarray([0, 0, 0, 1])
}

def go_and_interact(dist, orientation):
    if abs(dist[0]) > 0 or abs(dist[1]) > 0:
        towards_item = go_towards(dist[0], dist[1])
        print("------------->>>>> ", (ori[towards_item] == orientation).all(), " >>>>> ", towards_item)
        print("------------->>>>> ", ori[towards_item], " || ", orientation)
        if ((abs(dist[0]) == 0 and abs(dist[1]) == 1) or (abs(dist[0]) == 1 and abs(dist[1]) == 0)) and (ori[towards_item] == orientation).all():
            return "interact"
        return towards_item
    return None

def primary_fun(trajectory: trajectory, observation: label, agent_name: str, label_manager: label_manager) -> label:
    data = label_manager.one_hot_decode_observation(
        observation=observation, agent=agent_name)
    print(agent_name, "'s observation: ", data)

    orientation = data["orientation"]
    held_object = data["held_object"]
    dist_onion = data["dist_onion"]
    dist_soup = data["dist_soup"]
    dist_dish = data["dist_dish"]
    dist_serving = data["dist_serving"]

    pot_0_is_empty = data[f"pot_0_is_empty"]
    pot_0_is_full = data[f"pot_0_is_full"]
    pot_0_is_cooking = data[f"pot_0_is_cooking"]
    pot_0_is_ready = data[f"pot_0_is_ready"]
    pot_0_num_onions = data[f"pot_0_num_onions"]
    pot_0_cook_time = data[f"pot_0_cook_time"]
    pot_0_dist = data[f"pot_0_dist"]

    if (pot_0_is_ready == 1 or pot_0_is_cooking == 1) and np.all(held_object == 0):
        action = go_and_interact(dist_dish, orientation)
        if action is not None:
            print("àààààààààààààààààààààààààààà")
            return label_manager.one_hot_encode_action(action)
        else:
            return label_manager.one_hot_encode_action("nothing")
    else:
        if pot_0_is_full == 1 and pot_0_is_cooking == 0:
            return label_manager.one_hot_encode_action("interact")

        action = go_and_interact(dist_onion, orientation)
        if action is not None:
            return label_manager.one_hot_encode_action(action)

        action = go_and_interact(pot_0_dist, orientation)
        if action is not None:
            return label_manager.one_hot_encode_action(action)

    absolute_position = data["absolute_position"]
    #               onion
    # held_objects [0,      0,      0,      0]

    # go and interact

    return label_manager.one_hot_encode_action("nothing")


def secondary_fun(trajectory: trajectory, observation: label, agent_name: str, label_manager: label_manager) -> label:
    data = label_manager.one_hot_decode_observation(
        observation=observation, agent=agent_name)
    # print("\n", agent_name, "'s observation: ", data)
    return label_manager.one_hot_encode_action("nothing")


oa_label_mngr = overcooked_label_manager()

ao_model = organizational_model(
    structural_specifications(
        roles={
            "role_primary": role_logic(label_manager=oa_label_mngr).registrer_script_rule(primary_fun),
            "role_secondary": role_logic(label_manager=oa_label_mngr).registrer_script_rule(secondary_fun)},
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
env = marl.make_env(environment_name="overcooked",
                    map_name="asymmetric_advantages", organizational_model=ao_model)

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
                 'params_path': "./exp_results/mappo_mlp_asymmetric_advantages_copy/MAPPOTrainer_overcooked_asymmetric_advantages_08cc3_00000_0_2025-03-19_10-25-01/params.json",
                 'model_path': "./exp_results/mappo_mlp_asymmetric_advantages_copy/MAPPOTrainer_overcooked_asymmetric_advantages_08cc3_00000_0_2025-03-19_10-25-01/checkpoint_000020/checkpoint-20",
                 'render': True
             },
             local_mode=True,
             share_policy="group",
             checkpoint_end=False)
