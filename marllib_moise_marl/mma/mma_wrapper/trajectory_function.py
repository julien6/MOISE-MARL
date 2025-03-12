import importlib
import itertools

from typing import Callable, Dict, List, Tuple, get_type_hints
from mma_wrapper.label_manager import label_manager
from mma_wrapper.utils import label, load_function, trajectory, trajectory_str
from mma_wrapper.utils import observation, action


class trajectory_functions:

    def __init__(self):
        self.custom_functions = {
            "policy_functions": [], "reward_functions": []}

    def add_function(self, function: Callable[[trajectory, observation, str, label_manager], List[Tuple[action, float]]]) -> None:

        # if function.__name__ == '<lambda>':
        #     pass
        # else:
        #     pass

        module_name = function.__module__
        custom_function = {
            'function_name': function.__name__,
            'module_name': module_name  # ,
            # 'source_code': source_code
        }
        return_type = get_type_hints(function).get('return')

        if return_type == float:
            self.custom_functions["reward_functions"] += [custom_function]
        elif return_type.__origin__ == list:
            self.custom_functions["policy_functions"] += [custom_function]
        else:
            raise Exception("The return type is nor List[label] nor float")

    def get_actions(self, trajectory: trajectory_str, observation_label: label, agent_name: str = None) -> List[label]:
        actions = []
        for custom_function in self.custom_functions["policy_functions"]:
            act = load_function(custom_function)[0](trajectory.split(
                ",") if trajectory is not None else None, observation_label, agent_name)
            if act is None:
                return None
            else:
                actions += [act]

        return list(set(list(itertools.chain.from_iterable(actions))))

    def get_reward(self, trajectory: trajectory_str, agent_name: str = None) -> float:

        if type(trajectory) == list:
            if len(trajectory) > 0 and type(trajectory[0]) == tuple:
                trajectory = list(
                    set(list(itertools.chain.from_iterable([[l1, l2] for l1, l2 in trajectory]))))
            trajectory = ",".join(trajectory)
        if len(trajectory.split(",")) % 2 == 1:
            raise Exception(
                "trajectory should have full (observation, action) couples")

        reward = 0

        for custom_reward_func in self.custom_functions["reward_functions"]:
            rew = load_function(custom_reward_func)[0](trajectory.split(
                ",") if trajectory is not None else None, agent_name)
            if rew is None:
                raise Exception(
                    "Custom reward function should not return None")
            else:
                reward += rew

        # TODO: handle policy
        # for custom_policy_func in self.custom_functions["policy_functions"]:
        #     pass

        return reward

    def to_dict(self) -> Dict:
        return self.custom_functions

    def from_dict(self, data: Dict) -> None:
        self.custom_functions = data


if __name__ == '__main__':

    def manual_custom1(trajectory: trajectory, observation_label: label, agent_name: str) -> List[label]:
        if "o2" in trajectory and observation_label == "o13":
            return ["a1", "a2"]
        else:
            return ["a0"]

    def manual_custom2(trajectory: trajectory, observation_label: label, agent_name: str) -> List[label]:
        if "o3" in trajectory:
            return ["a1", "a2"]
        return ["a12", "a14"]

    def manual_rew_func(trajectory: trajectory, agent_name: str) -> float:
        if "o4" in trajectory:
            return 10
        return -10

    cf = trajectory_functions()
    cf.add_function(manual_custom1)
    cf.add_function(manual_custom2)
    print(cf.get_actions("o2,a3,o3,a3,o4,a4", "o13"))

    cf.add_function(manual_rew_func)
    print(cf.get_reward("o2,a3,o3,a3,o4,a4"))
