import imageio
import json

from marllib import marl
from marllib.marl.algos.core.CC.mappo import MAPPOTrainer


# prepare env
env = marl.make_env(environment_name="mpe",
                    map_name="simple_world_comm", organizational_model=None)  # mpe_model)


# initialize algorithm with appointed hyper-parameters
# mappo = marl.algos.mappo(hyperparam_source="mpe")
mappo = marl.algos.mappo(hyperparam_source="test")

# build agent model based on env + algorithms + user preference
model = marl.build_model(
    env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})


conf = json.load(open("./exp_results/mappo_mlp_simple_world_comm/MAPPOTrainer_mpe_simple_world_comm_f5f5f_00000_0_2025-03-09_18-28-48/params.json", "r"))

trainer = MAPPOTrainer(config=conf).restore("./exp_results/mappo_mlp_simple_world_comm/MAPPOTrainer_mpe_simple_world_comm_f5f5f_00000_0_2025-03-09_18-28-48/checkpoint_000020/checkpoint-20")

print("okkkkkkkk")

# 6) Rollout personnalisé
frames = []
NUM_EPISODES = 2
for ep in range(NUM_EPISODES):
    obs = env.reset()
    done = {"__all__": False}
    while not done["__all__"]:
        # Rendu
        frame = env.render(mode="rgb_array")
        frames.append(frame)

        # Multi-agent : calcul d'actions
        actions = {}
        for agent_id, agent_obs in obs.items():
            # 'policy_id' selon la config (shared = "default_policy" ou "policy_0", etc.)
            actions[agent_id] = trainer.compute_action(agent_obs, policy_id="default_policy")
        
        obs, rew, done, info = env.step(actions)

env.close()

# 7) On génère le GIF
imageio.mimsave("rollout.gif", frames, fps=10)
