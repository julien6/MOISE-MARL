from marllib import marl

# prepare env
env = marl.make_env(environment_name="mpe", map_name="simple_spread")
# env = marl.make_env(environment_name="mcy", map_name="moving_company")
# env = marl.make_env(environment_name="wmt", map_name="warehouse_management")
# env = marl.make_env(environment_name="overcooked", map_name="asymmetric_advantages")
# env = marl.make_env(environment_name="cyborg", map_name="cage3")


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
