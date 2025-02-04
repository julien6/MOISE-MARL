import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
import optax
from flax.training.train_state import TrainState
from functools import partial
from jaxmarl.environments.warehouse_management import WarehouseManagement, Actions  # Import de l'environnement
import chex

# Configuration de l'entraînement
config = {
    "NUM_ENVS": 8,  # Nombre d'environnements parallèles
    "NUM_AGENTS": 2,  # Nombre d'agents dans l'environnement
    "GAMMA": 0.99,  # Facteur de discount
    "LAMBDA": 0.95,  # Facteur pour GAE
    "LEARNING_RATE": 3e-4,  # Taux d'apprentissage
    "NUM_UPDATES": 10,  # Nombre de mises à jour
    "EPSILON_CLIP": 0.2,  # Facteur PPO de clipping
    "ENTROPY_COEF": 0.01,  # Poids de l'entropie pour l'exploration
}

class ActorCritic(nn.Module):
    """
    Réseau d'acteur-critique pour l'environnement Warehouse Management.
    Chaque agent prend des décisions basées sur une observation de la grille locale.
    """
    action_dim: int  # Nombre d'actions possibles

    def setup(self):
        self.dense1 = nn.Dense(64)
        self.dense2 = nn.Dense(64)
        self.actor_head = nn.Dense(self.action_dim)
        self.critic_head = nn.Dense(1)

    def __call__(self, x):
        x = nn.relu(self.dense1(x))
        x = nn.relu(self.dense2(x))
        actor_logits = self.actor_head(x)
        policy = distrax.Categorical(logits=actor_logits)
        critic_value = self.critic_head(x)
        critic_value = jnp.squeeze(critic_value, axis=-1)
        return policy, critic_value


def create_train_state(rng, action_dim):
    model = ActorCritic(action_dim=action_dim)
    dummy_obs = jnp.zeros((1, 5, 5))  # Taille de la fenêtre d'observation
    params = model.init(rng, dummy_obs)
    optimizer = optax.adam(config["LEARNING_RATE"])
    return TrainState.create(apply_fn=model.apply, params=params, tx=optimizer), model


def calculate_gae(rewards, values, dones, gamma, lam):
    advs = jnp.zeros_like(rewards)
    last_adv = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        advs = advs.at[t].set(delta + gamma * lam * (1 - dones[t]) * last_adv)
        last_adv = advs[t]
    return advs, advs + values[:-1]


def ppo_update(train_state, traj_batch, gae, targets):
    def loss_fn(params):
        policy, value = train_state.apply_fn(params, traj_batch["observations"])
        log_probs = policy.log_prob(traj_batch["actions"])
        ratio = jnp.exp(log_probs - traj_batch["log_probs"])
        adv = (gae - gae.mean()) / (gae.std() + 1e-8)
        clipped_ratio = jnp.clip(ratio, 1 - config["EPSILON_CLIP"], 1 + config["EPSILON_CLIP"])
        loss_policy = -jnp.minimum(ratio * adv, clipped_ratio * adv).mean()
        loss_critic = jnp.square(targets - value).mean()
        entropy_bonus = policy.entropy().mean()
        loss = loss_policy + 0.5 * loss_critic - config["ENTROPY_COEF"] * entropy_bonus
        return loss
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(train_state.params)
    return train_state.apply_gradients(grads=grads)


def train(env, train_state):
    rng = jax.random.PRNGKey(42)
    for update in range(config["NUM_UPDATES"]):
        observations, actions, rewards, dones, values, log_probs = [], [], [], [], [], []
        for _ in range(config["NUM_ENVS"]):
            obs, state = env.reset(rng)
            actions_dict = {}
            for agent in env.agents:
                policy, value = train_state.apply_fn(train_state.params, obs[agent])
                action = policy.sample(seed=rng)
                actions_dict[agent] = action
                log_prob = policy.log_prob(action)
                observations.append(obs[agent])
                actions.append(action)
                values.append(value)
                log_probs.append(log_prob)
            obs, state, reward_dict, done_dict, _ = env.step_env(rng, state, actions_dict)
            rewards.append([reward_dict[agent] for agent in env.agents])
            dones.append([done_dict[agent] for agent in env.agents])
        traj_batch = {
            "observations": jnp.stack(observations),
            "actions": jnp.stack(actions),
            "log_probs": jnp.stack(log_probs),
        }
        gae, targets = calculate_gae(rewards, values, dones, config["GAMMA"], config["LAMBDA"])
        train_state = ppo_update(train_state, traj_batch, gae, targets)
    return train_state


if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    env = WarehouseManagement(num_agents=config["NUM_AGENTS"])
    train_state, model = create_train_state(key, action_dim=len(Actions))
    train_state = train(env, train_state)
