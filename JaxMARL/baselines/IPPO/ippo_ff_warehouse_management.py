from typing import Sequence
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
from flax.linen.initializers import orthogonal, constant
from jaxmarl.wrappers.baselines import MPELogWrapper as LogWrapper 
from omegaconf import OmegaConf

import jax
import optax
import jax.numpy as jnp
import flax.linen as nn
import distrax
import numpy as np
import distrax
import jaxmarl
import matplotlib.pyplot as plt
import hydra
import wandb


class ActorCritic(nn.Module):
    """
    Réseau neuronal utilisé pour IPPO dans l'environnement Warehouse Management.
    
    Ce modèle suit l'architecture d'un réseau de type Actor-Critic :
    - **Actor** : prédit une distribution de probabilités sur les actions.
    - **Critic** : estime la valeur de l'état actuel.

    Paramètres :
    - action_dim : Nombre d'actions possibles pour un agent.
    - activation : Type de fonction d'activation à utiliser ("relu" ou "tanh").
    """
    
    action_dim: int  # Nombre d'actions possibles
    activation: str = "tanh"  # Activation par défaut : tanh

    @nn.compact
    def __call__(self, x):
        """
        Forward pass du réseau.

        Entrée :
        - x : l'observation d'un agent (une portion de la grille centrée sur lui)

        Sorties :
        - pi : une distribution de probabilités sur les actions possibles.
        - critic : une estimation de la valeur de l'état actuel.
        """

        # 🔹 Définition de la fonction d'activation
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh  # Par défaut, tanh est utilisé.

        ## ===============================
        ## 🚀 1. Construction de l'Acteur (Actor)
        ## ===============================
        # Cet acteur génère une distribution de probabilité sur les actions.

        # Première couche dense (entrée → couche cachée)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)  # Activation

        # Deuxième couche dense
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)  # Activation

        # Dernière couche : produit des logits pour les actions
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        # 🔹 Création de la distribution de probabilité sur les actions possibles
        pi = distrax.Categorical(logits=actor_mean)

        ## ===============================
        ## 📊 2. Construction du Critique (Critic)
        ## ===============================
        # Ce Critique prédit la "valeur" de l'état actuel.

        # Première couche dense
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)

        # Deuxième couche dense
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)

        # Dernière couche : sortie unique représentant la valeur de l'état
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        # 🔹 `jnp.squeeze(critic, axis=-1)`: 
        # - La sortie de `nn.Dense(1)` est une matrice de taille (batch_size, 1).
        # - On retire la dimension inutile `1` pour obtenir un vecteur de taille (batch_size,).
        critic = jnp.squeeze(critic, axis=-1)

        return pi, critic  # L'acteur (pi) et la valeur de l'état (critic).

class Transition(NamedTuple):
    """
    Structure de données pour stocker les informations d'une transition.

    Chaque transition correspond à un pas dans l'environnement et est utilisée
    pour entraîner l'agent IPPO.

    Attributs :
    - done : Indique si l'épisode est terminé (booléen, 1 pour "fini", 0 sinon).
    - action : Action choisie par l'agent à cet état.
    - value : Estimation de la valeur de l'état par le Critique.
    - reward : Récompense reçue après avoir pris l'action.
    - log_prob : Log-probabilité de l'action sélectionnée par l'Acteur.
    - obs : Observation actuelle de l'agent (son champ de vision).
    - info : Autres informations éventuelles renvoyées par l’environnement.
    """
    
    done: jnp.ndarray  # Booléen indiquant si l'épisode est terminé (1: terminé, 0: en cours)
    action: jnp.ndarray  # Action effectuée par l'agent
    value: jnp.ndarray  # Valeur estimée de l'état par le Critique
    reward: jnp.ndarray  # Récompense reçue après avoir pris l'action
    log_prob: jnp.ndarray  # Log-probabilité de l'action sélectionnée par l'Acteur
    obs: jnp.ndarray  # Observation reçue par l'agent (une partie de la grille)
    info: jnp.ndarray  # Informations supplémentaires fournies par l’environnement (si applicable)


def batchify(x: dict, agent_list, num_actors):
    """
    Convertit un dictionnaire de données par agent en un unique tenseur JAX.
    
    Args :
    - x (dict) : Dictionnaire contenant les observations ou actions des agents.
                 Clés = noms des agents, valeurs = tableaux JAX.
    - agent_list (list) : Liste des agents (ex : ["agent_0", "agent_1", ...]).
    - num_actors (int) : Nombre total d'agents.

    Returns :
    - jnp.ndarray : Matrice (num_actors, feature_dim) regroupant les données de tous les agents.
    """
    # Trouve la plus grande dimension parmi les observations de chaque agent.
    max_dim = max([x[a].shape[-1] for a in agent_list])

    # Fonction interne pour compléter (pad) les données si elles sont plus courtes que max_dim.
    def pad(z, length):
        return jnp.concatenate([z, jnp.zeros(z.shape[:-1] + (length - z.shape[-1],))], -1)

    # Empile les données des agents dans une matrice unique, avec padding si nécessaire.
    x = jnp.stack([x[a] if x[a].shape[-1] == max_dim else pad(x[a], max_dim) for a in agent_list])
    
    # Redimensionne le tenseur pour respecter le nombre total d'acteurs.
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    """
    Sépare un tenseur batché en un dictionnaire par agent.

    Args :
    - x (jnp.ndarray) : Matrice batchée (num_actors, num_envs, feature_dim).
    - agent_list (list) : Liste des agents (ex : ["agent_0", "agent_1", ...]).
    - num_envs (int) : Nombre d’environnements parallèles.
    - num_actors (int) : Nombre total d'agents.

    Returns :
    - dict : Dictionnaire où chaque agent a ses propres données sous forme de tableau JAX.
    """
    # Reshape pour retrouver la structure originale (num_actors, num_envs, feature_dim).
    x = x.reshape((num_actors, num_envs, -1))
    
    # Reconstruit un dictionnaire où chaque agent récupère ses données.
    return {a: x[i] for i, a in enumerate(agent_list)}




===============
def create_train_state(rng, action_dim):
    """
    Initialise le train state et les paramètres du réseau pour IPPO.
    """
    model = ActorCritic(action_dim=action_dim)
    
    # Création d'un batch factice d'observation (taille de la vision de l'agent)
    dummy_obs = jnp.zeros((1, 5, 5))  # Taille de la fenêtre d'observation (VIEW_SCOPE * 2 + 1)
    
    params = model.init(rng, dummy_obs)
    optimizer = optax.adam(config["LEARNING_RATE"])

    return TrainState.create(apply_fn=model.apply, params=params, tx=optimizer), model

def calculate_gae(rewards, values, dones, gamma, lam):
    """
    Calcule l'Avantage GAE pour l'agent.
    """
    advs = jnp.zeros_like(rewards)
    last_adv = 0.0

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        advs = advs.at[t].set(delta + gamma * lam * (1 - dones[t]) * last_adv)
        last_adv = advs[t]

    return advs, advs + values[:-1]

def ppo_update(train_state, traj_batch, gae, targets):
    """
    Effectue une mise à jour IPPO avec les batchs de trajectoires et les avantages.
    """
    def loss_fn(params):
        policy, value = train_state.apply_fn(params, traj_batch["observations"])
        log_probs = policy.log_prob(traj_batch["actions"])
        ratio = jnp.exp(log_probs - traj_batch["log_probs"])

        # Perte PPO avec clipping
        adv = (gae - gae.mean()) / (gae.std() + 1e-8)
        clipped_ratio = jnp.clip(ratio, 1 - config["EPSILON_CLIP"], 1 + config["EPSILON_CLIP"])
        loss_policy = -jnp.minimum(ratio * adv, clipped_ratio * adv).mean()

        # Perte du Critique
        loss_critic = jnp.square(targets - value).mean()

        # Bonus d'entropie pour exploration
        entropy_bonus = policy.entropy().mean()
        loss = loss_policy + 0.5 * loss_critic - config["ENTROPY_COEF"] * entropy_bonus

        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(train_state.params)

    return train_state.apply_gradients(grads=grads)

def train(env, train_state):
    """
    Entraîne les agents dans l'environnement Warehouse Management.
    """
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
