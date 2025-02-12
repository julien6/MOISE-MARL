import json
import jax.experimental
import orbax.checkpoint as ocp
import jax
import optax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import distrax
import jaxmarl
import matplotlib.pyplot as plt
import hydra
import wandb
import pickle
import os
import imageio
import keyboard

from functools import partial
from typing import Sequence
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
from jaxmarl.wrappers.baselines import LogWrapper
from omegaconf import OmegaConf
from jaxmarl.environments.warehouse_management import WarehouseManagement, WarehouseManagementVisualizer


CHECKPOINT_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "checkpoints")
SAVE_INTERVAL = 10000  # Sauvegarde tous les 10k steps
TEST_EPISODES = 10      # Nombre d'épisodes de test

# Initialisation du gestionnaire de checkpoints avec Orbax
options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
checkpointer = ocp.PyTreeCheckpointer()
manager = ocp.CheckpointManager(CHECKPOINT_DIR, checkpointer, options)


# =======================
# 🎯 Détection des GPUs
# =======================
devices = jax.devices()
# ✅ Prend en compte "cuda"
use_pmap = "gpu" in str(jax.extend.backend.get_backend().platform).lower()

if use_pmap:
    print(f"🚀 {len(devices)} GPU(s) détecté(s) : {[str(d) for d in devices]}")
    num_gpus = sum(1 for d in devices if "gpu" in str(d).lower() or "cuda" in str(d).lower(
    ) or "tesla" in str(d).lower())
    if num_gpus < 2:
        print("Juste un GPU, exécution sur CPU uniquement.")
    else:
        print("Plus que un GPU, parallélisation avec 'pmap'")
else:
    print("⚠️ Aucun GPU détecté, exécution sur CPU uniquement.")


def save_checkpoint(train_state, step):
    """
    Sauvegarde un checkpoint du modèle avec Orbax.

    Arguments:
        - train_state (TrainState) : état actuel du modèle à sauvegarder.
        - step (int) : numéro de l'étape d'entraînement.
    """
    print(f"💾 Sauvegarde du checkpoint à l'étape {step}...")

    # Utilisation de la nouvelle API pour sauvegarder avec Orbax
    manager.save(step, train_state[0])
    print(f"✅ Checkpoint sauvegardé à l'étape {step}")


def load_checkpoint():
    """
    Charge le dernier checkpoint disponible avec Orbax.

    Retourne:
        - train_state mis à jour avec les paramètres du dernier checkpoint.
        - step correspondant au checkpoint chargé.
        - None si aucun checkpoint n'est trouvé.
    """
    step = manager.latest_step()

    if step is None:
        print("⚠️ Aucun checkpoint trouvé, initialisation à zéro.")
        return None, 0  # Aucun checkpoint disponible

    print(f"🔄 Chargement du checkpoint {step}...")

    # Charger le train_state à partir du dernier checkpoint
    train_state = manager.restore(step)

    print(f"✅ Checkpoint {step} chargé avec succès.")
    return train_state, step


def select_best_model(metrics):
    """
    Sélectionne le meilleur modèle parmi les runs effectués avec différentes graines.

    Arguments :
        - metrics (dict) : Dictionnaire contenant les métriques des `NUM_SEEDS` runs.
    Retourne :
        - best_seed_idx : L'index de la meilleure graine.
        - best_score : Score moyen final du meilleur modèle.
    """

    metrics = {k: v.tolist() for k, v in jax.device_get(metrics).items()}

    # Extraction des scores finaux sur les 10 derniers updates pour éviter les fluctuations aléatoires
    final_returns = np.mean(metrics["returned_episode_returns"], axis=1)

    # Critère principal : choisir le modèle ayant le plus haut score moyen final
    best_seed_idx = np.argmax(final_returns)
    best_score = final_returns[best_seed_idx]

    # 🔎 Optionnel : Vérification d'autres métriques pour confirmer la robustesse du modèle
    best_actor_loss = np.mean(
        metrics["actor_loss"][best_seed_idx])  # Perte de l'acteur
    best_critic_loss = np.mean(
        metrics["critic_loss"][best_seed_idx])  # Perte du critique
    best_entropy = np.mean(metrics["entropy"][best_seed_idx])  # Entropie
    best_total_loss = np.mean(
        metrics["total_loss"][best_seed_idx])  # Total loss

    print(f"🏆 Meilleur modèle : Seed {best_seed_idx}")
    print(f"🔹 Score moyen final : {best_score:.2f}")
    print(f"🎭 Actor Loss : {best_actor_loss:.6f}")
    print(f"📉 Critic Loss : {best_critic_loss:.2f}")
    print(f"🎲 Entropy : {best_entropy:.3f}")
    print(f"📊 Total Loss : {best_total_loss:.2f}")

    return int(best_seed_idx), best_score


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

        # ===============================
        # 🚀 1. Construction de l'Acteur (Actor)
        # ===============================
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

        # ===============================
        # 📊 2. Construction du Critique (Critic)
        # ===============================
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

        # 🔹 jnp.squeeze(critic, axis=-1):
        # - La sortie de nn.Dense(1) est une matrice de taille (batch_size, 1).
        # - On retire la dimension inutile 1 pour obtenir un vecteur de taille (batch_size,).
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

    # Booléen indiquant si l'épisode est terminé (1: terminé, 0: en cours)
    done: jnp.ndarray
    action: jnp.ndarray  # Action effectuée par l'agent
    value: jnp.ndarray  # Valeur estimée de l'état par le Critique
    reward: jnp.ndarray  # Récompense reçue après avoir pris l'action
    log_prob: jnp.ndarray  # Log-probabilité de l'action sélectionnée par l'Acteur
    obs: jnp.ndarray  # Observation reçue par l'agent (une partie de la grille)
    # Informations supplémentaires fournies par l’environnement (si applicable)
    info: jnp.ndarray


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
    max_dim = max([x[a].flatten().shape[0] for a in agent_list])

    # Fonction interne pour compléter (pad) les données si elles sont plus courtes que max_dim.
    def pad(z, length):
        return jnp.concatenate([z, jnp.zeros(length - z.shape[0])], -1)

    # Vérification
    for a in agent_list:
        if x[a].flatten().shape[0] != max_dim:
            print(
                f"⚠️ Warning: Observation shape mismatch for {a}, padding required!")

    # Empile les données des agents dans une matrice unique, avec padding si nécessaire.
    x = jnp.stack([pad(x[a].flatten(), max_dim) for a in agent_list])

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
    x = x.reshape((num_actors, num_envs, -1))

    # 🛠 Correction : s'assurer que chaque action est un scalaire int32 et non un tenseur (16,1)
    return {a: x[i, :, 0] for i, a in enumerate(agent_list)}


def make_train(config, rng_init):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        # config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
        5000 // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env, replace_info=True)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"]
                                * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    # INIT NETWORK
    network = ActorCritic(env.action_space(
        env.agents[0]).n, activation=config["ACTIVATION"])
    obs_dim = np.prod(env.observation_space(
        env.agents[0]).shape, dtype=int)  # Aplatissement correct
    init_x = jnp.zeros(obs_dim, dtype=jnp.float32)

    network_params = network.init(rng_init, init_x)

    if config["ANNEAL_LR"]:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(optax.clip_by_global_norm(
            config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))

    @jax.jit
    def train(rng, train_state=None, start_step=0):

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])

        obsv, env_state = jax.vmap(env.reset)(reset_rng)

        # TRAIN LOOP

        # COLLECT TRAJECTORIES
        @jax.jit
        def _env_step(runner_state, unused):
            train_state, env_state, last_obs, update_count, rng = runner_state

            obs_batch = batchify(last_obs, env.agents,
                                 config["NUM_ACTORS"])

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)

            pi, value = network.apply(train_state.params, obs_batch)

            action = pi.sample(seed=_rng)

            log_prob = pi.log_prob(action)
            env_act = unbatchify(action, env.agents,
                                 config["NUM_ENVS"], env.num_agents)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(env.step)(
                rng_step, env_state, env_act,
            )

            info = jax.tree.map(lambda x: x.reshape(
                (config["NUM_ACTORS"])), info)
            transition = Transition(
                batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                action,
                value,
                batchify(reward, env.agents,
                         config["NUM_ACTORS"]).squeeze(),
                log_prob,
                obs_batch,
                info,
            )
            runner_state = (train_state, env_state,
                            obsv, update_count, rng)
            return runner_state, transition

        # UPDATE MINIBATCH

        @jax.jit
        def _update_minbatch(train_state, batch_info):
            traj_batch, advantages, targets = batch_info

            def _loss_fn(params, traj_batch, gae, targets):
                # RERUN NETWORK

                pi, value = network.apply(params, traj_batch.obs)
                log_prob = pi.log_prob(traj_batch.action)

                # CALCULATE VALUE LOSS
                value_pred_clipped = traj_batch.value + (
                    value - traj_batch.value
                ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                value_losses = jnp.square(value - targets)
                value_losses_clipped = jnp.square(
                    value_pred_clipped - targets)
                value_loss = (
                    0.5 * jnp.maximum(value_losses,
                                      value_losses_clipped).mean()
                )

                # CALCULATE ACTOR LOSS
                ratio = jnp.exp(log_prob - traj_batch.log_prob)
                gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                loss_actor1 = ratio * gae
                loss_actor2 = (
                    jnp.clip(
                        ratio,
                        1.0 - config["CLIP_EPS"],
                        1.0 + config["CLIP_EPS"],
                    )
                    * gae
                )
                loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                loss_actor = loss_actor.mean()
                entropy = pi.entropy().mean()

                total_loss = (
                    loss_actor
                    + config["VF_COEF"] * value_loss
                    - config["ENT_COEF"] * entropy
                )
                return total_loss, (value_loss, loss_actor, entropy, ratio)

            grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
            total_loss, grads = grad_fn(
                train_state.params, traj_batch, advantages, targets
            )
            train_state = train_state.apply_gradients(grads=grads)

            loss_info = {
                "total_loss": total_loss[0],
                "actor_loss": total_loss[1][1],
                "critic_loss": total_loss[1][0],
                "entropy": total_loss[1][2],
                "ratio": total_loss[1][3],
            }

            return train_state, loss_info

        # UPDATE NETWORK

        @jax.jit
        def _update_epoch(update_state, unused):

            train_state, traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)
            batch_size = config["MINIBATCH_SIZE"] * \
                config["NUM_MINIBATCHES"]
            assert (
                batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
            ), "batch size must be equal to number of steps * number of actors"
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)

            batch = jax.tree.map(lambda x: x.reshape(
                (-1,) + x.shape[2:]), batch)

            shuffled_batch = jax.tree.map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree.map(
                lambda x: jnp.reshape(
                    x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )
            train_state, loss_info = jax.lax.scan(
                _update_minbatch, train_state, minibatches)
            update_state = (train_state, traj_batch,
                            advantages, targets, rng)
            return update_state, loss_info

        # GET ADVANTAGES

        @jax.jit
        def _get_advantages(gae_and_next_value, transition):
            gae, next_value = gae_and_next_value
            done, value, reward = (
                transition.done,
                transition.value,
                transition.reward,
            )
            delta = reward + config["GAMMA"] * \
                next_value * (1 - done.astype(jnp.float32)) - value
            gae = (
                delta
                + config["GAMMA"] *
                config["GAE_LAMBDA"] *
                (1 - done.astype(jnp.float32)) * gae
            )
            return (gae, value), gae

        # UPDATE STEP

        @jax.jit
        def _update_step(runner_state, unused):

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"])

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, update_count, rng = runner_state
            last_obs_batch = batchify(
                last_obs, env.agents, config["NUM_ACTORS"])
            _, last_val = network.apply(train_state.params, last_obs_batch)

            def _calculate_gae(traj_batch, last_val):
                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=8,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            update_count = update_count + 1

            def callback(metric):
                # Si nécessaire, on peut aussi envoyer ces métriques à WandB
                if metric["update_count"] % 10 == 0 and config["WANDB_MODE"] != "disabled":
                    wandb.log(metric, step=metric["update_count"])

                    print("\n")
                    # Définir les colonnes à afficher et leur largeur (en caractères)
                    columns = [
                        ("update_count", 12),
                        ("total_loss", 12),
                        ("returned_episode", 20),
                        ("returned_episode_returns", 25),
                        ("actor_loss", 12),
                        ("critic_loss", 12)
                    ]

                    # Afficher l'en-tête de la table lorsque update_count vaut 1 (ou lors du premier appel)
                    header = "| " + \
                        " | ".join(
                            [f"{col_name:^{width}}" for col_name, width in columns]) + " |"
                    print(header)
                    print("-" * len(header))

                    # Construire et afficher la ligne correspondant aux métriques actuelles
                    row = "| " + \
                        " | ".join(
                            [f"{str(metric[col_name]):^{width}}" for col_name, width in columns]) + " |"
                    print(row)

            r0 = {"ratio0": loss_info["ratio"][0, 0].mean()}
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            metric = jax.tree.map(lambda x: x.mean(), metric)
            metric["update_count"] = update_count
            metric["env_step"] = update_count * \
                config["NUM_STEPS"] * config["NUM_ENVS"]
            metric = {**metric, **loss_info, **r0}

            jax.experimental.io_callback(callback, None, metric)

            runner_state = (train_state, env_state,
                            last_obs, update_count, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        if not train_state:
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx)

        runner_state = (train_state, env_state, obsv, start_step, _rng)

        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"]
                                            )

        return runner_state, metric

    return train


def make_test(config, rng_init, network_params=None, start_step=0):
    """
    Exécute plusieurs épisodes de test en enregistrant les récompenses et les états.

    Arguments :
    - config (dict) : Configuration contenant les paramètres de l'environnement.
    - rng_init : Graine aléatoire pour l'initialisation.
    - network_params (any) : Paramètres du réseau entraîné (ou None pour réinitialiser).
    - start_step (int) : Numéro de l'étape de départ.

    Retourne :
    - reward_sums (list) : Liste des sommes de récompenses pour chaque épisode.
    - all_states (list) : Liste des séquences d'états pour chaque épisode.
    """

    TEST_EPISODES = config["TEST_EPISODES"]
    NUM_STEPS = config["NUM_STEPS"]

    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    # INIT NETWORK
    network = ActorCritic(env.action_space(
        env.agents[0]).n, activation=config["ACTIVATION"])
    obs_dim = np.prod(env.observation_space(
        env.agents[0]).shape, dtype=int)  # Aplatissement correct

    if network_params is None:
        network_params = network.init(
            rng_init, jnp.zeros(obs_dim, dtype=jnp.float32))

    # Initialisation des listes de stockage
    reward_sums = []  # Stocke la somme des récompenses de chaque épisode
    all_states = []   # Stocke la liste des états pour chaque épisode

    key = jax.random.PRNGKey(0)
    key_e = jax.random.split(key, TEST_EPISODES)

    for episode in range(TEST_EPISODES):

        key, key_r, key_a = jax.random.split(key_e[episode], 3)

        last_obs, state = env.reset(key_r)

        reward_seq = []  # Liste des récompenses pour cet épisode
        state_seq = []   # Liste des états pour cet épisode

        for step in range(NUM_STEPS):

            state_seq.append(state)
            # Iterate random keys and sample actions
            key, key_s, key_a = jax.random.split(key, 3)

            obs_batch = batchify(last_obs, env.agents, env.num_agents)

            # SELECT ACTION
            pi, value = network.apply(network_params, obs_batch)
            action = pi.sample(seed=key_a)

            actions = unbatchify(action, env.agents, 1, env.num_agents)
            actions = jax.device_get({agent: jax.device_get(
                action).item() for agent, action in actions.items()})

            # Step environment
            last_obs, state, rewards, dones, infos = env.step(
                key_s, state, actions)

            # Stocke la récompense obtenue
            reward_seq.append(
                sum([v.item() for v in rewards.values()]) // env.num_agents)

        # Stocker la somme des récompenses de l'épisode
        reward_sums.append(sum(reward_seq))

        # Stocker la séquence complète des états de l'épisode
        all_states.append(state_seq)

    return reward_sums, all_states


@hydra.main(version_base=None, config_path="config", config_name="ippo_ff_warehouse_management")
def main(config):
    """Fonction principale de l'entraînement IPPO-FF sur Warehouse Management."""

    # Charger un checkpoint existant
    checkpoint = load_checkpoint()  # Par défaut, charge le dernier
    if checkpoint is not None:
        train_state, start_step = checkpoint
        print(f"🔄 Reprise de l'entraînement depuis le step {start_step}")
    else:
        train_state = None
        start_step = 0

    # 🔹 Convertit la configuration Hydra en dictionnaire standard
    config = OmegaConf.to_container(config, resolve=True)

    # 🚀 Initialisation Weights & Biases si activé
    if config["WANDB_MODE"] != "disabled":
        wandb.init(
            entity=config["ENTITY"],
            project=config["PROJECT"],
            tags=["IPPO", "FF", "Warehouse Management"],
            config=config,
            mode=config["WANDB_MODE"],
        )

    rng = jax.random.PRNGKey(int(config["SEED"]))

    # Sinon, plusieurs graines pour vmap
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    train_jit = jax.jit(make_train(config, rng))
    train_fn = jax.vmap(train_jit, in_axes=0)
    runner_state, metrics = train_fn(rngs)
    last_step = runner_state[3]

    print(f"First training finished at {last_step[0]}, starting second one =============")

    save_checkpoint(runner_state, last_step[0])
    print(type(runner_state[0]))
    train_state_1, last_step_1 = load_checkpoint()
    print(type(train_state_1))
    runner_state, metrics = train_fn(rngs, train_state_1, last_step) # jax.numpy.array([last_step_1] * int(config["SEED"])))
    print("Second training finished, starting third one =============")

    # best_seed_index, best_score = select_best_model(metrics)
    # trained_params = jax.tree.map(
    #     lambda x: x[best_seed_index], runner_state[0].params)

    # plt.plot(metrics["returned_episode_returns"].mean(axis=0))
    # plt.savefig(f"ippo_ff_{config['ENV_NAME']}.png")
    # plt.xlabel("Updates")
    # plt.ylabel("Returns")
    # plt.title(f"IPPO-FF={config['ENV_NAME']}")
    # # plt.show()

    # # 🚀 Ajout des logs sur WandB si activé
    # if config["WANDB_MODE"] != "disabled" and "returned_episode_returns" in metrics:
    #     updates_x = jnp.arange(
    #         metrics["returned_episode_returns"][0].shape[0])
    #     returns_table = jnp.stack(
    #         [updates_x, metrics["returned_episode_returns"].mean(axis=0)], axis=1)
    #     returns_table = wandb.Table(
    #         data=returns_table.tolist(), columns=["updates", "returns"])

    #     wandb.log({
    #         "returns_plot": wandb.plot.line(returns_table, "updates", "returns", title="Returns vs Updates"),
    #         "final_returns": metrics["returned_episode_returns"][:, -1].mean(),
    #     })

    # rewards, states = make_test(config, rng, trained_params, start_step)
    # wz = WarehouseManagementVisualizer()
    # wz.animate_eps(states, "ippo_ff_warehouse_management.gif")


if __name__ == '__main__':
    main()
