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

from typing import Sequence
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
from jaxmarl.wrappers.baselines import MPELogWrapper as LogWrapper
from omegaconf import OmegaConf
from jaxmarl.environments.warehouse_management import WarehouseManagement


# =======================
# 🎯 Détection des GPUs
# =======================
devices = jax.devices()
use_pmap = any(d.device_kind == "gpu" for d in devices)  # Vérifie la présence d'un GPU

if use_pmap:
    print(f"🚀 {len(devices)} GPU(s) détecté(s), activation de `pmap`.")
else:
    print("⚠️ Aucun GPU détecté, exécution sur CPU uniquement (sans `pmap`).")


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
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
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
        env.agents[0]).shape, dtype=int)  # Aplatissement correctœœ
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

    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    # 🔹 Répliquer l'état d'entraînement sur tous les GPU si `pmap` est activé
    if use_pmap:
        train_state = jax.device_put_replicated(train_state, devices)
        print(
            f"✅ Réplication train_state pour {len(devices)} GPU(s) : {jax.tree_map(lambda x: x.shape, train_state)}")

    def train(rng):

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
            batch = jax.tree.map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree.map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree.map(
                lambda x: jnp.reshape(
                    x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )
            train_state, loss_info = jax.lax.scan(_update_minbatch, train_state, minibatches)
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
            
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])

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

            def callback(metric):
                if wandb.run is not None:  # Vérifie si wandb.init() a été appelé
                    if metric["update_step"] % 10 == 0:  # Log tous les 10 updates
                        wandb.log(metric, step=metric["update_step"])

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            update_count = update_count + 1

            def log_fn(_):
                mean_reward = jnp.mean(traj_batch.reward)
                jax.debug.print(
                    "[Update {x}] Mean Reward: {y}", x=update_count, y=mean_reward)
                return None

            # jax.lax.cond(update_count % 10 == 0, log_fn, lambda _: None, None)

            r0 = {"ratio0": loss_info["ratio"][0, 0].mean()}
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            metric = jax.tree.map(lambda x: x.mean(), metric)

            metric["update_step"] = update_count
            metric["env_step"] = update_count * \
                config["NUM_STEPS"] * config["NUM_ENVS"]

            metric = {**metric, **loss_info, **r0}

            jax.experimental.io_callback(callback, None, metric)
            runner_state = (train_state, env_state,
                            last_obs, update_count, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, 0, _rng)
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metric}

    return train


@hydra.main(version_base=None, config_path="config", config_name="ippo_ff_warehouse_management")
def main(config):
    """Fonction principale de l'entraînement IPPO-FF sur Warehouse Management."""

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

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    train_jit = jax.jit(make_train(config, rng))

    if use_pmap:
        train_fn = jax.pmap(jax.vmap(train_jit, in_axes=0), axis_name="batch")
    else:
        train_fn = jax.vmap(train_jit, in_axes=0)

    out = train_fn(rngs)

    plt.plot(out["metrics"]["returned_episode_returns"].mean(axis=0))
    plt.savefig(f"ippo_ff_{config['ENV_NAME']}.png")
    plt.xlabel("Updates")
    plt.ylabel("Returns")
    plt.title(f"IPPO-FF={config['ENV_NAME']}")
    # plt.show()

    # 🚀 Ajout des logs sur WandB si activé
    if config["WANDB_MODE"] != "disabled" and "returned_episode_returns" in out["metrics"]:
        updates_x = jnp.arange(
            out["metrics"]["returned_episode_returns"][0].shape[0])
        returns_table = jnp.stack(
            [updates_x, out["metrics"]["returned_episode_returns"].mean(axis=0)], axis=1)
        returns_table = wandb.Table(
            data=returns_table.tolist(), columns=["updates", "returns"])

        wandb.log({
            "returns_plot": wandb.plot.line(returns_table, "updates", "returns", title="Returns vs Updates"),
            "final_returns": out["metrics"]["returned_episode_returns"][:, -1].mean(),
        })


if __name__ == '__main__':
    main()
