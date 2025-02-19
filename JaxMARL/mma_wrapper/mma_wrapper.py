import jax
import jax.numpy as jnp
import json
import chex

from typing import Dict
from functools import partial
from flax import struct
from typing import Tuple, Optional
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.wrappers.baselines import JaxMARLWrapper
from jaxmarl.environments.warehouse_management import WarehouseManagement

from jaxmarl.environments.multi_agent_env import State

# TODO : La classe MMAWrapper doit pouvoir envelopper un environnement MultiAgentEnv en prenant en compte les spécifications organisationnelles.
# Les spécifications organisationnelles sont définis dans un dictionnaire modélisant le modèle MOISE+
# La MMAWrapper doit pouvoir appliquer de manière effective les spécifications organisationnelles durant l'apprentissage en modifiant la récompense à la volée et/ou les actions des agents (soit on a accès à la distribution de probabilité des actions et on choisit l'action autorisée avec la plus forte probabilité soit on n'y a pas accès et on choisit une action autorisée aléatoirement).


class MMAWrapper(JaxMARLWrapper):
    def __init__(self, env: MultiAgentEnv, org_spec: dict):
        super().__init__(env)
        self.org_spec = org_spec
        self._last_obs = None
        self._last_history = None

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def _batchify_floats(self, x: dict):
        return jnp.stack([x[a] for a in self._env.agents])

    def _apply_rag(self, actions: Dict[str, chex.Array]) -> Dict[str, chex.Array]:
        self._last_history
        self._last_obs

        actions: Dict[str, chex.Array] = {},
        return actions

    def _apply_rrg(self, rewards) -> Dict[str, float]:
        self._last_obs
        self._last_history

        rewards: Dict[str, float] = {}
        return None

    def _apply_grg(self, rewards)-> Dict[str, float]:
        self._last_history

        rewards: Dict[str, float] = {}
        return None

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        self.last_obs, _ = self._env.reset(key)

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
        reset_state: Optional[State] = None,
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Performs step transitions in the environment. Resets the environment if done.
        To control the reset state, pass `reset_state`. Otherwise, the environment will reset randomly."""

        # Replace possibly unauthorized actions by authorized ones depending on the constraint hardness in hard constraint roles
        actions = self._apply_rag(actions)

        # Proceed the regular state step
        obs, states, rewards, dones, infos = self._env.step(
            key, state, actions)

        # Adjust the rewards according to soft constraint roles and goals in missions
        rewards = self._apply_rrg(rewards)
        rewards = self._apply_grg(rewards)

        self._last_history += [(self._last_obs, actions, rewards)]
        self._last_obs = obs

        return obs, states, rewards, dones, infos
