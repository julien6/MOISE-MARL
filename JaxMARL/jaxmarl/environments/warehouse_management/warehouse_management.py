from enum import IntEnum
import time
import jax
import jax.numpy as jnp
import numpy as onp
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.mpe.default_params import *
import chex
from jaxmarl.environments.spaces import Box, Discrete
from flax import struct
from typing import Tuple, Optional, Dict
from functools import partial

import matplotlib.pyplot as plt
import matplotlib


class Actions(IntEnum):
    noop = 0
    up = 1
    down = 2
    left = 3
    right = 4
    pick = 5
    drop = 6


MOVE_ACTIONS = jnp.array([
    [0, 0],   # noop
    [-1, 0],  # up
    [1, 0],   # down
    [0, -1],  # left
    [0, 1],   # right
])


class Items(IntEnum):
    empty = 0
    wall = 1
    agent_without_object = 2
    agent_with_primary = 3
    agent_with_secondary = 4
    primary_object = 5
    secondary_object = 6
    empty_input = 7
    input_with_object = 8
    empty_input_craft = 9
    input_craft_with_object = 10
    empty_output_craft = 11
    output_craft_with_object = 12
    empty_output = 13
    output_with_object = 14


TRANSFORMS = jnp.array([
    jnp.array([Items.input_with_object.value,
               Items.agent_with_primary.value, Items.empty_input.value]),
    jnp.array([Items.input_craft_with_object.value,
               Items.agent_with_primary.value, Items.empty_input_craft.value]),
    jnp.array([Items.output_with_object.value,
               Items.agent_with_secondary.value, Items.empty_output.value]),
    jnp.array([Items.output_craft_with_object.value,
               Items.agent_with_secondary.value, Items.empty_output_craft.value]),
    jnp.array([Items.primary_object.value,
              Items.agent_with_primary.value, Items.empty.value]),
    jnp.array([Items.secondary_object.value,
              Items.agent_with_secondary.value,  Items.empty.value])
])


@struct.dataclass
class State:
    """Basic MPE State"""
    agent_positions: chex.Array
    agent_states: chex.Array
    grid: chex.Array
    step: int  # current step
    reward: int  # current reward


GRID_SIZE = 10
VIEW_SCOPE = 2
MAX_STEPS = 50


class WarehouseManagement(MultiAgentEnv):
    def __init__(
        self,
        num_agents=2,
        **kwargs,
    ):
        # Nombre d'agents dans l'environnement.
        self.num_agents = num_agents

        # Définition des agents sous forme de noms, par exemple "agent_0", "agent_1", ...
        self.agents = [f"agent_{i}" for i in range(num_agents)]

        # Définition des espaces d'actions pour chaque agent.
        # On utilise ici la classe Discrete, et le nombre d'actions est la taille de l'énumération Actions.
        self.action_spaces = {agent: Discrete(
            len(Actions)) for agent in self.agents}

        # Définition des espaces d'observations.
        # Ici, nous supposons que l'observation de chaque agent est une grille de taille (GRID_SIZE, GRID_SIZE)
        # avec des valeurs allant de 0 à 14 (correspondant aux différents types d'items).
        self.observation_spaces = {
            agent: Box(low=0, high=14, shape=(VIEW_SCOPE * 2 +
                       1, VIEW_SCOPE * 2 + 1), dtype=jnp.int32)
            for agent in self.agents
        }

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict):
        """
        Effectue une transition d'état de l'environnement en appliquant les actions de tous les agents.

        Les actions disponibles sont :
        - 0: noop
        - 1 à 4: mouvement (up, down, left, right)
        - 5: pick (prendre un objet)
        - 6: drop (déposer un objet)

        Cette fonction décompose la mise à jour en plusieurs sous-fonctions :
        - update_positions_and_status
        - apply_pick_drop
        - update_craft
        - check_done
        - compute_reward
        """

        # Conversion du dictionnaire d'actions en tableau ordonné selon l'ordre des agents.
        # Cela permet d'avoir un tableau où l'élément d'indice i correspond à l'action de l'agent self.agents[i].
        actions_array = jnp.array([actions[agent]
                                  for agent in self.agents], dtype=jnp.int32)

        # --- 1. Mise à jour des positions et statuts pour les actions de déplacement ---
        def update_positions_and_status(state):
            # Définition d'une fonction locale qui met à jour l'état d'un agent d'indice i.
            def update_agent(i, state):
                # Récupération de l'action de l'agent i depuis le tableau actions_array.
                action = actions_array[i]
                # Récupération de la position actuelle de l'agent i dans l'état.
                pos = state.agent_positions[i]

                # On va mettre à jour la position uniquement pour les actions de mouvement (actions 1 à 4).
                def move_fn(_):
                    # Calcul de dx : si l'action est 'up', dx = -1, si 'down', dx = 1, sinon 0.
                    dx = jnp.where(action == Actions.up.value, -1,
                                   jnp.where(action == Actions.down.value, 1, 0))
                    # Calcul de dy : si l'action est 'left', dy = -1, si 'right', dy = 1, sinon 0.
                    dy = jnp.where(action == Actions.left.value, -1,
                                   jnp.where(action == Actions.right.value, 1, 0))
                    # Calcul de la position candidate en ajoutant le vecteur de déplacement à la position actuelle.
                    candidate = pos + jnp.array([dx, dy], dtype=jnp.int32)
                    # Contrainte de la position candidate pour qu'elle reste dans les bornes de la grille.
                    candidate = jnp.clip(candidate, a_min=jnp.array([0, 0]),
                                         a_max=jnp.array(state.grid.shape) - 1)
                    # Vérification que la cellule candidate est vide (c'est-à-dire que la valeur dans la grille est Items.empty).
                    valid_move = state.grid[candidate[0],
                                            candidate[1]] == Items.empty.value
                    # Utilisation de jax.lax.cond pour choisir la nouvelle position :
                    # si le mouvement est valide, la nouvelle position est candidate, sinon reste à la position actuelle.
                    new_pos = jax.lax.cond(valid_move,
                                           lambda _: candidate,
                                           lambda _: pos,
                                           operand=None)
                    return new_pos

                # Appliquer le déplacement uniquement si l'action est une action de mouvement (entre up et right).
                # Sinon, la position reste inchangée.
                new_pos = jax.lax.cond(jnp.squeeze((action == Actions.up.value) | (action == Actions.left.value)
                                       | (action == Actions.down.value) | (action == Actions.right.value)),
                                       move_fn,
                                       lambda _: pos,
                                       operand=None)
                # Mettre à jour le tableau des positions des agents en remplaçant la position de l'agent i par new_pos.
                new_agent_positions = state.agent_positions.at[i].set(new_pos)
                # Mettre à jour la grille :
                # On remet à EMPTY la cellule occupée précédemment par l'agent.
                grid_updated = state.grid.at[pos[0], pos[1]].set(
                    Items.empty.value)
                # Puis, on place l'agent dans la nouvelle cellule en marquant cette position avec Items.agent_without_object.
                grid_updated = grid_updated.at[new_pos[0], new_pos[1]].set(
                    Items.agent_without_object.value)
                # Retourner un nouvel état avec la position de l'agent mise à jour et la grille modifiée.
                return state.replace(agent_positions=new_agent_positions, grid=grid_updated)

            # Boucle sur tous les agents, de 0 à num_agents, en appliquant update_agent pour chaque agent.
            # La fonction jax.lax.fori_loop permet de réaliser cette itération de manière compilable.
            new_state = jax.lax.fori_loop(
                0, self.num_agents, update_agent, state)
            # Retourner l'état mis à jour après avoir traité tous les agents.
            return new_state

        # --- 2. Application des actions de pick et drop ---

        def apply_pick_drop(state):
            # Pour chaque agent, si l'action est "pick" (5) ou "drop" (6),
            # on met à jour l'état (agent_states) et la grille (grid) en conséquence.
            def update_agent(i, state):
                # Récupération de l'action de l'agent d'indice i depuis le tableau actions_array.
                action = actions_array[i]
                # Récupération de la position actuelle de l'agent i.
                pos = state.agent_positions[i]

                # --- Sous-fonction pour gérer l'action "pick" (action == 5) ---
                def pick_fn(_):
                    """
                    Pour l'agent d'indice i (capturé depuis la portée externe), vérifie d'abord
                    que son état est "agent_without_object". Puis, pour chaque direction dans MOVE_ACTIONS,
                    parcourt les transformations (TRANSFORMS) dans l'ordre de priorité.
                    Dès qu'une condition est satisfaite, l'état de l'agent et la grille sont mis à jour.
                    Retourne l'état mis à jour.
                    """
                    # On vérifie que l'agent est sans objet.
                    def no_pick(_):
                        return state

                    def pick_logic(_):
                        # Récupère la position actuelle de l'agent.
                        pos = state.agent_positions[i]

                        # Définir la fonction qui traite une direction donnée, avec un accumulateur "done".
                        def process_direction(d, carry):
                            # "carry" est un tuple (state, done) indiquant si une transformation a déjà été appliquée.
                            state, done = carry

                            # Si une transformation a déjà été appliquée, on ne fait rien (skip).
                            def skip_direction(_):
                                return (state, done)

                            # Sinon, on essaie cette direction.
                            def try_direction(_):
                                # Calcul de la position candidate en ajoutant le vecteur de déplacement correspondant.
                                candidate = pos + MOVE_ACTIONS[d]
                                # On s'assure que candidate reste dans les bornes de la grille.
                                candidate = jnp.clip(candidate, a_min=jnp.array(
                                    [0, 0]), a_max=jnp.array(state.grid.shape) - 1)
                                # Récupère la valeur dans la cellule candidate.
                                cell_val = state.grid[candidate[0],
                                                      candidate[1]]

                                # Maintenant, pour chaque transformation dans TRANSFORMS, on vérifie la condition.
                                # On parcourt les transformations à l'aide d'une boucle fori_loop sur l'indice j.
                                num_trans = TRANSFORMS.shape[0]
                                # L'accumulateur interne est un tuple (state, done_inner).

                                def process_transform(j, inner_carry):
                                    state_inner, done_inner = inner_carry
                                    # Si déjà mis à jour pour cette direction, on passe.

                                    def skip_transform(_):
                                        return (state_inner, done_inner)
                                    # Sinon, on teste la transformation j.

                                    def try_transform(_):
                                        # Récupère le triplet de transformation.
                                        transform_row = jnp.take(
                                            TRANSFORMS, j, axis=0)
                                        init_cell_state = transform_row[0]
                                        new_agent_state = transform_row[1]
                                        final_cell_state = transform_row[2]

                                        # Vérifie si la valeur de la cellule correspond à init_cell_state.
                                        cond = cell_val == init_cell_state
                                        # Si la condition est vraie, on met à jour l'état de l'agent et la grille.

                                        def do_update(_):
                                            updated_state = state_inner.replace(
                                                agent_states=state_inner.agent_states.at[i].set(
                                                    new_agent_state),
                                                grid=state_inner.grid.at[candidate[0], candidate[1]].set(
                                                    final_cell_state)
                                            )
                                            return (updated_state, True)
                                        # Sinon, on ne modifie rien.
                                        return jax.lax.cond(cond, do_update, lambda _: (state_inner, False), operand=None)
                                    return jax.lax.cond(done_inner, skip_transform, try_transform, operand=None)
                                # Initialisation de l'accumulateur interne : aucun update n'a encore été réalisé.
                                init_inner = (state, False)
                                state_after, done_inner = jax.lax.fori_loop(
                                    0, num_trans, process_transform, init_inner)
                                # Retourne le nouvel état et un flag indiquant si un update a été effectué pour cette direction.
                                return (state_after, done_inner)

                            # Utilise jax.lax.cond pour vérifier si, pour cette direction, un update a déjà été effectué.
                            return jax.lax.cond(done, skip_direction, try_direction, operand=None)

                        # Nombre de directions disponibles (par exemple, 4 pour haut, bas, gauche, droite).
                        num_dirs = MOVE_ACTIONS.shape[0]
                        # Initialisation de l'accumulateur externe : pas de mise à jour effectuée pour l'instant.
                        init_outer = (state, False)
                        # Itère sur toutes les directions avec une boucle fori_loop.
                        final_state, final_done = jax.lax.fori_loop(
                            0, num_dirs, process_direction, init_outer)
                        return final_state

                    # On applique la logique de pick uniquement si l'agent est sans objet.
                    return jax.lax.cond(
                        state.agent_states[i] == Items.agent_without_object.value,
                        pick_logic,
                        no_pick,
                        operand=None
                    )

                # --- Sous-fonction pour gérer l'action "drop" (action == 6) ---
                def drop_fn(_):
                    """
                    Pour l'agent d'indice i, vérifie que son statut est soit 'agent_with_primary' soit 'agent_with_secondary'.
                    Puis, parcourt toutes les directions (MOVE_ACTIONS) et tente d'appliquer une transformation selon TRANSFORMS.
                    Si une transformation est applicable, la cellule est mise à jour et l'agent passe à 'agent_without_object'.
                    """
                    # Vérification préalable du statut de l'agent.
                    def no_drop(_):
                        return state

                    # On procède uniquement si le statut est agent_with_primary ou agent_with_secondary.
                    def drop_logic(_):
                        pos = state.agent_positions[i]
                        dirs = MOVE_ACTIONS
                        num_trans = TRANSFORMS.shape[0]
                        num_dirs = dirs.shape[0]
                        total = num_trans * num_dirs

                        def body_fn(k, carry):
                            state, done = carry

                            def skip_fn(_):
                                return (state, done)

                            def try_fn(_):
                                transformation_index = k // num_dirs
                                direction_index = k % num_dirs

                                # Récupérer le triplet de transformation à partir du tableau TRANSFORMS.
                                transform_row = jnp.take(
                                    TRANSFORMS, transformation_index, axis=0)
                                expected_cell = transform_row[0]
                                new_agent_state = transform_row[1]
                                deposit_value = transform_row[2]

                                direction = dirs[direction_index]
                                candidate = pos + direction
                                candidate = jnp.clip(candidate, a_min=jnp.array([0, 0]),
                                                     a_max=jnp.array(state.grid.shape) - 1)
                                cell_val = state.grid[candidate[0],
                                                      candidate[1]]

                                # La condition : la cellule candidate doit avoir la valeur attendue.
                                cond = (cell_val == deposit_value)

                                def do_update(_):
                                    updated_state = state.replace(
                                        agent_states=state.agent_states.at[i].set(
                                            Items.agent_without_object.value),
                                        grid=state.grid.at[candidate[0], candidate[1]].set(
                                            expected_cell)
                                    )
                                    return (updated_state, True)

                                def no_update(_):
                                    return (state, False)
                                return jax.lax.cond(cond, do_update, no_update, operand=None)
                            return jax.lax.cond(done, skip_fn, try_fn, operand=None)

                        init = (state, False)
                        final_state, final_done = jax.lax.fori_loop(
                            0, total, body_fn, init)
                        return final_state

                    # On applique la condition sur le statut de l'agent.
                    return jax.lax.cond(
                        (state.agent_states[i] == Items.agent_with_primary.value) |
                        (state.agent_states[i] ==
                         Items.agent_with_secondary.value),
                        drop_logic,
                        no_drop,
                        operand=None
                    )

                # --- Application de la logique "pick" ou "drop" selon l'action de l'agent ---
                # Si l'action est "pick" (5), appliquer pick_fn, sinon ne rien changer.
                state = jax.lax.cond(
                    action == Actions.pick.value,
                    pick_fn,
                    lambda _: state,
                    operand=None
                )
                # Si l'action est "drop" (6), appliquer drop_fn, sinon ne rien changer.
                state = jax.lax.cond(
                    action == Actions.drop.value,
                    drop_fn,
                    lambda _: state,
                    operand=None
                )
                # Retourner l'état modifié après traitement de l'action de l'agent i.
                return state

            # Itérer sur tous les agents (de 0 à num_agents) en appliquant update_agent pour chacun,
            # via jax.lax.fori_loop pour conserver la compatibilité JAX et la compilation avec jit.
            new_state = jax.lax.fori_loop(
                0, self.num_agents, update_agent, state)
            # Retourner l'état final après avoir appliqué les actions pick et drop pour tous les agents.
            return new_state

        # --- 3. Mise à jour de la grille pour le craft ---
        def update_craft(state):
            """
            Met à jour la grille pour effectuer le craft d'un objet secondaire lorsque
            toutes les zones d'input craft sont remplies.

            Logique :
            1) Si la grille contient au moins une cellule de type EMPTY_INPUT_CRAFT,
                cela signifie qu'il reste un input craft vide. Dans ce cas, on ne fait rien.
            2) Sinon, on parcourt la grille et on remplace chaque cellule de type
                INPUT_CRAFT_WITH_OBJECT par EMPTY_INPUT_CRAFT.
            3) Ensuite, on cherche la première cellule (par ordre ligne-major) qui est de type
                EMPTY_OUTPUT_CRAFT et on la met à jour pour qu'elle devienne OUTPUT_CRAFT_WITH_OBJECT.
                (On ne met à jour qu'une seule cellule.)
            """
            # Étape 1 : Vérifier si la grille contient au moins une cellule EMPTY_INPUT_CRAFT.
            empty_input_craft_exists = jnp.any(
                state.grid == Items.empty_input_craft.value)

            # Si c'est le cas, on retourne l'état inchangé.
            def do_nothing(_):
                return state

            # Sinon, on effectue les mises à jour.
            def do_update(_):
                # Étape 2 : Remplacer toutes les cellules INPUT_CRAFT_WITH_OBJECT par EMPTY_INPUT_CRAFT.
                grid1 = jnp.where(state.grid == Items.input_craft_with_object.value,
                                  Items.empty_input_craft.value,
                                  state.grid)

                # Étape 3 : Chercher la première cellule EMPTY_OUTPUT_CRAFT et la mettre à jour.
                # Créer un masque booléen pour les cellules de type EMPTY_OUTPUT_CRAFT.
                mask = grid1 == Items.empty_output_craft.value
                # Vérifier s'il existe au moins une cellule avec cette valeur.
                any_output = jnp.any(mask)

                # Si aucune cellule n'est trouvée, on ne change rien.
                def update_output(_):
                    # Aplatir le masque pour obtenir un vecteur et obtenir l'indice du premier True.
                    flat_idx = jnp.argmax(mask.flatten())
                    # Convertir l'indice plat en indices 2D (ligne, colonne).
                    row = flat_idx // grid1.shape[1]
                    col = flat_idx % grid1.shape[1]
                    # Mettre à jour la cellule correspondante pour qu'elle devienne OUTPUT_CRAFT_WITH_OBJECT.
                    grid2 = grid1.at[row, col].set(
                        Items.output_craft_with_object.value)
                    return grid2

                # Appliquer la mise à jour sur la grille uniquement si au moins une cellule correspond.
                grid2 = jax.lax.cond(
                    any_output, update_output, lambda _: grid1, operand=None)
                # Retourner le nouvel état en remplaçant la grille.
                return state.replace(grid=grid2)

            # Utiliser jax.lax.cond pour choisir entre ne rien faire et effectuer la mise à jour.
            new_state = jax.lax.cond(
                empty_input_craft_exists, do_nothing, do_update, operand=None)
            return new_state

        # --- 4. Vérification de la terminaison ("done") ---
        def check_done(state: State) -> bool:
            """
            Vérifie si l'environnement a atteint un état terminal.
            - L'épisode est terminé si l'on a atteint `MAX_STEPS`.
            - OU si toutes les cases `empty_output` ont été remplacées.
            """

            # Vérifie si le nombre d'étapes a dépassé la limite
            reached_max_steps = state.step >= MAX_STEPS

            # Vérifie si toutes les cases EMPTY_OUTPUT sont remplacées
            no_empty_outputs_left = jnp.all(
                state.grid != Items.empty_output.value)

            # L'épisode se termine si l'une des conditions est remplie
            return jnp.logical_or(reached_max_steps, no_empty_outputs_left)


        # --- 5. Calcul de la récompense ---
        def compute_reward(state: State, previous_state: State) -> Dict[str, float]:
            """
            Calcule la récompense totale en fonction de l'état courant (state.grid) et de l'état précédent (previous_state).

            La logique est la suivante :
            - Pour chaque cellule de la grille, on ajoute ou soustrait une valeur selon plusieurs conditions.
            - Si la cellule est de type EMPTY_OUTPUT, on soustrait 1.
            - Si la cellule est de type INPUT_CRAFT_WITH_OBJECT et que dans l'état précédent la cellule était EMPTY_INPUT_CRAFT, on ajoute 15.
            - Si la cellule est de type OUTPUT_CRAFT_WITH_OBJECT et que dans l'état précédent la cellule était EMPTY_OUTPUT_CRAFT, on ajoute 30.
            - Si la cellule est de type AGENT_WITH_SECONDARY et que dans l'état précédent la cellule était AGENT_WITHOUT_OBJECT, on ajoute 50.
            - Si la cellule est de type OUTPUT_WITH_OBJECT et que dans l'état précédent la cellule était EMPTY_OUTPUT, on ajoute 100.
            - Les autres conditions ajoutant 0 sont omises.

            Le total est ensuite attribué de manière identique à chaque agent dans un dictionnaire.
            """

            # Pour chaque cellule, si elle vaut INPUT_CRAFT_WITH_OBJECT et dans l'état précédent la cellule était EMPTY_INPUT_CRAFT, on ajoute 15.
            reward_input_craft = jnp.where(
                (state.grid == Items.input_craft_with_object.value) &
                (previous_state.grid == Items.empty_input_craft.value),
                15, 0)

            # Pour chaque cellule, si elle vaut OUTPUT_CRAFT_WITH_OBJECT et dans l'état précédent la cellule était EMPTY_OUTPUT_CRAFT, on ajoute 30.
            reward_output_craft = jnp.where(
                (state.grid == Items.output_craft_with_object.value) &
                (previous_state.grid == Items.empty_output_craft.value),
                30, 0)

            # Pour chaque cellule, si elle vaut AGENT_WITH_SECONDARY et dans l'état précédent la cellule était AGENT_WITHOUT_OBJECT, on ajoute 50.
            reward_agent_secondary = jnp.where(
                (state.grid == Items.agent_with_secondary.value) &
                (previous_state.grid == Items.agent_without_object.value),
                50, 0)

            # Pour chaque cellule, si elle vaut OUTPUT_WITH_OBJECT et dans l'état précédent la cellule était EMPTY_OUTPUT, on ajoute 100.
            reward_output = jnp.where(
                (state.grid == Items.output_with_object.value) &
                (previous_state.grid == Items.empty_output.value),
                100, 0)

            # On somme toutes les contributions sur l'ensemble de la grille.
            total_reward = previous_state.reward + jnp.sum(reward_input_craft + reward_output_craft +
                                                           reward_agent_secondary + reward_output) - 1

            return total_reward

        # Appliquer les sous-fonctions en séquence :
        state_after_move = update_positions_and_status(state)
        state_after_pick_drop = apply_pick_drop(state_after_move)
        state_after_craft = update_craft(state_after_pick_drop)

        # Incrémenter le compteur d'étapes
        new_state = state_after_craft.replace(step=state_after_craft.step + 1)

        # Vérifier la terminaison
        done_flag = check_done(new_state)

        # Calculer la récompense
        reward = compute_reward(new_state, state)

        # Mettre à jour la récompense
        new_state = new_state.replace(reward=reward)

        # Construire les dictionnaires de récompenses et de terminaisons pour chaque agent
        rewards = {agent: reward for agent in self.agents}
        dones = {agent: done_flag for agent in self.agents}
        dones["__all__"] = done_flag
        infos = {}  # Informations supplémentaires (vide ici)

        # Extraire les observations depuis le nouvel état
        obs = self.get_obs(new_state)

        return obs, new_state, rewards, dones, infos

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Performs resetting of the environment."""

        # Fonction locale pour initialiser la grille
        def _initialize_grid(key: chex.PRNGKey) -> chex.Array:
            # Créer une grille remplie de la valeur "empty"
            grid = jnp.full((GRID_SIZE, GRID_SIZE),
                            Items.empty.value, dtype=jnp.int32)
            # Définir des obstacles
            grid = grid.at[4, 4].set(Items.wall.value)
            grid = grid.at[5, 4].set(Items.wall.value)
            # Définir des zones d'entrée
            grid = grid.at[2, 0].set(Items.empty_input.value)
            grid = grid.at[3, 0].set(Items.input_with_object.value)
            grid = grid.at[4, 0].set(Items.input_with_object.value)
            grid = grid.at[5, 0].set(Items.input_with_object.value)
            grid = grid.at[6, 0].set(Items.input_with_object.value)
            grid = grid.at[7, 0].set(Items.empty_input.value)
            # Définir des zones de craft
            grid = grid.at[3, 4].set(Items.empty_input_craft.value)
            grid = grid.at[6, 4].set(Items.empty_input_craft.value)
            grid = grid.at[4, 5].set(Items.empty_output_craft.value)
            grid = grid.at[5, 5].set(Items.empty_output_craft.value)
            # Définir des zones de sortie
            grid = grid.at[4, 9].set(Items.empty_output.value)
            grid = grid.at[5, 9].set(Items.empty_output.value)
            # Rembourrer la grille avec des valeurs "wall" (ou une autre valeur adaptée) autour
            # pour gérer les bords lors des observations.
            grid = jnp.pad(
                grid,
                pad_width=((VIEW_SCOPE, VIEW_SCOPE), (VIEW_SCOPE, VIEW_SCOPE)),
                constant_values=Items.wall.value
            )
            return grid

        # Fonction locale pour initialiser les agents
        def _initialize_agents() -> Tuple[chex.Array, chex.Array]:
            # Positionner les agents à des positions fixes dans la grille originale,
            # ici (i+1, 1) pour i allant de 0 à num_agents-1.
            # Puis, comme la grille a été paddée de VIEW_SCOPE sur chaque côté,
            # on ajoute VIEW_SCOPE à chaque coordonnée pour obtenir les positions correctes dans la grille paddée.
            agent_positions = jnp.array(
                [[i + 1 + VIEW_SCOPE, 1 + VIEW_SCOPE]
                    for i in range(self.num_agents)],
                dtype=jnp.int32
            )
            # Tous les agents commencent sans objet (état "agent_without_object")
            agent_statuses = jnp.full(
                (self.num_agents,),
                Items.agent_without_object.value,
                dtype=jnp.int32
            )
            return agent_positions, agent_statuses

        def _update_grid_with_agents(grid: chex.Array, agent_positions: chex.Array, num_agents: int) -> chex.Array:
            def body_fn(i, grid):
                pos = agent_positions[i]
                # Met à jour la grille à la position pos avec Items.agent_without_object.value
                return grid.at[pos[0], pos[1]].set(Items.agent_without_object.value)
            return jax.lax.fori_loop(0, num_agents, body_fn, grid)

        # Séparation de la clé PRNG pour usage dans l'initialisation de la grille
        sub_key, key = jax.random.split(key)

        grid = _initialize_grid(sub_key)
        agent_positions, agent_statuses = _initialize_agents()

        # Mise à jour de la grille pour marquer la position des agents.
        grid = _update_grid_with_agents(grid, agent_positions, self.num_agents)

        state = State(
            agent_positions=agent_positions,
            agent_states=agent_statuses,
            grid=grid,
            step=0,
            reward=0
        )
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """
        Retourne un dictionnaire d'observations pour chaque agent.

        Pour chaque agent, l'observation est une tranche de la grille (déjà paddée) de taille 
        (2 * VIEW_SCOPE + 1, 2 * VIEW_SCOPE + 1), centrée sur la position de l'agent.
        On suppose que state.grid contient déjà le padding nécessaire.
        """
        window_size = VIEW_SCOPE * 2 + 1

        def obs_for_agent(pos):
            # pos correspond à la position de l'agent dans la grille paddée.
            # Pour obtenir la tranche centrée sur l'agent, on soustrait VIEW_SCOPE :
            start = pos - VIEW_SCOPE
            # Extraction dynamique de la tranche d'observation
            obs = jax.lax.dynamic_slice(
                state.grid, start, (window_size, window_size))
            return obs  # Pas d'ajout de dimension, observation en 2D

        # Application vectorisée pour tous les agents.
        obs_array = jax.vmap(obs_for_agent)(state.agent_positions)

        # Création du dictionnaire d'observations avec les noms des agents.
        return {self.agents[i]: obs_array[i] for i in range(self.num_agents)}

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: State) -> Dict[str, chex.Array]:
        """Returns the available actions for each agent."""
        return {agent: [action.value for action in Actions] for agent in self.agents}


if __name__ == '__main__':
    # Création d'une clé PRNG initiale
    key = jax.random.PRNGKey(0)

    # Création de l'environnement avec 2 agents et un nombre maximal d'étapes défini par MAX_STEPS
    env = WarehouseManagement(num_agents=2)

    # Réinitialisation de l'environnement
    obs, state = env.reset(key)

    # # Création d'un dictionnaire d'actions pour le pas courant
    # # Ici, chaque agent effectue l'action "right" (4)
    actions = {agent: Actions.noop.value for agent in env.agents}

    # Effectuer un pas dans l'environnement
    for i in range(0, 50):
        obs, state, rewards, dones, infos = env.step(
            key, state, actions)
        print(i, " => \n", rewards, " || ", dones)
