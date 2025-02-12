import os
import jax
import pygame
import numpy as np
import imageio

from jaxmarl.environments.warehouse_management import WarehouseManagement, GRID_SIZE, VIEW_SCOPE

# Dimensions de la grille et des cellules
CELL_SIZE = 50  # Taille en pixels d'une cellule
WINDOW_SIZE = ((2 * VIEW_SCOPE + GRID_SIZE) * CELL_SIZE,
               (2 * VIEW_SCOPE + GRID_SIZE) * CELL_SIZE)

# Couleurs et images associées aux types de cellules
COLORS = {
    0: (255, 255, 255),  # EMPTY: blanc
    1: (128, 128, 128),  # OBSTACLE: gris
    2: None,  # AGENT_WITHOUT_OBJECT: image agent
    3: None,  # AGENT_WITH_PRIMARY: agent avec objet primaire
    4: None,  # AGENT_WITH_SECONDARY: agent avec objet secondaire
    5: None,  # PRIMARY_OBJECT: objet primaire
    6: None,  # SECONDARY_OBJECT: objet secondaire
    7: (173, 216, 230),  # EMPTY_INPUT: bleu clair
    8: (173, 216, 230),  # INPUT_WITH_OBJECT: image entrée
    9: (255, 200, 128),  # EMPTY_INPUT_CRAFT: orange clair
    10: (255, 200, 128),  # INPUT_CRAFT_WITH_OBJECT
    11: (255, 165, 0),  # EMPTY_OUTPUT_CRAFT: orange foncé
    12: (255, 165, 0),  # OUTPUT_CRAFT_WITH_OBJECT
    13: (255, 182, 193),  # EMPTY_OUTPUT: rouge clair
    14: (255, 182, 193),  # OUTPUT_WITH_OBJECT
}

IMAGES = {
    2: "assets/agent.png",
    3: "assets/agent_primary_object.png",
    4: "assets/agent_secondary_object.png",
    5: "assets/primary_object.png",
    6: "assets/secondary_object.png",
    7: "assets/input.png",
    8: "assets/primary_object.png",
    9: "assets/input.png",
    10: "assets/primary_object.png",
    11: "assets/output.png",
    12: "assets/secondary_object.png",
    13: "assets/output.png",
    14: "assets/secondary_object.png",
}


class WarehouseManagementVisualizer:
    def __init__(self):
        pygame.display.init()
        pygame.display.set_mode((1, 1))

        # ✅ Surface en mémoire (pas de fenêtre)
        self.screen = pygame.Surface(WINDOW_SIZE)

        # Effacer l'écran
        self.screen.fill((255, 255, 255))

        # Charger les images en mémoire
        self.loaded_images = {
            k: pygame.image.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), v)).convert_alpha() if v else None for k, v in IMAGES.items() if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), v))
        }

    def render(self, state):
        """
        Convertit un état en une image (tableau de pixels).
        """
        state = jax.device_get(state)
        grid = np.array(state.grid)

        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                cell_value = grid[row, col]
                cell_rect = pygame.Rect(
                    col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE
                )

                # Dessiner la couleur de la cellule
                # Blanc par défaut
                color = COLORS.get(cell_value, (255, 255, 255))
                if color:
                    pygame.draw.rect(self.screen, color, cell_rect)

                # Dessiner une image si elle existe
                image = self.loaded_images.get(cell_value)
                if image:
                    image = pygame.transform.scale(
                        image, (CELL_SIZE, CELL_SIZE))
                    self.screen.blit(
                        image, (col * CELL_SIZE, row * CELL_SIZE))

        # Dessiner les agents
        for agent_index in range(0, state.agent_positions.shape[0]):
            row, col = state.agent_positions[agent_index]
            agent_image = self.loaded_images.get(
                state.agent_states[agent_index], self.loaded_images.get(2))  # Default to agent image
            if agent_image:
                agent_image = pygame.transform.scale(
                    agent_image, (CELL_SIZE, CELL_SIZE))
                self.screen.blit(
                    agent_image, (col * CELL_SIZE, row * CELL_SIZE))

        # Retourne l'image sous forme de tableau numpy (RGB)
        return pygame.surfarray.array3d(self.screen)

    def animate_ep(self, state_seq, filename="animation.gif"):
        """
        Génère un GIF animé à partir d'une séquence d'états.
        """
        frames = [self.render(state) for state in state_seq]
        imageio.mimsave(filename, frames, fps=10, loop=0)
        print(f"🎥 Animation sauvegardée : {filename}")

    def animate_eps(self, state_seqs, filename="animation.gif"):
        """
        Génère un GIF en mosaïque avec plusieurs épisodes.
        """

        # episode_frames = [[self.render(state)
        #                    for state in ep] for ep in state_seqs]

        episode_frames = [[np.rot90(np.flip(self.render(state), axis=1), k=1)
                           for state in ep] for ep in state_seqs]

        # Déterminer la disposition de la mosaïque
        num_episodes = len(state_seqs)
        # Carré presque parfait
        grid_size = int(np.ceil(np.sqrt(num_episodes)))

        # Construire chaque frame du GIF mosaïque
        frames = []
        # Assumer que tous les épisodes ont la même longueur
        for frame_idx in range(len(episode_frames[0])):
            frame_rows = []
            for i in range(grid_size):
                row_frames = []
                for j in range(grid_size):
                    idx = i * grid_size + j
                    if idx < num_episodes:
                        row_frames.append(episode_frames[idx][frame_idx])
                    else:
                        # Ajouter un espace blanc si on dépasse le nombre d'épisodes
                        row_frames.append(np.ones_like(
                            episode_frames[0][0]) * 255)
                frame_rows.append(np.concatenate(row_frames, axis=1))
            mosaic_frame = np.concatenate(frame_rows, axis=0)
            frames.append(mosaic_frame)

        # Sauvegarde du GIF en mosaïque
        imageio.mimsave(filename, frames, fps=10, loop=0)
        print(f"🎥 Animation mosaïque sauvegardée : {filename}")
