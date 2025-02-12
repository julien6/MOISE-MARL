import os
import jax
import pygame
import numpy as np

from jaxmarl.environments.warehouse_management import WarehouseManagement, GRID_SIZE, VIEW_SCOPE

# Dimensions de la grille et des cellules
CELL_SIZE = 50  # Taille en pixels d'une cellule
WINDOW_SIZE = ((2*VIEW_SCOPE + GRID_SIZE) * CELL_SIZE,
               (2*VIEW_SCOPE + GRID_SIZE) * CELL_SIZE)


# Couleurs et images associées aux types de cellules
COLORS = {
    0: (255, 255, 255),  # EMPTY: white
    1: (128, 128, 128),  # OBSTACLE: grey
    2: None,  # AGENT_WITHOUT_OBJECT: agent image
    3: None,  # AGENT_WITH_PRIMARY: agent with primary object image
    4: None,  # AGENT_WITH_SECONDARY: agent with secondary object image
    5: None,  # PRIMARY_OBJECT: primary object image
    6: None,  # SECONDARY_OBJECT: secondary object image
    7: (173, 216, 230),  # EMPTY_INPUT: light blue
    8: (173, 216, 230),  # INPUT_WITH_OBJECT: input object image
    9: (255, 200, 128),  # EMPTY_INPUT_CRAFT: light orange
    # INPUT_CRAFT_WITH_OBJECT: input craft image@ext:GitHub.copilot-chat
    10: (255, 200, 128),
    11: (255, 165, 0),  # EMPTY_OUTPUT_CRAFT: dark orange
    12: (255, 165, 0),  # OUTPUT_CRAFT_WITH_OBJECT: output craft image
    13: (255, 182, 193),  # EMPTY_OUTPUT: light red
    14: (255, 182, 193),  # OUTPUT_WITH_OBJECT: output object image
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

# Mapping des touches pour les actions
KEY_ACTIONS = {
    pygame.K_UP: 1,     # Move up
    pygame.K_DOWN: 2,   # Move down
    pygame.K_LEFT: 3,   # Move left
    pygame.K_RIGHT: 4,  # Move right
    pygame.K_a: 5,  # Pick up
    pygame.K_q: 6  # Drop
}


def draw_grid(screen, grid, agent_positions, agent_states):
    """
    Dessine la grille en fonction de l'état actuel.
    Args:
        screen (pygame.Surface): Surface de la fenêtre PyGame.
        grid (numpy.ndarray): La grille représentant l'environnement.
        agent_positions (dict): Positions des agents dans la grille.
    """
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            cell_value = grid[row, col]
            cell_rect = pygame.Rect(
                col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)

            # Dessiner la couleur de la cellule
            color = COLORS.get(cell_value, (255, 255, 255))  # Blanc par défaut
            if color:
                pygame.draw.rect(screen, color, cell_rect)

            # Dessiner une image si elle existe
            image_path = IMAGES.get(cell_value)
            if image_path:
                image_path = os.path.join(os.path.dirname(
                    os.path.abspath(__file__)), image_path)
                image = pygame.image.load(image_path)
                image = pygame.transform.scale(image, (CELL_SIZE, CELL_SIZE))
                screen.blit(image, (col * CELL_SIZE, row * CELL_SIZE))

    # Dessiner les agents par-dessus la grille
    for agent_index in range(0, agent_positions.shape[0]):
        agent = f"agent_{agent_index}"
        row = agent_positions[agent_index][0]
        col = agent_positions[agent_index][1]
        agent_image_path = IMAGES.get(
            agent_states[agent_index], "assets/agent.png")
        agent_image_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), agent_image_path)

        # agent_image = pygame.image.load(agent_image_path)
        agent_image = pygame.image.load(agent_image_path).convert_alpha()

        agent_image = pygame.transform.scale(
            agent_image, (CELL_SIZE, CELL_SIZE))
        screen.blit(agent_image, (col * CELL_SIZE, row * CELL_SIZE))


def run_visualization():
    """
    Lance l'environnement visuel avec PyGame.
    Permet de contrôler un agent manuellement via les flèches du clavier.
    """
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Warehouse Management Visualization")

    # Création d'une clé PRNG initiale
    key = jax.random.PRNGKey(0)

    # Charger l'environnement
    warehouse_env = WarehouseManagement(num_agents=3)
    observations, state = warehouse_env.reset(key)

    # Agent à contrôler manuellement
    manual_agent = "agent_0"
    clock = pygame.time.Clock()
    running = True

    while running:
        actions = {manual_agent: 0}

        # Gestion des événements (clavier)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Assigner une action à l'agent contrôlé manuellement
                if event.key in KEY_ACTIONS:
                    actions[manual_agent] = KEY_ACTIONS[event.key]

        # Actions automatiques pour les autres agents
        for agent in warehouse_env.agents:
            if agent != manual_agent:
                actions[agent] = warehouse_env.action_spaces[agent].sample(
                    key).item()
        # Effectuer une étape dans l'environnement
        observations, state, rewards, dones, infos = warehouse_env.step(
            key, state, actions)

        observations = jax.device_get(observations)
        state = jax.device_get(state)
        rewards = jax.device_get(rewards)
        dones = jax.device_get(dones)
        infos = jax.device_get(infos)

        if True in dones.values():
            break

        # Effacer l'écran
        screen.fill((255, 255, 255))

        # Dessiner la grille
        draw_grid(screen, state.grid,
                  state.agent_positions, state.agent_states)

        # Afficher les scores en haut de l'écran
        font = pygame.font.Font(None, 36)
        score_text = font.render(
            f"Rewards: {str({agent: val.item() for agent, val in rewards.items()})}", True, (0, 0, 0))
        screen.blit(score_text, (10, 10))

        # Mettre à jour l'écran
        pygame.display.flip()

        # Contrôle de la vitesse de simulation
        clock.tick(5)  # 5 FPS

    pygame.quit()


if __name__ == "__main__":
    run_visualization()
