#!/bin/bash

# Variables
dir="./exp_results/mappo_mlp_simple_world_comm"
keep1="basic-variant-state-2024-07-16_14-55-49.json"
keep2="experiment_state-2024-07-16_14-55-49.json"
keep3="MAPPOTrainer_mpe_simple_world_comm_b9929_00000_0_2024-07-16_14-55-49"

# Trouver et supprimer tous les fichiers sauf ceux spécifiés
find "$dir" -maxdepth 1 -type f ! -name "$keep1" ! -name "$keep2" ! -name "$keep3" -exec rm -f {} +

# Trouver et supprimer tous les répertoires sauf ceux spécifiés
find "$dir" -maxdepth 1 -mindepth 1 -type d ! -name "$keep3" -exec rm -rf {} +
