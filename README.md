# Multi-Agent Reinforcement Learning with MOISE+MARL

**Update**: We are excited to share that **MOISE+MARL** was accepted at **AAMAS 2025** in the *Learning and Adaptation* track. See you in Detroit!

**MOISE+MARL** is a MARL framework designed to integrate organizational concepts—such as roles, missions, and goals—into the learning process. By embedding these structures directly into standard MARL algorithms, MOISE+MARL enables more interpretable, efficient coordination among agents, helping them discover and follow well-defined organizational patterns while still adapting to complex or dynamic environments.

This repository provides an implementation of the **MOISE+MARL framework**, compatible with [MARLlib](https://marllib.readthedocs.io/en/latest/) algorithms and [PettingZoo](https://pettingzoo.farama.org/) environments.

A **JAX-based** version of MOISE+MARL is currently under development to support environments and algorithms from [JaxMARL](https://github.com/FLAIROx/JaxMARL).

---

## Environments 🌍

| Environment          | Reference                                                   | README                                                                             | Summary                                                                 |
|----------------------|-------------------------------------------------------------|-------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| 🔴 **MPE**           | [Paper](https://arxiv.org/abs/1706.02275)                  | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/mpe)     | Communication-oriented tasks in a multi-agent particle world           |
| 🍲 **Overcooked**    | [Paper](https://arxiv.org/abs/1910.05789)                  | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/overcooked) | Fully-cooperative human–AI coordination tasks based on the Overcooked video game |
| 🎆 **Warehouse Management** | Novel                                           | [Source](https://github.com/julien6/OMARLE)                                         | Fully-cooperative, partially-observable multiplayer management game    |
| 👾 **SMAX**          | Novel                                                      | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/smax)    | Simplified cooperative StarCraft micromanagement environment           |
| 🧮 **STORM: Spatial-Temporal Representations of Matrix Games** | [Paper](https://openreview.net/forum?id=54F8woU8vhq) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/storm)   | Matrix games represented as grid-based scenarios                        |
| 🧭 **JaxNav**        | [Paper](https://www.arxiv.org/abs/2408.15099)              | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/jaxnav)  | 2D geometric navigation for differential drive robots                  |
| 🪙 **3rd CAGE Challenge** | [Paper](https://arxiv.org/abs/2108.09118)           | [Source](https://github.com/cage-challenge/cage-challenge-3)                        | Cyberdefense tasks against malware in a drone swarm scenario           |

---

## Baseline Algorithms 🦉

We employ [CleanRL](https://docs.cleanrl.dev/) implementations of MARL algorithms, preserving CleanRL’s single-file philosophy. Our JAX-based algorithms follow the same CleanRL approach, consistent with [JaxMARL](https://github.com/FLAIROx/JaxMARL).

| Algorithm    | Reference                                                 | README                                                                                 |
|--------------|-----------------------------------------------------------|----------------------------------------------------------------------------------------|
| **MAPPO**    | [Paper](https://arxiv.org/abs/2103.01955)                | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/MAPPO)                |
| **COMA**     | [Paper](https://arxiv.org/abs/1705.08926)                | [Source](https://github.com/matteokarldonati/Counterfactual-Multi-Agent-Policy-Gradients) |
| **QMIX**     | [Paper](https://arxiv.org/abs/1803.11485)                | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/QLearning)            |
| **MADDPG**   | [Paper](https://arxiv.org/abs/1706.02275)                | [Source](https://github.com/openai/maddpg)                                            |
| **IQL**      | [Paper](https://arxiv.org/abs/1312.5602v1)               | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/QLearning)            |
| **VDN**      | [Paper](https://arxiv.org/abs/1706.05296)                | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/QLearning)            |
| **IPPO**     | [Paper](https://arxiv.org/pdf/2011.09533.pdf)            | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/IPPO)                 |
| **TransfQMIX** | [Paper](https://www.southampton.ac.uk/~eg/AAMAS2023/pdfs/p1679.pdf) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/QLearning)  |
| **SHAQ**     | [Paper](https://arxiv.org/abs/2105.15013)                | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/QLearning)            |
| **PQN-VDN**  | [Paper](https://arxiv.org/abs/2407.04811)                | [Source](https://github.com/mttga/purejaxql)                                          |

---

## Installation 🧗

### 1) Install a virtual Python environment

Make sure you have a valid Conda installation. To install Miniconda (used in our experiments), run:

```bash
curl -o Miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda.sh
./Miniconda.sh -b -p $HOME/miniconda
rm Miniconda.sh
export PATH="$HOME/miniconda/bin:$PATH"
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
echo "Miniconda installed successfully."
source ~/miniconda/etc/profile.d/conda.sh
```

Create and activate the Conda environment:

```bash
conda create -n moise-marl python=3.10
conda activate moise-marl
```

Then clone the repository:

```bash
git clone https://github.com/julien6/MOISE-MARL.git && cd MOISE-MARL
```

### 2.1) Install the **MARLlib** version of MOISE+MARL

To install dependencies for both MARLlib algorithms and PettingZoo environments:

```bash
pip install -e .[marllib]
```

### 2.2) [Work in Progress] Install the **JaxMARL** version of MOISE+MARL

To install dependencies for JaxMARL-based algorithms and environments:

```bash
pip install -e .[jaxmarl]
```

If you encounter JAX or JaxMARL-specific issues, please ensure a proper JaxMARL installation. For details, refer to the [JaxMARL documentation](https://github.com/FLAIROx/JaxMARL?tab=readme-ov-file#installation--).

### 3) Verify the Installation

Run:

```bash
./check_installation.sh
```

This script attempts to launch each algorithm across multiple environments. If it completes without errors and displays **"All OK"**, the installation is successful.

### 4) Run Evaluation

<div class="collage">

    <div class="column" align="center">
        <div class="row" align="center">
            <img src="https://github.com/julien6/MOISE-MARL/blob/main/JaxMARL/docs/imgs/cramped_room.gif?raw=true" alt="Overcooked" width="20%">
            <img src="https://github.com/julien6/MOISE-MARL/blob/main/JaxMARL/docs/imgs/storm.gif?raw=true" alt="STORM" width="20%">
            <img src="https://github.com/julien6/MOISE-MARL/blob/main/JaxMARL/docs/imgs/warehouse_management.gif?raw=true" alt="Warehouse Management" width="20%">
        </div>
        <div class="row" align="center">
            <img src="https://github.com/julien6/MOISE-MARL/blob/main/JaxMARL/docs/imgs/qmix_MPE_simple_tag_v3.gif?raw=true" alt="MPE" width="20%">
            <img src="https://github.com/julien6/MOISE-MARL/blob/main/JaxMARL/docs/imgs/jaxnav-ma.gif?raw=true" alt="JaxNav" width="20%">
            <img src="https://github.com/julien6/MOISE-MARL/blob/main/JaxMARL/docs/imgs/smax.gif?raw=true" alt="SMAX" width="20%">
        </div>
    </div>

</div>

*Examples of rendered environments as animated GIFs (adapted from [JaxMARL](https://github.com/FLAIROx/JaxMARL))*

---

To train and test a MARL algorithm on a specific environment, run:

```bash
./run <algorithm_name>_<environment_name>
```

**Notes:**

* Incomplete training sessions are automatically saved as checkpoints.
* Checkpoints contain the latest training data (learning curves) as well as test data (rendered environments).
* Use the `--resume` option to continue from the most recent checkpoint.

For example, to train agents using **MAPPO** on the **Overcooked AI** environment:

```bash
./run mappo_overcooked_ai
```

---

### Basic MOISE+MARL API Usage for MARLlib 🖥️

The **MOISE+MARL API (MMA)** provides an `mma_wrapper` environment wrapper that expects a parallel PettingZoo environment and a MOISE+MARL model ( `mm_model` ) with predefined organizational specifications (e.g., roles, goals, missions).

We have consolidated RLClean’s algorithms under MMA via the `make_train` function, which accepts any PettingZoo environment alongside a MARL algorithm configuration. Users can either supply a custom configuration or select one from our fine-tuned options in the `configuration/training/marl_env/` directory. We leverage [Hydra](https://github.com/facebookresearch/hydra) to handle checkpoints, saving them at specified intervals and resuming from the latest checkpoint until reaching the designated maximum number of training steps. Upon completion, `make_train` outputs a joint-policy model as well as training and testing statistics, all of which are recorded in the final checkpoint.

MMA also features the **Trajectory-based Evaluation in MOISE+MARL (TEMM)** method, accessed via the `TEMM` function. This function takes a joint-policy model and produces a new MOISE+MARL model augmented with inferred organizational specifications. Internally, it employs unsupervised learning techniques—such as hierarchical clustering or k-means—to identify and characterize emergent behaviors as organizational constructs. The resulting figures and accompanying data are stored in a dedicated directory, assisting users in analyzing the learned organizational structures.

```python 
from pettingzoo.mpe import simple_world_comm_v3
from moise_marl.make_train import make_train
from moise_marl. TEMM import TEMM
from moise_marl.configuration.training.marl_env import mappo_simple_world_comm

env = simple_world_comm_v3.env(render_mode="human")

# Define a MOISE+MARL model

simple_model = mm_model(structural_specifications= {}, functional_specifications = {}, deontic_specifications= {})

# Wrap the environment

env = mma_wrapper(env, simple_model)

# Launch the training with given algorithm and training hyper-parameters

policy_model, stats = make_train(env, mappo_simple_world_comm, resume_from_last_checkpoint=True)

# Run the TEMM method to infer implicit organizational specifications

inferred_model = TEMM(policy_model, output_directory="analysis_results")

```

<h2 name="cite" id="cite">Citing MOISE+MARL📜 </h2>
If you use MOISE+MARL in your work, please cite us as follows:

```
@inproceedings{soule2024moise_marl, 
  title     = {An Organizationally-Oriented Approach to Enhancing Explainability and Control in Multi-Agent Reinforcement Learning}, 
  author    = {Soulé, Julien and Jamont, Jean-Paul and Occello, Michel and Traonouez, Louis-Marie and Théron, Paul}, 
  booktitle = {Proceedings of the 24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS)}, 
  year      = {2024}, 
  series    = {AAMAS '24}, 
  pages     = {XXX--XXX}, % TBD
  publisher = {International Foundation for Autonomous Agents and Multiagent Systems}, 
  address   = {Detroit, USA}, 
  month     = {May}, 
  abstract  = {Multi-Agent Reinforcement Learning can lead to the development of collaborative agent behaviors that show similarities with organizational concepts. Pushing forward this perspective, we introduce a novel framework that explicitly incorporates organizational roles and goals from the $\mathcal{M}OISE^+$ model into the MARL process, guiding agents to satisfy corresponding organizational constraints. By structuring training with roles and goals, we aim to enhance both the explainability and control of agent behaviors at the organizational level, whereas much of the literature primarily focuses on individual agents. Additionally, our framework includes a post-training analysis method to infer implicit roles and goals, offering insights into emergent agent behaviors. This framework has been applied across various MARL environments and algorithms, demonstrating coherence between predefined organizational specifications and those inferred from trained agents.}, 
  keywords  = {Multi-Agent Reinforcement Learning, Organizational Explainability, Organizational Control}, 
}
```

## See Also 🙌

There are a number of other libraries which inspired this work, we encourage you to take a look!

#### Related Projects:

 - *ROMA*: https://github.com/TonghanWang/ROMA
 - *Roco*: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5060074
 - *CORD*: https://arxiv.org/abs/2501.02221
 - *TarMAC*: https://arxiv.org/abs/1810.11187
 - *Feudal Multi-Agent Hierarchies for Cooperative Reinforcement Learning*
: https://arxiv.org/abs/1901.08492
