<h1 align="center">MOISE+MARL</h1>

---
<div class="collage">
    <div class="column" align="centre">
        <div class="row" align="centre">
            <img src="https://github.com/julien6/MOISE-MARL/blob/main/JaxMARL/docs/imgs/cramped_room.gif?raw=true" alt="Overcooked" width="25%">
            <img src="https://github.com/julien6/MOISE-MARL/blob/main/JaxMARL/docs/imgs/storm.gif?raw=true" alt="STORM" width="25%">
            <img src="https://github.com/julien6/MOISE-MARL/blob/main/JaxMARL/docs/imgs/warehouse_management.gif?raw=true" alt="warehouse_management" width="25%">
        </div>
        <div class="row" align="centre">
            <img src="https://github.com/julien6/MOISE-MARL/blob/main/JaxMARL/docs/imgs/qmix_MPE_simple_tag_v3.gif?raw=true" alt="MPE" width="25%">
            <img src="https://github.com/julien6/MOISE-MARL/blob/main/JaxMARL/docs/imgs/jaxnav-ma.gif?raw=true" alt="jaxnav" width="25%">
            <img src="https://github.com/julien6/MOISE-MARL/blob/main/JaxMARL/docs/imgs/smax.gif?raw=true" alt="SMAX" width="25%">
        </div>
    </div>
</div>

## Multi-Agent Reinforcement Learning with MOISE+MARL

🎉 **Update: MOISE+MARL was accepted at AAMAS 2025 on "Learning and Adaptation" Track. See you in Detroit!**

This repository contains an implementation of the MOISE+MARL framework compatible with [MARLlib](https://marllib.readthedocs.io/en/latest/) algorithms and [PettingZoo](https://pettingzoo.farama.org/) environments.

The development of the MOISE+MARL framework version compatible with [Jax](https://github.com/jax-ml/jax) and related libraries is currently going on for environments and algorithms based on [JaxMARL](https://github.com/FLAIROx/JaxMARL).


<h2 name="environments" id="environments">Environments 🌍 </h2>

| Environment | Reference | README | Summary |
| --- | --- | --- | --- |
| 🔴 MPE | [Paper](https://arxiv.org/abs/1706.02275) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/mpe) | Communication orientated tasks in a multi-agent particle world
| 🍲 Overcooked | [Paper](https://arxiv.org/abs/1910.05789) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/overcooked) | Fully-cooperative human-AI coordination tasks based on the homonyms video game | 
| 🎆 Warehouse Management | Novel | [Source](https://github.com/julien6/OMARLE) | Fully-cooperative partially-observable multiplayer management game |
| 👾 SMAX | Novel | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/smax) | Simplified cooperative StarCraft micro-management environment |
| 🧮 STORM: Spatial-Temporal Representations of Matrix Games | [Paper](https://openreview.net/forum?id=54F8woU8vhq) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/storm) | Matrix games represented as grid world scenarios
| 🧭 JaxNav | [Paper](https://www.arxiv.org/abs/2408.15099) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/jaxnav) | 2D geometric navigation for differential drive robots
| 🪙 3rd CAGE Challenge | [Paper](https://arxiv.org/abs/2108.09118) | [Source](https://github.com/cage-challenge/cage-challenge-3) | Cyberdefenders fighting against malware programs in a drom swarm scenario
 
<h2 name="algorithms" id="algorithms">Baseline Algorithms 🦉 </h2>

We used [CleanRL](https://docs.cleanrl.dev/) implementation for MARL algorithms trying to have a single file content for each one.

Jax algorithms implementation also follow the CleanRL philosophy in accordance to JaxMARL's ones.

| Algorithm | Reference | README | 
| --- | --- | --- | 
| MAPPO | [Paper](https://arxiv.org/abs/2103.01955) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/MAPPO) | 
| COMA | [Paper](https://arxiv.org/abs/1705.08926) | [Source](https://github.com/matteokarldonati/Counterfactual-Multi-Agent-Policy-Gradients) | 
| QMIX | [Paper](https://arxiv.org/abs/1803.11485) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/QLearning) |
| MADDPG | [Paper](https://arxiv.org/abs/1706.02275) | [Source](https://github.com/openai/maddpg) | 
| IQL | [Paper](https://arxiv.org/abs/1312.5602v1) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/QLearning) | 
| VDN | [Paper](https://arxiv.org/abs/1706.05296)  | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/QLearning) |
| IPPO | [Paper](https://arxiv.org/pdf/2011.09533.pdf) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/IPPO) | 
| TransfQMIX | [Paper](https://www.southampton.ac.uk/~eg/AAMAS2023/pdfs/p1679.pdf) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/QLearning) |
| SHAQ | [Paper](https://arxiv.org/abs/2105.15013) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/QLearning) |
| PQN-VDN | [Paper](https://arxiv.org/abs/2407.04811) | [Source](https://github.com/mttga/purejaxql) |



<h2 name="install" id="install">Installation 🧗 </h2>

### 1) Install virtual Python environment

Be sure to have a proper Conda installation.
To install Miniconda (we used to conduct our experiments), please type:

```
curl -o Miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda.sh
./Miniconda.sh -b -p $HOME/miniconda
rm Miniconda.sh
export PATH="$HOME/miniconda/bin:$PATH"
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
print_step "Miniconda installé avec succès"
source ~/miniconda/etc/profile.d/conda.sh
```

Create the conda virtual environment as follows:

```
conda create -n moise-marl python=3.10
conda activate moise-marl
```

Then clone the repo as follows:
```
git clone https://github.com/julien6/MOISE-MARL.git && cd MOISE-MARL
```


### 2.1) Installation of the **MARLlib** MOISE+MARL version

To install all dependencies for both MARLlib algorithms and PettingZoo environments, run package installation as follows:

```
pip install -e .[marllib]
```

### 2.2) [Work in Progress] Installation of the **JaxMARL** MOISE+MARL version

To install all dependencies for both JaxMARL algorithms and environments, run package installation as follows:

```
pip install -e .[jaxmarl]
```

*For any Jax/JaxMARL issue, be sure to have a proper JaxMARL installation. Please refer to [JaxMARL](https://github.com/FLAIROx/JaxMARL?tab=readme-ov-file#installation--) installation guide.*

### 3) Installation checking

Check MOISE+MARL has been properly installed typing:

```
./check_installation.sh
```
This script will test all of the algorithms can be launched accross environments without raising an exception.

The installation is properly completeted if "All OK" message is displayed

### 4) Run evaluation

Demonstration training and test for an MARL algorithm can be runned on an environment, using :
```
./run <algorithm_name>_<environment_name>
```

*Note:*
 - *any non-finished training will be saved as a checkpoint*
 - *use the "--resume" option to resume from last saved checkpoint*

For instance, MAPPO can be used to train agents on the "Overcooked AI" environment as follows:
```
./run mappo_overcooked_ai
```

### Basic MOISE+MARL API Usage for MARLlib 🖥️

*MOISE+MARL API* (MMA) showcases an `mma_wrapper` environment wrapper expecting a parallel PettingZoo environment and the MOISE+MARL model `mm_model` whose organizational specifications such as roles, goals, missions are previously defined.

We conveniently gathered RLClean's algorithms through the MMA `make_train` function expecting any PettingZoo environment and a MARL algorithm configuration (possibly choosing among our fine-tuned ones in "configuration/training/marl_env/").
We used [Hydra](https://github.com/facebookresearch/hydra) to configure the checkpoint functioning: checkpoints are saved every defined step interval and resume from last one until reaching the planned max step number.
After training, this function outputs a joint-policy model and training/testing stats (as well as saving those as the last checkpoint)

MMMA also comprises the *Trajectory-based Evaluation in MOISE+MARL* (TEMM) method through the `TEMM` function expecting.
This function expects a joint-policy model and outputs another MOISE+MARL model completed with genuinely inferred organizational specifications.
This function uses unsupervised learning techniques such as Hierarchical Clustering or K-means and save their output as visualized figures and additional data in a dedicated directory to help users in characterizing emergent behaviors as organizational specifications.

```python 
from pettingzoo.mpe import simple_world_comm_v3
from moise_marl.make_train import make_train
from moise_marl.TEMM import TEMM
from moise_marl.configuration.training.marl_env import mappo_simple_world_comm

env = simple_world_comm_v3.env(render_mode="human")

# Define a MOISE+MARL model
simple_model = mm_model(structural_specifications= {}, functional_specifications = {}, deontic_specifications= {})

# Wrap the environment
env = mma_wrapper(env, simple_model)

# Launch the training with given algorithm and training hyper-parameters
policy_model, stats = make_train(env, mappo_simple_world_comm, resume_from_last_checkpoint=True)

# Run the TEMM method to infer implicit organizational specifications
inferred_model = TEMM(policy_model)

```

<h2 name="cite" id="cite">Citing MOISE+MARL📜 </h2>
If you use MOISE+MARLin your work, please cite us as follows:

```

@inproceedings{soule2024moise_marl, 
  title     = {MOISE+MARL: Enhancing Explainability and Control in Multi-Agent Reinforcement Learning Through Organizational Integration}, 
  author    = {Soulé, Julien and Jamont, Jean-Paul and Occello, Michel and Traonouez, Louis-Marie and Théron, Paul}, 
  booktitle = {Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems (AAMAS)}, 
  year      = {2024}, 
  series    = {AAMAS '24}, 
  pages     = {XXX--XXX}, % remplacer par les pages exactes
  publisher = {International Foundation for Autonomous Agents and Multiagent Systems}, 
  address   = {Auckland, New Zealand}, 
  month     = {May}, 
  abstract  = {This paper presents MOISE+MARL, a framework that enhances explainability and control in Multi-Agent Reinforcement Learning by integrating the MOISE+ organizational model. The proposed approach introduces trajectory-based evaluation methods and organizational constraints to guide agent learning while improving interpretability. Experimental results demonstrate significant improvements in policy stability, organizational fit, and task performance across multiple environments.}, 
  keywords  = {Multi-Agent Reinforcement Learning, Organizational Models, Explainability, MOISE+, Policy Evaluation, Policy Control}, 
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