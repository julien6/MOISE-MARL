<h1 align="center">MOISE+MARL</h1>


[**Installation**](#install) | [**Quick Start**](#start) | [**Environments**](#environments) | [**Algorithms**](#algorithms) | [**Citation**](#cite)


<div class="collage">
    <div class="column" align="centre">
        <div class="row" align="centre">
            <img src="https://github.com/julien6/MOISE-MARL/blob/main/JaxMARL/docs/imgs/cramped_room.gif?raw=true" alt="Overcooked" width="20%">
            <img src="https://github.com/julien6/MOISE-MARL/blob/main/JaxMARL/docs/imgs/storm.gif?raw=true" alt="STORM" width="20%">
            <img src="https://github.com/julien6/MOISE-MARL/blob/main/JaxMARL/docs/imgs/warehouse_management.gif?raw=true" alt="warehouse_management" width="20%">
        </div>
        <div class="row" align="centre">
            <img src="https://github.com/julien6/MOISE-MARL/blob/main/JaxMARL/docs/imgs/coin_game.png?raw=true" alt="coin_game" width="20%">
            <img src="https://github.com/julien6/MOISE-MARL/blob/main/JaxMARL/docs/imgs/qmix_MPE_simple_tag_v3.gif?raw=true" alt="MPE" width="20%">
            <img src="https://github.com/julien6/MOISE-MARL/blob/main/JaxMARL/docs/imgs/jaxnav-ma.gif?raw=true" alt="jaxnav" width="20%">
            <img src="https://github.com/julien6/MOISE-MARL/blob/main/JaxMARL/docs/imgs/smax.gif?raw=true" alt="SMAX" width="20%">
        </div>
    </div>
</div>

## Multi-Agent Reinforcement Learning with MOISE+MARL

🎉 **Update: MOISE+MARLwas accepted at AAMAS 2025 on "Learning and Adaptation" Track. See you in Detroit!**




<h2 name="environments" id="environments">Environments 🌍 </h2>

| Environment | Reference | README | Summary |
| --- | --- | --- | --- |
| 🔴 MPE | [Paper](https://arxiv.org/abs/1706.02275) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/mpe) | Communication orientated tasks in a multi-agent particle world
| 🍲 Overcooked | [Paper](https://arxiv.org/abs/1910.05789) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/overcooked) | Fully-cooperative human-AI coordination tasks based on the homonyms video game | 
| 🎆 Warehouse Management | Novel | [Source](https://github.com/julien6/OMARLE) | Fully-cooperative partially-observable multiplayer management game |
| 👾 SMAX | Novel | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/smax) | Simplified cooperative StarCraft micro-management environment |
| 🧮 STORM: Spatial-Temporal Representations of Matrix Games | [Paper](https://openreview.net/forum?id=54F8woU8vhq) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/storm) | Matrix games represented as grid world scenarios
| 🧭 JaxNav | [Paper](https://www.arxiv.org/abs/2408.15099) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/jaxnav) | 2D geometric navigation for differential drive robots
| 🪙 Coin Game | [Paper](https://arxiv.org/abs/1802.09640) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/coin_game) | Two-player grid world environment which emulates social dilemmas
| 💡 Switch Riddle | [Paper](https://proceedings.neurips.cc/paper_files/paper/2016/hash/c7635bfd99248a2cdef8249ef7bfbef4-Abstract.html) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/switch_riddle) | Simple cooperative communication game included for debugging

 
<h2 name="algorithms" id="algorithms">Baseline Algorithms 🦉 </h2>

We follow CleanRL's philosophy of providing single file implementations which can be found within the `baselines` directory. We use Hydra to manage our config files, with specifics explained in each algorithm's README. Most files include `wandb` logging code, this is disabled by default but can be enabled within the file's config.

| Algorithm | Reference | README | 
| --- | --- | --- | 
| IPPO | [Paper](https://arxiv.org/pdf/2011.09533.pdf) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/IPPO) | 
| MAPPO | [Paper](https://arxiv.org/abs/2103.01955) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/MAPPO) | 
| IQL | [Paper](https://arxiv.org/abs/1312.5602v1) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/QLearning) | 
| VDN | [Paper](https://arxiv.org/abs/1706.05296)  | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/QLearning) |
| QMIX | [Paper](https://arxiv.org/abs/1803.11485) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/QLearning) |
| TransfQMIX | [Paper](https://www.southampton.ac.uk/~eg/AAMAS2023/pdfs/p1679.pdf) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/QLearning) |
| SHAQ | [Paper](https://arxiv.org/abs/2105.15013) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/QLearning) |
| PQN-VDN | [Paper](https://arxiv.org/abs/2407.04811) | [Source](https://github.com/mttga/purejaxql) |

<h2 name="install" id="install">Installation 🧗 </h2>

**Environments** - Before installing, ensure you have the correct [JAX installation](https://github.com/google/jax#installation) for your hardware accelerator. We have tested up to JAX version 0.4.25. The MOISE+MARLenvironments can be installed directly from PyPi:

```
pip install jaxmarl 
```

**Algorithms** - If you would like to also run the algorithms, install the source code as follows:

1. Clone the repository:
    ```
    git clone https://github.com/FLAIROx/JaxMARL.git && cd JaxMARL
    ```
2. Install requirements:
    ``` 
    pip install -e .[algs]
    export PYTHONPATH=./JaxMARL:$PYTHONPATH
    ```
3. For the fastest start, we reccoment using our Dockerfile, the usage of which is outlined below.

**Development** - If you would like to run our test suite, install the additonal dependencies with:
 `pip install -e .[dev]`, after cloning the repository.

<h2 name="start" id="start">Quick Start 🚀 </h2>

We take inspiration from the [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) and [Gymnax](https://github.com/RobertTLange/gymnax) interfaces. You can try out training an agent in our [Colab notebook](https://colab.research.google.com/github/FLAIROx/JaxMARL/blob/main/jaxmarl/tutorials/JaxMARL_Walkthrough.ipynb). Further introduction scripts can be found [here](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/tutorials).

### Basic MOISE+MARLAPI  Usage 🖥️

Actions, observations, rewards and done values are passed as dictionaries keyed by agent name, allowing for differing action and observation spaces. The done dictionary contains an additional `"__all__"` key, specifying whether the episode has ended. We follow a parallel structure, with each agent passing an action at each timestep. For asynchronous games, such as Hanabi, a dummy action is passed for agents not acting at a given timestep.

```python 
import jax
from jaxmarl import make

key = jax.random.PRNGKey(0)
key, key_reset, key_act, key_step = jax.random.split(key, 4)

# Initialise environment.
env = make('MPE_simple_world_comm_v3')

# Reset the environment.
obs, state = env.reset(key_reset)

# Sample random actions.
key_act = jax.random.split(key_act, env.num_agents)
actions = {agent: env.action_space(agent).sample(key_act[i]) for i, agent in enumerate(env.agents)}

# Perform the step transition.
obs, state, reward, done, infos = env.step(key_step, state, actions)
```

### Dockerfile 🐋
To help get experiments up and running we include a [Dockerfile](https://github.com/FLAIROx/JaxMARL/blob/main/Dockerfile) and its corresponding [Makefile](https://github.com/FLAIROx/JaxMARL/blob/main/Makefile). With Docker and the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) installed, the container can be built with:
```
make build
```
The built container can then be run:
```
make run
```

## Contributing 🔨
Please contribute! Please take a look at our [contributing guide](https://github.com/FLAIROx/JaxMARL/blob/main/CONTRIBUTING.md) for how to add an environment/algorithm or submit a bug report. Our roadmap also lives there.

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

#### Role base related Projects:
 - TODO

#### Organizational inference related Projects:
 - TODO

<!-- - [Mava](https://github.com/instadeepai/Mava): JAX implementations of IPPO and MAPPO, two popular MARL algorithms.
- [PureJaxRL](https://github.com/luchris429/purejaxrl): JAX implementation of PPO, and demonstration of end-to-end JAX-based RL training.
- [Minimax](https://github.com/facebookresearch/minimax/): JAX implementations of autocurricula baselines for RL.
- [JaxIRL](https://github.com/FLAIROx/jaxirl?tab=readme-ov-file): JAX implementation of algorithms for inverse reinforcement learning.

JAX-native environments:
- [Gymnax](https://github.com/RobertTLange/gymnax): Implementations of classic RL tasks including classic control, bsuite and MinAtar.
- [Jumanji](https://github.com/instadeepai/jumanji): A diverse set of environments ranging from simple games to NP-hard combinatorial problems.
- [Pgx](https://github.com/sotetsuk/pgx): JAX implementations of classic board games, such as Chess, Go and Shogi.
- [Brax](https://github.com/google/brax): A fully differentiable physics engine written in JAX, features continuous control tasks.
- [XLand-MiniGrid](https://github.com/corl-team/xland-minigrid): Meta-RL gridworld environments inspired by XLand and MiniGrid.
- [Craftax](https://github.com/MichaelTMatthews/Craftax): (Crafter + NetHack) in JAX. -->