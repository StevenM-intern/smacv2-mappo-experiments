# SMACv2 MAPPO Experiments

This project explores Multi-Agent Reinforcement Learning using MAPPO on SMACv2.

The goal is to study how different unit compositions (baseline vs medivac-heavy) affect learning performance in cooperative environments.

---

## Setup

```bash
conda create -n marl python=3.10
conda activate marl
pip install -r requirements.txt
```

---

## Running Experiments

### Linux / Mac

```bash
./run_experiment.sh baseline
./run_experiment.sh medivac
```

### Windows

```bat
run_experiment.bat baseline
run_experiment.bat medivac
run_experiment.bat zerg
run_experiment.bat protoss
```


---
## Notes

* Ensure your conda environment name matches the one in the script (`marl-ppo-suite`), or update the script accordingly.
* SMACv2 and StarCraft II must be installed correctly before running experiments.
* If running on a different machine, verify environment dependencies using `requirements.txt`.

  ## Prerequisites

Before running this project, ensure the following are installed:

* Python 3.10
* Anaconda / Miniconda
* PyTorch
* StarCraft II
* SMACv2 environment

### Setup Instructions

```bash
conda create -n marl-ppo-suite python=3.10
conda activate marl-ppo-suite
pip install -r requirements.txt
```

### Install SMAC + StarCraft II

Follow official SMAC setup:
https://github.com/oxwhirl/smac

Make sure StarCraft II is installed and the SC2PATH environment variable is set correctly.



## Experiments

* Baseline (Terran 5v5)
* Medivac-heavy configuration
* Additional maps (10v10 planned)


---

## Metrics Collected

* Train reward
* Train win rate
* Evaluation win rate
* Episode length
* Actor loss
* Critic loss

  ## Running Experiments

### Linux / Mac

```bash
./run_experiment.sh baseline
./run_experiment.sh medivac
```

### Windows

```bat
run_experiment.bat baseline
run_experiment.bat medivac
run_experiment.bat zerg
run_experiment.bat protoss
```


---

## Goal

Evaluate how unit composition impacts learning stability and performance in multi-agent reinforcement learning.

---

## Acknowledgements

This project is built on top of an open-source MAPPO implementation.

Original implementation by Dmitri Manajev (MIT License).

Modifications, experiments, and analysis by Steven Marsh.
