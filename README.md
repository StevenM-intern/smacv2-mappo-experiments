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

### Windows

```bash
baseline_5v5_10M.bat
medivac_5v5_10M.bat
```

### Linux / Mac

```bash
./run_experiment.sh
```

---

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

---

## Goal

Evaluate how unit composition impacts learning stability and performance in multi-agent reinforcement learning.

---

## Acknowledgements

This project is built on top of an open-source MAPPO implementation.

Original implementation by Dmitri Manajev (MIT License).

Modifications, experiments, and analysis by Steven Marsh.
