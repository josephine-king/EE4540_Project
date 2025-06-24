# EE4540 Final Project: Distributed Signal Processing

This repository contains the final project for EE4540: Distributed Signal Processing at TU Delft. The project explores distributed consensus algorithms, focusing on:

- **Randomized Gossip Algorithms**
- **Primal-Dual Method of Multipliers (PDMM)**

Experiments and simulations are implemented in Python and demonstrated through Jupyter notebooks and plots.

## Project Structure

- `distributed_project.ipynb` — Main notebook with experiments and plots  
- `pdmm.py` — PDMM algorithm implementation (average and median consensus) 
- `randomized_gossip.py` — Gossip algorithm implementation  
- `utils.py` — Utility functions for simulations  
- `dist_env.yml` — Conda environment file (optional)  
- `figures/` — Output plots from experiments  
- `README.md`

## Features

- Simulation of average/median consensus over sensor networks
- Comparison of broadcast vs unicast communication models
- Performance metrics for PDMM vs Randomized Gossip methods
- Generation of plots for convergence behavior

## Running the Project

Launch the Jupyter notebook:

```bash
jupyter notebook distributed_project.ipynb
```

Make sure all `.py` files are in the same directory as the notebook for imports to work correctly.

## Results

All result plots are stored in the `figures/` directory, including:

- Convergence plots for gossip and PDMM
- Effect of communication models on consensus speed
- Network topology visualization
