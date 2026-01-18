# Gavetti & Levinthal (2000) Replication with Agentic AI

This system replicates the core simulation from the paper *"Looking Forward and Looking Backward: Cognitive and Experiential Search"* (Administrative Science Quarterly, 2000). It demonstrates how cognitive representations (simplified mental models) guide experiential learning on rugged landscapes (NK Model).

This repository is developed in Python and requires Python version 3.9+.

## Features


1. **Classic Replication**: 
   - **Figure 6**: Demonstrates the value of "Cognitive Jumps" (Joint Search) vs. pure Experiential Search.
   - **Figure 7**: Analyzes the "disciplining" effect of cognitive constraints.
2. **Dynamics of Representation (Fig 9 & 10)**:
   - Simulates how organizations evolve their mental models over time (Fixed vs. Random Change vs. Semi-Intelligent Imitation).
3. **Agentic Extension (RL)**: 
   - Introduces a **Reinforcement Learning (Q-Learning) Agent**.
   - Compares the "Heuristic Cognition" of standard agents against the "Algorithmic Experience" of RL agents.
4. **Dual Interface**:
   - **Web UI**: Interactive dashboard built with Streamlit for real-time parameter tweaking.
   - **CLI**: Fast, scriptable command-line execution for batch experiments.

## 🏃 Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Modify Configuration:

Edit `config.yaml` to change simulation parameters:

- Landscape: `N` (Dimensions) and `K` (Complexity)
- Simulation: `pop_size`, `trials` and `seed`
- RL Agent: 

    `learning_rate`: Step size for Q-updates.

    `discount_factor`: How model values future reward in Q-learning.

    `epsilon`: Exploration rate during pre-training.

    `train_episodes`: How long the RL agent learns before the simulation starts.

### Usage

#### Option 1: Web Interface (Recommended)

Launch the dashboard to visualize experiments interactively:
```bash
streamlit run app.py
```

#### Option 2: Command Line

Run experiments directly and save plots to disk (.png):

```bash
# Run classic Joint vs Experiential search (Fig 6)
python main.py --exp fig6 --trials 50

# Run Constrained vs Unconstrained experiential search with cognition
python main.py --exp fig7 --trails 50

# Run Representation Evolution (Fig 9/10 Logic)
python main.py --exp evolution

# Run the RL Agent Extension
python main.py --exp agentic
```

## Expected Outputs
You can check simulation results by using WebUI.

By running main.py under construction, you can check simulation graphs(.png) under this file.

## Agentic Extension Details

The RL agent learns from extensive pre-training episodes without relying on explicit mental models. This extension allows us to explore the trade-offs between data-driven decision-making and heuristic-guided strategies under varying levels of environmental complexity.

---

## Author

Created by ZHANG Mufan

Contact: [mzhangeb@connect.ust.hk](mailto:mzhangeb@connect.ust.hk)
```