import streamlit as st
import yaml
import matplotlib.pyplot as plt
from src.simulation import SimulationEngine

# Page Config
st.set_page_config(page_title="Gavetti Replication & AI Extension", layout="wide")

st.title("Looking Forward and Backward: AI Replication\n Author: Mufan (HKUST)")
st.markdown("""
Replication of **Gavetti & Levinthal (2000)** with Agentic Extensions.
Select an experiment below to run the simulation live.
""")

# --- Sidebar Configuration ---
st.sidebar.header("Global Landscape Settings")
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Landscape Params
config['landscape']['N'] = st.sidebar.slider("N (Policy Dimensions)", 5, 15, 10)
config['landscape']['K'] = st.sidebar.slider("K (Complexity/Interactions)", 0, config['landscape']['N'] - 1, 3)
config['simulation']['trials'] = st.sidebar.number_input("Number of Trials", 1, 100, 20)

# RL Parameters Section
st.sidebar.divider()
st.sidebar.header("RL Agent Parameters")
config['rl_agent']['learning_rate'] = st.sidebar.slider("Learning Rate (α)", 0.01, 0.5, 0.2)
config['rl_agent']['epsilon'] = st.sidebar.slider("Exploration (ε)", 0.0, 1.0, 0.1)
config['rl_agent']['train_episodes'] = st.sidebar.number_input("Pre-train Episodes", 100, 10000, 2000)

# Experiment Selection
tab1, tab2, tab3, tab4 = st.tabs([
    "Fig 6: Joint vs Exp",
    "Fig 7: Constraints",
    "Agentic AI Extension",
    "Evolution (Fig 9-10)"
])

engine = SimulationEngine(config)


def run_and_plot(exp_type, title, custom_config=None):
    current_config = config.copy()
    if custom_config:
        for key, value in custom_config.items():
            if key in current_config['landscape']:
                current_config['landscape'][key] = value

    temp_engine = SimulationEngine(current_config)

    with st.spinner(f"Running {exp_type} simulation trials..."):
        results = temp_engine.run_batch(exp_type)
        fig, ax = plt.subplots(figsize=(10, 5))
        for label, data in results.items():
            ax.plot(data, label=label, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Periods")
        ax.set_ylabel("Avg. Population Fitness")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Display final metrics
        cols = st.columns(len(results))
        for i, (label, data) in enumerate(results.items()):
            cols[i].metric(label, f"{data[-1]:.4f}")


with tab1:
    st.header("Figure 6: The Value of Cognition")
    st.markdown(
        "Compares pure experiential learning (hill-climbing) vs. Joint search (Cognitive 'Jump' + Experiential).")
    if st.button("Run Figure 6 Sim"):
        run_and_plot('fig6', "Effect of Initial Cognitive Jump")

with tab2:
    st.header("Figure 7: Constraints as Discipline")
    st.markdown("Does sticking to the cognitive map (Constrained) help avoid 'wandering off' the peak?")
    if st.button("Run Figure 7 Sim"):
        run_and_plot('fig7', "Constrained vs. Unconstrained Search")

with tab3:
    st.subheader("Standard Agents vs. RL-Optimized Agents")
    st.info("**What happens here?** The RL Agent pre-trains on the landscape to identify high-reward bit combinations.")
    if st.button("Run RL Extension Comparison"):
        run_and_plot('agentic', "Cognitive Search vs. Reinforcement Learning")

with tab4:
    st.header("Evolution of Representations (Fig 9-10)")
    st.markdown("""
    **Dynamics of Mental Models**: What happens when organizations change *how* they see the world?

    * **Fixed**: Mental model never changes.
    * **Random Change**: Periodically swaps a cognitive dimension randomly.
    * **Semi-Intelligent**: Imitates successful peers when performance is low.
    """)

    if st.button("Run Evolution Sim"):
        run_and_plot('evolution', "Fitness impact of Changing Representations")