# main.py
import argparse
import yaml
import matplotlib.pyplot as plt
from src.simulation import SimulationEngine


def load_config(path="config.yaml"):
    """Load simulation parameters from the YAML config file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def run_cli():
    parser = argparse.ArgumentParser(
        description="Gavetti (2000) Replication CLI with Agentic & Evolutionary Extensions"
    )
    parser.add_argument("--trials", type=int, default=None,
                        help="Override number of trials from config")
    parser.add_argument("--exp", type=str, required=True,
                        choices=['fig6', 'fig7', 'agentic', 'evolution'],
                        help=(
                            "Experiment type:\n"
                            " - fig6: Joint vs Experiential Search\n"
                            " - fig7: Constrained vs Unconstrained Cognition\n"
                            " - agentic: RL-Optimized Agents vs Standard\n"
                            " - evolution: Dynamics of Mental Model Change (Figs 9-10)"
                        ))
    args = parser.parse_args()

    config = load_config()

    # Override trials if provided in CLI
    if args.trials:
        config['simulation']['trials'] = args.trials

    print(f"--- Starting Experiment: {args.exp} ---")
    print(f"Landscape: N={config['landscape']['N']}, K={config['landscape']['K']}")

    engine = SimulationEngine(config)
    results = engine.run_batch(args.exp)

    # Visualization
    plt.figure(figsize=(10, 6))
    for label, data in results.items():
        plt.plot(data, label=label, linewidth=2)
        print(f"Final Average Fitness [{label}]: {data[-1]:.4f}")

    plt.title(f"Simulation Results: {args.exp.replace('_', ' ').title()}")
    plt.xlabel("Periods")
    plt.ylabel("Average Population Fitness")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    output_path = f"results_{args.exp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"--- Success! Plot saved to {output_path} ---")


if __name__ == "__main__":
    run_cli()