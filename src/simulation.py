import numpy as np
from copy import deepcopy
import random
from .agents import GavettiAgent, RLAgent
from .landscape import NKLandscape


class SimulationEngine:
    def __init__(self, config):
        self.cfg = config
        self.n = config['landscape']['N']
        self.k = config['landscape']['K']
        self.n1 = config['landscape']['N1']
        self.pop_size = config['simulation']['pop_size']
        self.periods = config['simulation']['periods']

    def train_rl_agent(self, agent, episodes=500):
        """Pre-train the RL Agent."""
        for _ in range(episodes):
            agent.state = random.randint(0, 2 ** self.n - 1)
            agent.fitness = agent.landscape.get_fitness(agent.state)
            for _ in range(20):
                agent.search_rl()

    def run_batch(self, experiment_type):
        """
        Runs multiple trials and averages results.
        Types: 'fig6', 'fig7', 'agentic', 'evolution' (Fig 9/10)
        """
        trials = self.cfg['simulation']['trials']
        results = {}

        # Initialize result holders
        if experiment_type == 'fig6':
            keys = ['Experiential', 'Joint']
        elif experiment_type == 'fig7':
            keys = ['Unconstrained', 'Constrained']
        elif experiment_type == 'agentic':
            keys = ['Standard_Joint', 'RL_Agent']
        elif experiment_type == 'evolution':
            # Fig 9/10: Three lines comparing representation strategies
            keys = ['Fixed', 'Random_Change', 'Semi_Intelligent']

        for k in keys:
            results[k] = np.zeros(self.periods)

        for t in range(trials):
            seed = self.cfg['simulation']['seed'] + t
            lscape = NKLandscape(self.n, self.k, self.n1, seed=seed)

            # --- Experiment Logic ---
            if experiment_type == 'fig6':
                pop_exp = [GavettiAgent(lscape, self.n1) for _ in range(self.pop_size)]
                pop_joint = deepcopy(pop_exp)
                for agent in pop_joint: agent.search_cognitive()

                for p in range(self.periods):
                    results['Experiential'][p] += np.mean([a.fitness for a in pop_exp])
                    results['Joint'][p] += np.mean([a.fitness for a in pop_joint])
                    for a in pop_exp: a.search_experiential(constrained_bits=0)
                    for a in pop_joint: a.search_experiential(constrained_bits=0)

            elif experiment_type == 'fig7':
                pop_base = [GavettiAgent(lscape, self.n1) for _ in range(self.pop_size)]
                for a in pop_base: a.search_cognitive()
                pop_unconst = deepcopy(pop_base)
                pop_const = deepcopy(pop_base)

                for p in range(self.periods):
                    results['Unconstrained'][p] += np.mean([a.fitness for a in pop_unconst])
                    results['Constrained'][p] += np.mean([a.fitness for a in pop_const])
                    for a in pop_unconst: a.search_experiential(constrained_bits=0)
                    for a in pop_const: a.search_experiential(constrained_bits=self.n1)

            elif experiment_type == 'agentic':
                pop_std = [GavettiAgent(lscape, self.n1) for _ in range(20)]
                pop_rl = [RLAgent(lscape, self.cfg['rl_agent']) for _ in range(20)]
                for a in pop_rl:
                    self.train_rl_agent(a, episodes=self.cfg['rl_agent']['train_episodes'])
                for a in pop_std: a.search_cognitive()

                for p in range(self.periods):
                    results['Standard_Joint'][p] += np.mean([a.fitness for a in pop_std])
                    results['RL_Agent'][p] += np.mean([a.fitness for a in pop_rl])
                    for a in pop_std: a.search_experiential()
                    for a in pop_rl: a.search_rl()

            elif experiment_type == 'evolution':
                # Run three separate sub-experiments on the same landscape for fairness

                # 1. Fixed Representation
                pop_fixed = [GavettiAgent(lscape, self.n1) for _ in range(self.pop_size)]
                for a in pop_fixed: a.search_cognitive()

                # 2. Random Change
                pop_random = deepcopy(pop_fixed)

                # 3. Semi-Intelligent Change
                pop_smart = deepcopy(pop_fixed)

                for p in range(self.periods):
                    # Record
                    results['Fixed'][p] += np.mean([a.fitness for a in pop_fixed])
                    results['Random_Change'][p] += np.mean([a.fitness for a in pop_random])
                    results['Semi_Intelligent'][p] += np.mean([a.fitness for a in pop_smart])

                    # [cite_start]Step 1: Experiential Search (All agents do this) [cite: 421]
                    # Note: Fig 9 is pure cognitive, Fig 10 is joint.
                    # Assuming we want Fig 10 logic (Joint) as it's more complete.
                    for a in pop_fixed: a.search_experiential()
                    for a in pop_random: a.search_experiential()
                    for a in pop_smart: a.search_experiential()

                    # Step 2: Shift Representations
                    # Fixed: Do nothing

                    # Random:
                    for a in pop_random:
                        a.shift_representation(mode='random')

                    # Semi-Intelligent: Needs population stats
                    # We iterate a copy or index to avoid changing pop while reading stats?
                    # shift_representation reads the pop stats.
                    # To be rigorous, we pass the current state of pop_smart.
                    current_smart_pop = list(pop_smart)  # shallow copy of list
                    for a in pop_smart:
                        a.shift_representation(mode='semi_intelligent', population=current_smart_pop)

        # Average over trials
        for k in results:
            results[k] /= trials

        return results