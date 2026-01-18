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
                # Fig 10: Joint search (cognitive + experiential) with representation change
                # 创建三个相同的初始群体
                pop_fixed = [GavettiAgent(lscape, self.n1) for _ in range(self.pop_size)]
                pop_random = deepcopy(pop_fixed)
                pop_smart = deepcopy(pop_fixed)

                # 所有群体都执行初始认知搜索（时期0）
                for a in pop_fixed: a.search_cognitive()
                for a in pop_random: a.search_cognitive()
                for a in pop_smart: a.search_cognitive()

                # 固定组：重置为初始认知维度（确保固定不变）
                for a in pop_fixed:
                    a.reset_to_initial_cognition()

                # 跟踪变化次数（用于调试）
                random_change_counts = []
                smart_change_counts = []

                for p in range(self.periods):
                    # 记录当前时期的适应度
                    results['Fixed'][p] += np.mean([a.fitness for a in pop_fixed])
                    results['Random_Change'][p] += np.mean([a.fitness for a in pop_random])
                    results['Semi_Intelligent'][p] += np.mean([a.fitness for a in pop_smart])

                    # PAPER ACCURATE: 经验搜索（约束在非认知维度）
                    # 对于Fig 10，经验搜索只针对非认知维度（N-N1个维度）
                    for a in pop_fixed:
                        a.search_experiential(constrained_bits=self.n1)
                    for a in pop_random:
                        a.search_experiential(constrained_bits=self.n1)
                    for a in pop_smart:
                        a.search_experiential(constrained_bits=self.n1)

                    # PAPER ACCURATE: 表征变化逻辑
                    # 固定组：什么都不做（保持固定表征）

                    # 随机变化组：每个周期有5%概率改变
                    random_changes_this_period = 0
                    for a in pop_random:
                        changed = a.shift_representation(
                            mode='random',
                            population=pop_random,  # 需要population参数，虽然随机变化不需要
                            full_n=self.n,
                            n1=self.n1,
                            period=p
                        )
                        if changed:
                            random_changes_this_period += 1
                    random_change_counts.append(random_changes_this_period)

                    smart_changes_this_period = 0

                    if pop_smart:
                        max_fitness = max(a.fitness for a in pop_smart)
                        threshold = 0.75 * max_fitness  # 论文中的阈值：最大适应度的75%

                        for a in pop_smart:
                            if a.fitness < threshold:
                                changed = a.shift_representation(
                                    mode='semi_intelligent',
                                    population=pop_smart,
                                    full_n=self.n,
                                    n1=self.n1,
                                    period=p
                                )
                                if changed:
                                    smart_changes_this_period += 1

                    smart_change_counts.append(smart_changes_this_period)

                    # 可选：每10期打印一次调试信息
                    if p % 10 == 0 and t == 0:  # 只在第一次试验时打印
                        avg_fixed = np.mean([a.fitness for a in pop_fixed])
                        avg_random = np.mean([a.fitness for a in pop_random])
                        avg_smart = np.mean([a.fitness for a in pop_smart])
                        print(f"Period {p}: Fixed={avg_fixed:.3f}, Random={avg_random:.3f}, Smart={avg_smart:.3f}")
                        print(
                            f"  Changes: Random={random_changes_this_period}/{self.pop_size}, Smart={smart_changes_this_period}/{self.pop_size}")

                # 可选：记录变化统计
                if t == 0:  # 只在第一次试验时记录
                    print(f"Trial {t}: Total random changes = {sum(random_change_counts)}")
                    print(f"Trial {t}: Total smart changes = {sum(smart_change_counts)}")

        # Average over trials
        for k in results:
            results[k] /= trials

        return results
