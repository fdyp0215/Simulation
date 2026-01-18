import random
import numpy as np
from copy import deepcopy


class BaseAgent:
    def __init__(self, landscape, start_mode='random'):
        self.landscape = landscape
        self.n = landscape.n
        self.state = random.randint(0, 2 ** self.n - 1) if start_mode == 'random' else 0
        self.fitness = self.landscape.get_fitness(self.state)

    def search_experiential(self, constrained_bits=0):
        """
        Backward-looking search: Flip 1 bit.
        """
        # For simplicity in inheritance, if constrained_bits > 0,
        # we assume the constraint applies to the Agent's CURRENT cognitive indices.
        # But for the base generic logic, we just use random bits.

        # Note: In GavettiAgent, we override this or manage the mask properly.
        # Here we keep the simple logic:

        bit_to_flip = random.randint(0, self.n - 1)

        # If this is a GavettiAgent with constraints, we retry if we hit a locked bit
        # (This is a simplified way to handle it without passing masks everywhere)
        if constrained_bits > 0 and hasattr(self, 'cog_indices'):
            # Locked bits are the ones in cog_indices
            # If constrained, we can only flip bits NOT in cog_indices
            free_indices = [i for i in range(self.n) if i not in self.cog_indices]
            if free_indices:
                bit_to_flip = random.choice(free_indices)
            else:
                return  # No moves possible

        candidate_state = self.state ^ (1 << bit_to_flip)
        candidate_fitness = self.landscape.get_fitness(candidate_state)

        if candidate_fitness > self.fitness:
            self.state = candidate_state
            self.fitness = candidate_fitness


class GavettiAgent(BaseAgent):
    """
    Replicates the agent from the 2000 paper.
    Now supports DYNAMIC cognitive maps (Fig 9-10).
    """

    def __init__(self, landscape, n1):
        super().__init__(landscape)
        self.n1 = n1
        # Initially, focus on the first N1 bits (default behavior)
        # But this can change over time in Fig 9-10.
        self.cog_indices = list(range(n1))

    def _get_avg_fitness_for_template(self, template_val):
        """
        Helper: Calculates average fitness of a cognitive template.
        template_val: The integer value of the N1 bits.
        """
        # This is computationally intensive but necessary for dynamic N1 masks.
        # Since N=10, 2^10=1024, iterating 2^(N-N1) = 128 states is fast.

        total_fit = 0.0
        count = 0

        # Identify fixed bits (cognitive) and free bits
        # We iterate all 2^(N-N1) possibilities for the free bits
        free_indices = [i for i in range(self.n) if i not in self.cog_indices]
        num_free = len(free_indices)

        for i in range(2 ** num_free):
            # Construct the full state
            temp_state = 0

            # 1. Set Cognitive Bits
            # Extract j-th bit from template_val and place it at cog_indices[j]
            for j, bit_idx in enumerate(self.cog_indices):
                bit_val = (template_val >> ((self.n1 - 1) - j)) & 1
                if bit_val:
                    temp_state |= (1 << bit_idx)

            # 2. Set Free Bits based on loop iterator 'i'
            for j, bit_idx in enumerate(free_indices):
                bit_val = (i >> j) & 1  # Simple mapping
                if bit_val:
                    temp_state |= (1 << bit_idx)

            total_fit += self.landscape.get_fitness(temp_state)
            count += 1

        return total_fit / count

    def search_cognitive(self):
        """
        Forward-looking search: Scan the current N1 map, find best template, apply it.
        """
        best_template = -1
        max_val = -1.0

        # Iterate all 2^N1 possibilities for the CURRENT cognitive dimensions
        for template in range(2 ** self.n1):
            val = self._get_avg_fitness_for_template(template)
            if val > max_val:
                max_val = val
                best_template = template

        # Apply the best template to the current state
        # We only change the bits in self.cog_indices
        new_state = self.state
        for j, bit_idx in enumerate(self.cog_indices):
            # Get the desired bit from the template
            desired_bit = (best_template >> ((self.n1 - 1) - j)) & 1

            # Clear the bit at bit_idx
            new_state &= ~(1 << bit_idx)
            # Set it if desired is 1
            if desired_bit:
                new_state |= (1 << bit_idx)

        self.state = new_state
        self.fitness = self.landscape.get_fitness(self.state)

    def reset_to_initial_cognition(self):
        """Reset cognitive dimensions to initial state (first n1 dimensions)"""
        self.cog_indices = list(range(self.n1))
        return self.cog_indices

    def shift_representation(self, mode, population=None, full_n=10, n1=3, period=None):
        """
        Improved implementation based on paper specifications for Fig 9-10

        Args:
            mode: 'fixed', 'random', or 'semi_intelligent'
            population: List of all agents (for semi-intelligent mode)
            full_n: Total number of dimensions (N)
            n1: Number of cognitive dimensions (N1)
            period: Current period (for change frequency)
        """
        if mode == 'fixed':
            return False  # No change for fixed representation

        if mode == 'random':
            # PAPER ACCURATE: Random change with 5% probability per period
            if random.random() < 0.05:  # 5% probability as in paper
                # Ensure we have exactly n1 cognitive dimensions
                if len(self.cog_indices) != n1:
                    # Reset if wrong size
                    self.cog_indices = list(range(n1))

                # Replace one random cognitive dimension with a random non-cognitive dimension
                non_cognitive = [d for d in range(full_n) if d not in self.cog_indices]

                if non_cognitive:  # Safety check
                    # Remove one cognitive dimension
                    dim_to_remove = random.choice(self.cog_indices)
                    self.cog_indices.remove(dim_to_remove)

                    # Add one non-cognitive dimension
                    dim_to_add = random.choice(non_cognitive)
                    self.cog_indices.append(dim_to_add)

                    # Sort for consistency
                    self.cog_indices.sort()

                    # PAPER ACCURATE: After changing representation, perform cognitive search
                    self.search_cognitive()
                    return True

            return False

        if mode == 'semi_intelligent':
            # PAPER ACCURATE: Change when fitness < 75% of max fitness
            if population and len(population) > 1:  # Need at least 2 agents for comparison
                # Find max fitness in population
                max_fitness = max(a.fitness for a in population)

                # PAPER SPECIFIC: Threshold is 75% of max fitness
                threshold = 0.75 * max_fitness

                # Check if this agent's performance is below threshold
                if self.fitness < threshold:
                    # PAPER SPECIFIC: Imitate from top third of population
                    sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
                    top_third_index = max(1, len(sorted_pop) // 3)  # Top 1/3
                    leaders = sorted_pop[:top_third_index]

                    if leaders:
                        # Randomly select a leader to imitate
                        leader = random.choice(leaders)

                        # Ensure leader has cog_indices attribute
                        if hasattr(leader, 'cog_indices') and leader.cog_indices:
                            # Copy the leader's cognitive dimensions
                            self.cog_indices = leader.cog_indices.copy()

                            # PAPER ACCURATE: After changing representation, perform cognitive search
                            self.search_cognitive()
                            return True

            return False

        return False


class RLAgent(BaseAgent):
    """
    Agentic Extension: Reinforcement Learning (Q-Learning).
    """

    def __init__(self, landscape, config):
        super().__init__(landscape)
        self.config = config
        self.n = landscape.n
        self.q_table = np.zeros((2 ** self.n, self.n + 1))
        self.lr = config['learning_rate']
        self.gamma = config['discount_factor']
        self.epsilon = config['epsilon']

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        self.q_table[state][action] += self.lr * (td_target - self.q_table[state][action])

    def search_rl(self):
        current_state = self.state
        action = self.select_action(current_state)
        if action < self.n:
            next_state = self.state ^ (1 << action)
        else:
            next_state = self.state

        next_fitness = self.landscape.get_fitness(next_state)
        reward = next_fitness - self.fitness
        self.update_q_table(current_state, action, reward, next_state)

        self.state = next_state
        self.fitness = next_fitness
