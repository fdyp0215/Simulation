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

    def shift_representation(self, mode, population=None):
        """
        [cite_start]Logic for changing the cognitive map (Fig 9-10). [cite: 390-402]
        """
        if mode == 'fixed':
            return

        perform_shift = False

        if mode == 'random':
            # "The probability of a shift... specified as a fixed probability"
            # Paper doesn't strictly specify P, we assume a small probability like 0.1 per period
            # or force it every period for the "Random Change" curve if implied by context.
            # Usually simulation agents have a probability P_change. Let's use 0.2.
            if random.random() < 0.2:
                perform_shift = True
                # [cite_start]"One of the N1 dimensions is simply replaced at random" [cite: 393]
                # Pick one to drop
                drop_idx = random.choice(self.cog_indices)
                self.cog_indices.remove(drop_idx)
                # Pick one to add
                available = [i for i in range(self.n) if i not in self.cog_indices]
                add_idx = random.choice(available)
                self.cog_indices.append(add_idx)
                # Sort for consistency (optional but good for debugging)
                self.cog_indices.sort()

        elif mode == 'semi_intelligent':
            # "If relative performance falls below a fixed percentage... threshold set at 25 percent below max"
            # [cite_start]i.e., Fitness < 0.75 * Max_Fitness [cite: 402]
            if population:
                max_fit = max(a.fitness for a in population)
                threshold = 0.75 * max_fit

                if self.fitness < threshold:
                    perform_shift = True
                    # [cite_start]"Imitates the cognition of one of the leading organizations... top third" [cite: 402]
                    leaders = sorted(population, key=lambda x: x.fitness, reverse=True)
                    top_third_idx = len(population) // 3
                    if top_third_idx < 1: top_third_idx = 1
                    target_agent = random.choice(leaders[:top_third_idx])

                    # Copy the mental model (indices), not the policy
                    self.cog_indices = deepcopy(target_agent.cog_indices)

        # [cite_start]"If a new cognitive representation is adopted... distinct set of N1 policy parameters will be identified" [cite: 407]
        # This implies we run cognitive search immediately after shifting.
        if perform_shift:
            self.search_cognitive()


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