import numpy as np
import random

class NKLandscape:
    """
    Represents the N-dimensional fitness landscape and the N1-dimensional
    cognitive simplification.

    Attributes:
        n (int): Total dimensions.
        k (int): Interaction count per gene.
        n1 (int): Cognitive dimensions.
        fitness_map (np.array): Pre-calculated fitness for all 2^N states.
        cognitive_map (np.array): Pre-calculated fitness for all 2^N1 cognitive states.
    """

    def __init__(self, n: int, k: int, n1: int, seed: int = None):
        self.n = n
        self.k = k
        self.n1 = n1
        if seed is not None:
            np.random.seed(seed)

        # 1. Generate Interaction Matrix (Contributions)
        # Shape: (N, 2^(K+1)). Random float [0, 1] for each interaction config.
        self.contributions = np.random.rand(n, 2 ** (k + 1))

        # 2. Pre-calculate FULL Landscape (2^N states)
        # Optimization: Flatten loop using vectorization would be faster for large N,
        # but for N=10, a straightforward generation is readable and fast enough.
        self.fitness_map = np.zeros(2 ** n)
        for i in range(2 ** n):
            self.fitness_map[i] = self._calculate_fitness_raw(i)

        # 3. Pre-calculate COGNITIVE Representation (2^N1 states)
        # Cognitive value = Average fitness of all actual states consistent with the cognitive pattern.
        self.cognitive_map = np.zeros(2 ** n1)
        for cog_state in range(2 ** n1):
            # The cognitive map 'sees' the first N1 bits.
            # We must average all 2^(N-N1) states that share this prefix.
            matching_fitnesses = []
            suffix_range = 2 ** (n - n1)

            # Construct the full integer from (prefix + suffix)
            base = cog_state << (n - n1)
            for suffix in range(suffix_range):
                full_state = base | suffix
                matching_fitnesses.append(self.fitness_map[full_state])

            self.cognitive_map[cog_state] = np.mean(matching_fitnesses)

    def _calculate_fitness_raw(self, state_int: int) -> float:
        """Internal helper to calculate fitness for one state."""
        fitness_sum = 0.0
        # Convert integer to bit array (MSB first)
        bits = [(state_int >> i) & 1 for i in range(self.n - 1, -1, -1)]

        for i in range(self.n):
            # Calculate index for contribution table based on neighbors (Circular)
            sub_state_idx = 0
            for offset in range(self.k + 1):
                neighbor_idx = (i + offset) % self.n
                bit_val = bits[neighbor_idx]
                sub_state_idx = (sub_state_idx << 1) | bit_val

            fitness_sum += self.contributions[i, sub_state_idx]

        return fitness_sum / self.n

    def get_fitness(self, state_int: int) -> float:
        """O(1) lookup for true fitness."""
        return self.fitness_map[state_int]

    def get_cognitive_fitness(self, cog_state_int: int) -> float:
        """O(1) lookup for cognitive belief."""
        return self.cognitive_map[cog_state_int]

    def get_cognitive_fitness_with_dims(self, cognitive_state, cognitive_dims, current_full_state):
        """
        Calculate cognitive fitness for given cognitive dimensions.

        Args:
            cognitive_state: Integer representing cognitive state (0 to 2^N1-1)
            cognitive_dims: List of dimension indices in cognitive representation
            current_full_state: Current full state of the agent

        Returns:
            Average fitness over all possible completions of non-cognitive dimensions
        """
        n1 = len(cognitive_dims)
        total_fitness = 0.0
        count = 0

        # Number of non-cognitive dimensions
        non_cognitive_count = self.n - n1

        # If n1 equals n, then cognitive representation is complete
        if n1 == self.n:
            # Construct full state from cognitive state
            full_state = 0
            for i, dim in enumerate(sorted(cognitive_dims)):
                bit = (cognitive_state >> i) & 1
                full_state = full_state | (bit << dim)
            return self.get_fitness(full_state)

        # Otherwise, average over all possible completions
        # Since 2^(N-N1) can be large, we sample instead of exhaustive enumeration
        num_samples = min(100, 2 ** non_cognitive_count)

        for _ in range(num_samples):
            # Start with current state
            test_state = current_full_state

            # Set cognitive dimensions according to cognitive_state
            for i, dim in enumerate(sorted(cognitive_dims)):
                bit = (cognitive_state >> i) & 1
                # Clear current bit
                test_state = test_state & ~(1 << dim)
                # Set to new value
                test_state = test_state | (bit << dim)

            # Randomize non-cognitive dimensions
            non_cognitive_dims = [d for d in range(self.n) if d not in cognitive_dims]
            for dim in non_cognitive_dims:
                # Random bit for non-cognitive dimension
                random_bit = random.randint(0, 1)
                # Clear current bit
                test_state = test_state & ~(1 << dim)
                # Set to random value
                test_state = test_state | (random_bit << dim)

            total_fitness += self.get_fitness(test_state)
            count += 1

        return total_fitness / count if count > 0 else 0.0