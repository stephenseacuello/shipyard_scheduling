"""Genetic Algorithm scheduler for shipyard block scheduling.

Implements an evolutionary optimization approach for finding good
schedules. Used as a baseline for comparing against RL approaches.

Features:
- Permutation-based chromosome representation
- Order crossover (OX) for schedule preservation
- Swap and reassignment mutations
- Tournament selection
- Elitism for best solution preservation
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable, TYPE_CHECKING
import random
from copy import deepcopy

if TYPE_CHECKING:
    from simulation.environment import ShipyardEnv


@dataclass
class Chromosome:
    """Represents a scheduling solution as a chromosome.

    The chromosome encodes:
    - Block processing order (permutation)
    - Equipment assignments (SPMT and crane for each block)
    """
    block_order: np.ndarray  # Permutation of block indices
    spmt_assignments: np.ndarray  # SPMT index for each block
    crane_assignments: np.ndarray  # Crane index for each block
    fitness: float = 0.0


@dataclass
class GAConfig:
    """Configuration for the genetic algorithm."""
    population_size: int = 100
    generations: int = 200
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    tournament_size: int = 5
    elitism_count: int = 2
    fitness_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.fitness_weights is None:
            self.fitness_weights = {
                "throughput": 1.0,
                "tardiness": -0.5,
                "makespan": -0.1,
                "utilization": 0.2,
            }


class GAScheduler:
    """Genetic Algorithm scheduler for shipyard optimization.

    Uses evolutionary optimization to find good block processing
    schedules. The solution is encoded as a permutation of blocks
    with equipment assignments.

    Args:
        config: GA configuration parameters.
        n_blocks: Number of blocks to schedule.
        n_spmts: Number of available SPMTs.
        n_cranes: Number of available cranes.
    """

    def __init__(
        self,
        config: GAConfig,
        n_blocks: int,
        n_spmts: int,
        n_cranes: int,
    ):
        self.config = config
        self.n_blocks = n_blocks
        self.n_spmts = n_spmts
        self.n_cranes = n_cranes

        self.population: List[Chromosome] = []
        self.best_solution: Optional[Chromosome] = None
        self.fitness_history: List[float] = []

    def initialize_population(self) -> None:
        """Create initial random population."""
        self.population = []

        for _ in range(self.config.population_size):
            chromosome = Chromosome(
                block_order=np.random.permutation(self.n_blocks),
                spmt_assignments=np.random.randint(0, self.n_spmts, self.n_blocks),
                crane_assignments=np.random.randint(0, self.n_cranes, self.n_blocks),
            )
            self.population.append(chromosome)

    def evaluate_fitness(
        self,
        chromosome: Chromosome,
        env: "ShipyardEnv",
    ) -> float:
        """Evaluate fitness of a chromosome by simulation.

        Args:
            chromosome: The solution to evaluate.
            env: Environment for simulation.

        Returns:
            Fitness score (higher is better).
        """
        # Reset environment
        env.reset()

        # Execute schedule according to chromosome
        fitness_components = {
            "throughput": 0.0,
            "tardiness": 0.0,
            "makespan": 0.0,
            "utilization": 0.0,
        }

        # Simulate the schedule
        for step, block_idx in enumerate(chromosome.block_order):
            spmt_idx = chromosome.spmt_assignments[block_idx]
            crane_idx = chromosome.crane_assignments[block_idx]

            # Create action based on current state
            trans_reqs = getattr(env, "transport_requests", [])
            n_trans = len(trans_reqs)

            if n_trans == 0:
                # No transport requests — hold
                action = {
                    "action_type": 3,
                    "spmt_idx": 0,
                    "request_idx": 0,
                    "crane_idx": 0,
                    "lift_idx": 0,
                    "erection_idx": 0,
                    "equipment_idx": 0,
                }
            else:
                action = {
                    "action_type": 0,
                    "spmt_idx": int(spmt_idx) % self.n_spmts,
                    "request_idx": int(block_idx) % n_trans,
                    "crane_idx": int(crane_idx) % self.n_cranes,
                    "lift_idx": 0,
                    "erection_idx": 0,
                    "equipment_idx": 0,
                }

            try:
                _, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            except Exception:
                # Invalid action, skip
                continue

        # Compute fitness components
        fitness_components["throughput"] = env.metrics.get("blocks_completed", 0) / max(env.sim_time, 1)
        fitness_components["tardiness"] = -env.metrics.get("total_tardiness", 0)
        fitness_components["makespan"] = -env.sim_time / 1000  # Normalize
        fitness_components["utilization"] = env.metrics.get("spmt_busy_time", 0) / (
            self.n_spmts * max(env.sim_time, 1)
        )

        # Weighted sum
        fitness = sum(
            self.config.fitness_weights.get(key, 0) * value
            for key, value in fitness_components.items()
        )

        chromosome.fitness = fitness
        return fitness

    def tournament_selection(self) -> Chromosome:
        """Select a chromosome using tournament selection."""
        tournament = random.sample(self.population, self.config.tournament_size)
        return max(tournament, key=lambda c: c.fitness)

    def order_crossover(
        self,
        parent1: Chromosome,
        parent2: Chromosome,
    ) -> Tuple[Chromosome, Chromosome]:
        """Order crossover (OX) for permutation chromosomes.

        Preserves relative ordering of elements from both parents.
        """
        size = self.n_blocks

        # Select crossover points
        cx1, cx2 = sorted(random.sample(range(size), 2))

        # Create offspring order arrays
        def do_ox(p1_order: np.ndarray, p2_order: np.ndarray) -> np.ndarray:
            child = np.full(size, -1)
            # Copy segment from parent 1
            child[cx1:cx2] = p1_order[cx1:cx2]
            # Fill remaining from parent 2
            p2_idx = cx2
            child_idx = cx2
            while -1 in child:
                if p2_order[p2_idx % size] not in child:
                    child[child_idx % size] = p2_order[p2_idx % size]
                    child_idx += 1
                p2_idx += 1
            return child

        child1_order = do_ox(parent1.block_order, parent2.block_order)
        child2_order = do_ox(parent2.block_order, parent1.block_order)

        # Crossover equipment assignments (uniform crossover)
        mask = np.random.random(size) < 0.5

        child1_spmt = np.where(mask, parent1.spmt_assignments, parent2.spmt_assignments)
        child2_spmt = np.where(mask, parent2.spmt_assignments, parent1.spmt_assignments)

        child1_crane = np.where(mask, parent1.crane_assignments, parent2.crane_assignments)
        child2_crane = np.where(mask, parent2.crane_assignments, parent1.crane_assignments)

        child1 = Chromosome(child1_order, child1_spmt, child1_crane)
        child2 = Chromosome(child2_order, child2_spmt, child2_crane)

        return child1, child2

    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """Apply mutation operators to a chromosome."""
        mutated = Chromosome(
            block_order=chromosome.block_order.copy(),
            spmt_assignments=chromosome.spmt_assignments.copy(),
            crane_assignments=chromosome.crane_assignments.copy(),
        )

        # Swap mutation for block order
        if random.random() < self.config.mutation_rate:
            i, j = random.sample(range(self.n_blocks), 2)
            mutated.block_order[i], mutated.block_order[j] = (
                mutated.block_order[j],
                mutated.block_order[i],
            )

        # Equipment reassignment mutation
        for i in range(self.n_blocks):
            if random.random() < self.config.mutation_rate:
                mutated.spmt_assignments[i] = random.randint(0, self.n_spmts - 1)
            if random.random() < self.config.mutation_rate:
                mutated.crane_assignments[i] = random.randint(0, self.n_cranes - 1)

        return mutated

    def evolve_generation(self, env: "ShipyardEnv") -> None:
        """Evolve population for one generation."""
        # Evaluate fitness
        for chromosome in self.population:
            if chromosome.fitness == 0.0:
                self.evaluate_fitness(chromosome, deepcopy(env))

        # Sort by fitness
        self.population.sort(key=lambda c: c.fitness, reverse=True)

        # Update best solution
        if self.best_solution is None or self.population[0].fitness > self.best_solution.fitness:
            self.best_solution = deepcopy(self.population[0])

        self.fitness_history.append(self.population[0].fitness)

        # Create new population
        new_population = []

        # Elitism: keep best individuals
        for i in range(self.config.elitism_count):
            new_population.append(deepcopy(self.population[i]))

        # Generate offspring
        while len(new_population) < self.config.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()

            if random.random() < self.config.crossover_rate:
                child1, child2 = self.order_crossover(parent1, parent2)
            else:
                child1 = deepcopy(parent1)
                child2 = deepcopy(parent2)

            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            new_population.append(child1)
            if len(new_population) < self.config.population_size:
                new_population.append(child2)

        self.population = new_population

    def solve(
        self,
        env: "ShipyardEnv",
        verbose: bool = True,
    ) -> Chromosome:
        """Run the genetic algorithm optimization.

        Args:
            env: The shipyard environment.
            verbose: Print progress messages.

        Returns:
            Best chromosome found.
        """
        self.initialize_population()

        for gen in range(self.config.generations):
            self.evolve_generation(env)

            if verbose and (gen + 1) % 10 == 0:
                print(f"Generation {gen + 1}/{self.config.generations}: "
                      f"Best fitness = {self.best_solution.fitness:.4f}")

        return self.best_solution

    def decode_schedule(
        self,
        chromosome: Chromosome,
    ) -> List[Dict]:
        """Convert chromosome to a sequence of scheduling decisions.

        Args:
            chromosome: The solution chromosome.

        Returns:
            List of action dictionaries.
        """
        schedule = []

        for block_idx in chromosome.block_order:
            action = {
                "block_idx": int(block_idx),
                "spmt_idx": int(chromosome.spmt_assignments[block_idx]),
                "crane_idx": int(chromosome.crane_assignments[block_idx]),
            }
            schedule.append(action)

        return schedule

    def get_statistics(self) -> Dict[str, float]:
        """Get optimization statistics."""
        return {
            "best_fitness": self.best_solution.fitness if self.best_solution else 0.0,
            "generations_run": len(self.fitness_history),
            "population_size": self.config.population_size,
            "final_avg_fitness": np.mean([c.fitness for c in self.population]),
            "final_std_fitness": np.std([c.fitness for c in self.population]),
        }


def run_ga_baseline(
    env: "ShipyardEnv",
    config: Optional[GAConfig] = None,
    verbose: bool = True,
) -> Dict:
    """Run GA optimization as a baseline.

    Args:
        env: The shipyard environment.
        config: GA configuration.
        verbose: Print progress.

    Returns:
        Dictionary with results.
    """
    if config is None:
        config = GAConfig()

    # Get environment dimensions
    n_blocks = len(env.entities.get("blocks", []))
    n_spmts = len(env.entities.get("spmts", []))
    n_cranes = len(env.entities.get("cranes", []))

    scheduler = GAScheduler(
        config=config,
        n_blocks=n_blocks,
        n_spmts=n_spmts,
        n_cranes=n_cranes,
    )

    best = scheduler.solve(env, verbose=verbose)
    schedule = scheduler.decode_schedule(best)
    stats = scheduler.get_statistics()

    return {
        "best_fitness": best.fitness,
        "schedule": schedule,
        "fitness_history": scheduler.fitness_history,
        **stats,
    }
