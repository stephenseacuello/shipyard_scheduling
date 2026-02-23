"""Train/validation/test data splits for shipyard scheduling experiments.

This module provides utilities for creating reproducible environment configurations
for training, validation, and testing. It ensures proper separation of evaluation
scenarios while maintaining consistent seeding for reproducibility.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List


class ShipyardDataSplits:
    """Manage train/val/test environment configurations.

    This class creates different environment configurations for:
    - Training: Full domain randomization enabled
    - Validation: Fixed seed, no randomization (for tracking progress)
    - Testing: Multiple scenarios with varying difficulty

    Example:
        splits = ShipyardDataSplits(base_config, seed=42)
        train_cfg = splits.get_train_config()
        val_cfg = splits.get_val_config()
        test_cfgs = splits.get_test_configs(n_scenarios=10)
    """

    def __init__(self, base_config: Dict[str, Any], seed: int = 42):
        """Initialize data splits manager.

        Args:
            base_config: Base configuration dictionary
            seed: Base random seed for reproducibility
        """
        self.base_config = base_config
        self.seed = seed

    def get_train_config(self) -> Dict[str, Any]:
        """Get training configuration with domain randomization enabled.

        Returns:
            Configuration dict optimized for training with exploration.
        """
        cfg = copy.deepcopy(self.base_config)
        cfg["domain_randomization"] = True
        cfg["seed"] = self.seed
        return cfg

    def get_val_config(self) -> Dict[str, Any]:
        """Get validation configuration with fixed seed and no randomization.

        Validation uses a different seed range (seed + 1000) to ensure
        episodes are unseen during training while remaining deterministic.

        Returns:
            Configuration dict for validation (deterministic evaluation).
        """
        cfg = copy.deepcopy(self.base_config)
        cfg["domain_randomization"] = False
        cfg["seed"] = self.seed + 1000
        return cfg

    def get_test_configs(self, n_scenarios: int = 10) -> List[Dict[str, Any]]:
        """Get multiple test configurations for robust evaluation.

        Test scenarios vary in:
        - Random seed (seed + 2000 + i)
        - Number of blocks (base + i * 10)
        - Max time (scaled proportionally)

        Args:
            n_scenarios: Number of test scenarios to generate

        Returns:
            List of configuration dicts for testing.
        """
        configs = []
        base_n_blocks = self.base_config.get("n_blocks", 50)
        base_max_time = self.base_config.get("max_time", 10000)

        for i in range(n_scenarios):
            cfg = copy.deepcopy(self.base_config)
            cfg["domain_randomization"] = False
            cfg["seed"] = self.seed + 2000 + i

            # Vary difficulty
            extra_blocks = i * 10
            cfg["n_blocks"] = base_n_blocks + extra_blocks

            # Scale max time proportionally to block count
            scale_factor = (base_n_blocks + extra_blocks) / base_n_blocks
            cfg["max_time"] = int(base_max_time * scale_factor)

            configs.append(cfg)

        return configs

    def get_ablation_configs(self, ablations: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate configurations for ablation studies.

        Args:
            ablations: Dict mapping parameter names to lists of values to test.
                      Example: {"entropy_coef": [0.01, 0.1, 0.5]}

        Returns:
            List of configurations for each ablation combination.
        """
        configs = []

        # Generate configs for each ablation parameter
        for param_name, values in ablations.items():
            for value in values:
                cfg = copy.deepcopy(self.base_config)
                cfg["domain_randomization"] = False
                cfg["seed"] = self.seed + 3000  # Fixed seed for ablations

                # Handle nested parameters (e.g., "ppo.entropy_coef")
                if "." in param_name:
                    parts = param_name.split(".")
                    current = cfg
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = value
                else:
                    cfg[param_name] = value

                cfg["_ablation_param"] = param_name
                cfg["_ablation_value"] = value
                configs.append(cfg)

        return configs
