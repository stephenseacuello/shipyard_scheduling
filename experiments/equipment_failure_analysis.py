#!/usr/bin/env python3
"""Equipment Failure Analysis Experiment.

This experiment analyzes how the shipyard scheduler handles equipment failures:
1. Runs extended simulation with increased degradation rates
2. Tracks when/which equipment fails
3. Analyzes recovery behavior after failures
4. Compares health-aware vs non-health-aware scheduling

Author: Claude Opus 4.5
Date: 2025-02-14
"""

from __future__ import annotations

import sys
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation.environment import ShipyardEnv
from src.simulation.degradation import WienerDegradationModel
from src.simulation.entities import SPMTStatus, GoliathCraneStatus


@dataclass
class FailureEvent:
    """Record of an equipment failure."""
    equipment_id: str
    equipment_type: str  # "SPMT" or "Goliath"
    failure_time: float
    health_at_failure: Dict[str, float]
    component_failed: str
    recovery_time: Optional[float] = None
    downtime: Optional[float] = None


@dataclass
class EquipmentTracker:
    """Tracks equipment health and failures over time."""
    health_history: Dict[str, List[Tuple[float, Dict[str, float]]]] = field(default_factory=lambda: defaultdict(list))
    failures: List[FailureEvent] = field(default_factory=list)
    maintenance_events: List[Dict[str, Any]] = field(default_factory=list)
    recovery_starts: Dict[str, float] = field(default_factory=dict)
    
    def record_health(self, equip_id: str, sim_time: float, health_dict: Dict[str, float]) -> None:
        """Record equipment health at a point in time."""
        self.health_history[equip_id].append((sim_time, health_dict.copy()))
    
    def record_failure(self, equip_id: str, equip_type: str, sim_time: float, 
                       health_dict: Dict[str, float], component: str) -> None:
        """Record a failure event."""
        event = FailureEvent(
            equipment_id=equip_id,
            equipment_type=equip_type,
            failure_time=sim_time,
            health_at_failure=health_dict.copy(),
            component_failed=component
        )
        self.failures.append(event)
        self.recovery_starts[equip_id] = sim_time
    
    def record_recovery(self, equip_id: str, sim_time: float) -> None:
        """Record when equipment recovers from failure."""
        if equip_id in self.recovery_starts:
            # Find the most recent failure for this equipment
            for event in reversed(self.failures):
                if event.equipment_id == equip_id and event.recovery_time is None:
                    event.recovery_time = sim_time
                    event.downtime = sim_time - event.failure_time
                    break
            del self.recovery_starts[equip_id]
    
    def record_maintenance(self, equip_id: str, equip_type: str, sim_time: float,
                          health_before: Dict[str, float], health_after: Dict[str, float]) -> None:
        """Record a maintenance event."""
        self.maintenance_events.append({
            "equipment_id": equip_id,
            "equipment_type": equip_type,
            "time": sim_time,
            "health_before": health_before.copy(),
            "health_after": health_after.copy()
        })


class HealthAwarePolicy:
    """Health-aware scheduling policy that triggers maintenance when health is low."""
    
    def __init__(self, maintenance_threshold: float = 40.0, prefer_healthy: bool = True):
        self.maintenance_threshold = maintenance_threshold
        self.prefer_healthy = prefer_healthy
    
    def select_action(self, env: ShipyardEnv, mask: Dict[str, np.ndarray]) -> Dict[str, int]:
        """Select action considering equipment health."""
        action = {
            "action_type": 3,  # Default: hold
            "spmt_idx": 0,
            "request_idx": 0,
            "crane_idx": 0,
            "lift_idx": 0,
            "equipment_idx": 0
        }
        
        # Priority 1: Perform maintenance on equipment with low health
        if mask["maintenance"].any():
            for i, needs_maint in enumerate(mask["maintenance"]):
                if needs_maint:
                    # Check if equipment health is below threshold
                    if i < len(env.spmts):
                        health = env.spmts[i].get_min_health()
                    else:
                        crane_idx = i - len(env.spmts)
                        if crane_idx < len(env.goliath_cranes):
                            health = env.goliath_cranes[crane_idx].get_min_health()
                        else:
                            continue
                    
                    if health < self.maintenance_threshold:
                        action["action_type"] = 2  # maintenance
                        action["equipment_idx"] = i
                        return action
        
        # Priority 2: Dispatch crane for lift requests (prefer healthiest)
        crane_dispatch = mask.get("crane_dispatch", np.array([]))
        if crane_dispatch.size > 0 and crane_dispatch.any() and len(env.lift_requests) > 0:
            crane_healths = []
            for i in range(crane_dispatch.shape[0]):
                if crane_dispatch[i].any():
                    crane_healths.append((i, env.goliath_cranes[i].get_min_health()))
            if crane_healths:
                if self.prefer_healthy:
                    crane_healths.sort(key=lambda x: x[1], reverse=True)
                best_crane_idx = crane_healths[0][0]
                # Find first valid lift for this crane
                for j in range(crane_dispatch.shape[1]):
                    if crane_dispatch[best_crane_idx, j]:
                        action["action_type"] = 1  # dispatch crane
                        action["crane_idx"] = best_crane_idx
                        action["lift_idx"] = j
                        return action
        
        # Priority 3: Dispatch SPMT for transport requests (prefer healthiest)
        spmt_dispatch = mask.get("spmt_dispatch", np.array([]))
        if spmt_dispatch.size > 0 and spmt_dispatch.any() and len(env.transport_requests) > 0:
            spmt_healths = []
            for i in range(spmt_dispatch.shape[0]):
                if spmt_dispatch[i].any():
                    spmt_healths.append((i, env.spmts[i].get_min_health()))
            if spmt_healths:
                if self.prefer_healthy:
                    spmt_healths.sort(key=lambda x: x[1], reverse=True)
                best_spmt_idx = spmt_healths[0][0]
                # Find first valid request for this SPMT
                for j in range(spmt_dispatch.shape[1]):
                    if spmt_dispatch[best_spmt_idx, j]:
                        action["action_type"] = 0  # dispatch SPMT
                        action["spmt_idx"] = best_spmt_idx
                        action["request_idx"] = j
                        return action
        
        # Priority 4: Preemptive maintenance on moderately degraded equipment
        if mask["maintenance"].any():
            for i, needs_maint in enumerate(mask["maintenance"]):
                if needs_maint:
                    action["action_type"] = 2
                    action["equipment_idx"] = i
                    return action
        
        return action


class NonHealthAwarePolicy:
    """Non-health-aware policy that ignores equipment health."""
    
    def select_action(self, env: ShipyardEnv, mask: Dict[str, np.ndarray]) -> Dict[str, int]:
        """Select action ignoring equipment health (no preemptive maintenance)."""
        action = {
            "action_type": 3,  # Default: hold
            "spmt_idx": 0,
            "request_idx": 0,
            "crane_idx": 0,
            "lift_idx": 0,
            "equipment_idx": 0
        }
        
        # Priority 1: Dispatch crane (first available, no health consideration)
        crane_dispatch = mask.get("crane_dispatch", np.array([]))
        if crane_dispatch.size > 0 and crane_dispatch.any() and len(env.lift_requests) > 0:
            for i in range(crane_dispatch.shape[0]):
                for j in range(crane_dispatch.shape[1]):
                    if crane_dispatch[i, j]:
                        action["action_type"] = 1
                        action["crane_idx"] = i
                        action["lift_idx"] = j
                        return action
        
        # Priority 2: Dispatch SPMT (first available, no health consideration)
        spmt_dispatch = mask.get("spmt_dispatch", np.array([]))
        if spmt_dispatch.size > 0 and spmt_dispatch.any() and len(env.transport_requests) > 0:
            for i in range(spmt_dispatch.shape[0]):
                for j in range(spmt_dispatch.shape[1]):
                    if spmt_dispatch[i, j]:
                        action["action_type"] = 0
                        action["spmt_idx"] = i
                        action["request_idx"] = j
                        return action
        
        # No preemptive maintenance - only react to breakdowns
        return action


def apply_forced_degradation(env: ShipyardEnv, dt: float = 1.0) -> List[Dict]:
    """Apply aggressive degradation to all equipment regardless of operating status.
    
    This simulates environmental wear, corrosion, and age-related degradation
    that occurs even when equipment is not actively operating.
    
    Returns list of failure events that occurred.
    """
    failures = []
    
    # Very aggressive degradation parameters
    base_decay = 0.08  # Health points per hour when idle
    operating_decay = 0.25  # Additional decay when operating
    load_factor = 0.15  # Extra decay per unit load ratio
    volatility = 0.3  # Randomness (standard deviation)
    failure_threshold = 20.0
    
    # Degrade SPMTs
    for spmt in env.spmts:
        if spmt.status == SPMTStatus.BROKEN_DOWN:
            continue
            
        # Check operating status
        operating = spmt.status in {
            SPMTStatus.TRAVELING_EMPTY,
            SPMTStatus.TRAVELING_LOADED,
            SPMTStatus.LOADING,
            SPMTStatus.UNLOADING,
        }
        
        # Calculate decay rate
        decay = base_decay
        if operating:
            decay += operating_decay
            if spmt.current_load:
                block = env._get_block(spmt.current_load)
                load_ratio = block.weight / spmt.capacity
                decay += load_factor * load_ratio
        
        # Apply degradation with noise
        noise = np.random.normal(0, volatility)
        
        # Degrade each component
        spmt.health_hydraulic = max(0, spmt.health_hydraulic - decay * dt + noise * np.sqrt(dt))
        spmt.health_tires = max(0, spmt.health_tires - (decay * 1.1) * dt + noise * np.sqrt(dt))  # Tires wear faster
        spmt.health_engine = max(0, spmt.health_engine - (decay * 0.9) * dt + noise * np.sqrt(dt))
        
        # Check for failure
        min_health = spmt.get_min_health()
        if min_health < failure_threshold:
            spmt.status = SPMTStatus.BROKEN_DOWN
            env.metrics["breakdowns"] = env.metrics.get("breakdowns", 0) + 1
            component = "hydraulic" if spmt.health_hydraulic < failure_threshold else \
                       "tires" if spmt.health_tires < failure_threshold else "engine"
            failures.append({
                "id": spmt.id,
                "type": "SPMT",
                "component": component,
                "health": {
                    "hydraulic": spmt.health_hydraulic,
                    "tires": spmt.health_tires,
                    "engine": spmt.health_engine
                }
            })
    
    # Degrade Goliath Cranes
    for crane in env.goliath_cranes:
        if crane.status == GoliathCraneStatus.BROKEN_DOWN:
            continue
            
        operating = crane.status in {GoliathCraneStatus.LIFTING, GoliathCraneStatus.POSITIONING}
        
        # Cranes degrade slower than SPMTs but still need maintenance
        crane_decay = base_decay * 0.7  # Cranes are more robust
        if operating:
            crane_decay += operating_decay * 0.8
        
        noise = np.random.normal(0, volatility * 0.8)
        
        crane.health_hoist = max(0, crane.health_hoist - (crane_decay * 1.2) * dt + noise * np.sqrt(dt))  # Hoist stressed most
        crane.health_trolley = max(0, crane.health_trolley - crane_decay * dt + noise * np.sqrt(dt))
        crane.health_gantry = max(0, crane.health_gantry - (crane_decay * 0.8) * dt + noise * np.sqrt(dt))
        
        min_health = crane.get_min_health()
        if min_health < failure_threshold:
            crane.status = GoliathCraneStatus.BROKEN_DOWN
            env.metrics["breakdowns"] = env.metrics.get("breakdowns", 0) + 1
            component = "hoist" if crane.health_hoist < failure_threshold else \
                       "trolley" if crane.health_trolley < failure_threshold else "gantry"
            failures.append({
                "id": crane.id,
                "type": "Goliath",
                "component": component,
                "health": {
                    "hoist": crane.health_hoist,
                    "trolley": crane.health_trolley,
                    "gantry": crane.health_gantry
                }
            })
    
    return failures


def perform_reactive_maintenance(env: ShipyardEnv) -> List[str]:
    """Perform reactive maintenance on broken equipment (simulating repair crews).
    
    In real shipyards, broken equipment gets repaired, not just left broken.
    This simulates a repair time of ~50-100 hours for major repairs.
    """
    repaired = []
    repair_time_per_step = 0.02  # 2% repair progress per hour (~50 hours to repair)
    
    # Track repair progress (stored in a simple dict)
    if not hasattr(env, '_repair_progress'):
        env._repair_progress = {}
    
    # Repair broken SPMTs
    for spmt in env.spmts:
        if spmt.status == SPMTStatus.BROKEN_DOWN:
            if spmt.id not in env._repair_progress:
                env._repair_progress[spmt.id] = 0.0
            
            env._repair_progress[spmt.id] += repair_time_per_step
            
            if env._repair_progress[spmt.id] >= 1.0:
                # Repair complete
                spmt.health_hydraulic = 90.0
                spmt.health_tires = 90.0
                spmt.health_engine = 90.0
                spmt.status = SPMTStatus.IDLE
                del env._repair_progress[spmt.id]
                repaired.append(spmt.id)
    
    # Repair broken cranes
    for crane in env.goliath_cranes:
        if crane.status == GoliathCraneStatus.BROKEN_DOWN:
            if crane.id not in env._repair_progress:
                env._repair_progress[crane.id] = 0.0
            
            env._repair_progress[crane.id] += repair_time_per_step * 0.5  # Cranes take longer to repair
            
            if env._repair_progress[crane.id] >= 1.0:
                crane.health_hoist = 90.0
                crane.health_trolley = 90.0
                crane.health_gantry = 90.0
                crane.status = GoliathCraneStatus.IDLE
                del env._repair_progress[crane.id]
                repaired.append(crane.id)
    
    return repaired


def run_simulation(env: ShipyardEnv, policy, tracker: EquipmentTracker, 
                   max_steps: int = 5000, verbose: bool = True,
                   use_health_aware_maintenance: bool = False) -> Dict[str, Any]:
    """Run simulation with given policy and track failures."""
    
    obs, info = env.reset()
    
    total_reward = 0.0
    blocks_completed = 0
    prev_failed_spmts = set()
    prev_failed_cranes = set()
    
    health_sample_interval = 50  # Sample health every 50 steps
    
    for step in range(max_steps):
        # Apply forced degradation (simulates environmental and age-related wear)
        new_failures = apply_forced_degradation(env, dt=1.0)
        
        # Record any new failures
        for failure in new_failures:
            tracker.record_failure(
                failure["id"], 
                failure["type"], 
                env.sim_time, 
                failure["health"],
                failure["component"]
            )
            if verbose:
                print(f"  [FAILURE] Step {step}: {failure['type']} {failure['id']} failed "
                      f"({failure['component']}) at health={failure['health'][failure['component']]:.1f}")
        
        # Perform reactive maintenance (repair broken equipment over time)
        repaired = perform_reactive_maintenance(env)
        for equip_id in repaired:
            tracker.record_recovery(equip_id, env.sim_time)
            if verbose:
                print(f"  [RECOVERY] Step {step}: {equip_id} repaired and operational")
        
        # Sample health periodically
        if step % health_sample_interval == 0:
            for spmt in env.spmts:
                health_dict = {
                    "hydraulic": spmt.health_hydraulic,
                    "tires": spmt.health_tires,
                    "engine": spmt.health_engine
                }
                tracker.record_health(spmt.id, env.sim_time, health_dict)
            
            for crane in env.goliath_cranes:
                health_dict = {
                    "hoist": crane.health_hoist,
                    "trolley": crane.health_trolley,
                    "gantry": crane.health_gantry
                }
                tracker.record_health(crane.id, env.sim_time, health_dict)
        
        # Get action mask and select action
        mask = env.get_action_mask()
        action = policy.select_action(env, mask)
        
        # Track maintenance actions
        if action["action_type"] == 2 and mask["maintenance"].any():
            equip_idx = action["equipment_idx"]
            if equip_idx < len(env.spmts):
                spmt = env.spmts[equip_idx]
                health_before = {
                    "hydraulic": spmt.health_hydraulic,
                    "tires": spmt.health_tires,
                    "engine": spmt.health_engine
                }
            else:
                crane_idx = equip_idx - len(env.spmts)
                if crane_idx < len(env.goliath_cranes):
                    crane = env.goliath_cranes[crane_idx]
                    health_before = {
                        "hoist": crane.health_hoist,
                        "trolley": crane.health_trolley,
                        "gantry": crane.health_gantry
                    }
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
        
        # Count current failures
        current_failed_spmts = sum(1 for s in env.spmts if s.status == SPMTStatus.BROKEN_DOWN)
        current_failed_cranes = sum(1 for c in env.goliath_cranes if c.status == GoliathCraneStatus.BROKEN_DOWN)
        
        # Progress indicator
        if verbose and step % 500 == 0:
            blocks_done = env.metrics.get("blocks_completed", 0)
            
            # Calculate average health
            avg_spmt_health = np.mean([s.get_min_health() for s in env.spmts])
            avg_crane_health = np.mean([c.get_min_health() for c in env.goliath_cranes])
            
            print(f"Step {step:5d} | Time: {env.sim_time:8.1f}h | "
                  f"Blocks: {blocks_done:3d} | "
                  f"SPMT Health: {avg_spmt_health:.1f} | Crane Health: {avg_crane_health:.1f} | "
                  f"Failed: {current_failed_spmts}S/{current_failed_cranes}C")
    
    return {
        "total_reward": total_reward,
        "blocks_completed": env.metrics.get("blocks_completed", 0),
        "total_breakdowns": env.metrics.get("breakdowns", 0),
        "planned_maintenance": env.metrics.get("planned_maintenance", 0),
        "final_time": env.sim_time,
        "steps": step + 1
    }


def analyze_results(tracker: EquipmentTracker) -> Dict[str, Any]:
    """Analyze failure and recovery statistics."""
    
    # Count failures by type
    spmt_failures = [f for f in tracker.failures if f.equipment_type == "SPMT"]
    goliath_failures = [f for f in tracker.failures if f.equipment_type == "Goliath"]
    
    # Component failure breakdown
    spmt_components = defaultdict(int)
    for f in spmt_failures:
        spmt_components[f.component_failed] += 1
    
    goliath_components = defaultdict(int)
    for f in goliath_failures:
        goliath_components[f.component_failed] += 1
    
    # Recovery time statistics
    spmt_downtimes = [f.downtime for f in spmt_failures if f.downtime is not None]
    goliath_downtimes = [f.downtime for f in goliath_failures if f.downtime is not None]
    
    # Time to first failure
    all_failure_times = [f.failure_time for f in tracker.failures]
    time_to_first_failure = min(all_failure_times) if all_failure_times else None
    
    return {
        "total_failures": len(tracker.failures),
        "spmt_failures": len(spmt_failures),
        "goliath_failures": len(goliath_failures),
        "spmt_component_failures": dict(spmt_components),
        "goliath_component_failures": dict(goliath_components),
        "spmt_avg_downtime": np.mean(spmt_downtimes) if spmt_downtimes else 0.0,
        "spmt_max_downtime": np.max(spmt_downtimes) if spmt_downtimes else 0.0,
        "goliath_avg_downtime": np.mean(goliath_downtimes) if goliath_downtimes else 0.0,
        "goliath_max_downtime": np.max(goliath_downtimes) if goliath_downtimes else 0.0,
        "maintenance_events": len(tracker.maintenance_events),
        "unrecovered_failures": sum(1 for f in tracker.failures if f.recovery_time is None),
        "time_to_first_failure": time_to_first_failure
    }


def main():
    """Run equipment failure analysis experiment."""
    
    print("=" * 80)
    print("EQUIPMENT FAILURE ANALYSIS EXPERIMENT")
    print("Shipyard Scheduling - HD Hyundai Heavy Industries Ulsan Configuration")
    print("=" * 80)
    print()
    
    # Load configuration
    config_path = "/Users/stepheneacuello/Projects/shipyard_scheduling/config/hhi_ulsan.yaml"
    print(f"Loading configuration: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config["max_time"] = 50000  # Extended time horizon
    
    print(f"\nForced degradation parameters:")
    print(f"  Base decay rate: 0.08 health/hour (idle)")
    print(f"  Operating decay: 0.25 health/hour (additional)")
    print(f"  Load factor: 0.15 health/hour per unit load")
    print(f"  Volatility: 0.3 (stochastic noise)")
    print(f"  Failure threshold: 20.0")
    print(f"  Repair time: ~50-100 hours")
    print(f"\nExpected: First failures within 500-1000 simulation hours")
    print()
    
    max_steps = 5000
    print(f"Running {max_steps}-step simulation...")
    print()
    
    # =========================================================================
    # Run 1: Health-Aware Policy
    # =========================================================================
    print("-" * 80)
    print("SCENARIO 1: Health-Aware Scheduling Policy")
    print("-" * 80)
    print("This policy preemptively maintains equipment and prefers healthier units.")
    print()
    
    env_ha = ShipyardEnv(config)
    tracker_ha = EquipmentTracker()
    policy_ha = HealthAwarePolicy(maintenance_threshold=40.0, prefer_healthy=True)
    
    start_time = time.time()
    results_ha = run_simulation(env_ha, policy_ha, tracker_ha, max_steps=max_steps,
                                use_health_aware_maintenance=True)
    ha_runtime = time.time() - start_time
    
    analysis_ha = analyze_results(tracker_ha)
    
    print(f"\nHealth-Aware Results (runtime: {ha_runtime:.1f}s):")
    print(f"  Blocks completed: {results_ha['blocks_completed']}")
    print(f"  Total reward: {results_ha['total_reward']:.2f}")
    print(f"  Total breakdowns: {results_ha['total_breakdowns']}")
    print(f"  Planned maintenance events: {results_ha['planned_maintenance']}")
    if analysis_ha['time_to_first_failure']:
        print(f"  Time to first failure: {analysis_ha['time_to_first_failure']:.1f}h")
    print()
    
    # =========================================================================
    # Run 2: Non-Health-Aware Policy
    # =========================================================================
    print("-" * 80)
    print("SCENARIO 2: Non-Health-Aware Scheduling Policy")
    print("-" * 80)
    print("This policy ignores equipment health and performs no preemptive maintenance.")
    print()
    
    env_nha = ShipyardEnv(config)
    tracker_nha = EquipmentTracker()
    policy_nha = NonHealthAwarePolicy()
    
    start_time = time.time()
    results_nha = run_simulation(env_nha, policy_nha, tracker_nha, max_steps=max_steps,
                                 use_health_aware_maintenance=False)
    nha_runtime = time.time() - start_time
    
    analysis_nha = analyze_results(tracker_nha)
    
    print(f"\nNon-Health-Aware Results (runtime: {nha_runtime:.1f}s):")
    print(f"  Blocks completed: {results_nha['blocks_completed']}")
    print(f"  Total reward: {results_nha['total_reward']:.2f}")
    print(f"  Total breakdowns: {results_nha['total_breakdowns']}")
    print(f"  Planned maintenance events: {results_nha['planned_maintenance']}")
    if analysis_nha['time_to_first_failure']:
        print(f"  Time to first failure: {analysis_nha['time_to_first_failure']:.1f}h")
    print()
    
    # =========================================================================
    # Comparative Analysis
    # =========================================================================
    print("=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)
    print()
    
    print("FAILURES BY EQUIPMENT TYPE:")
    print("-" * 65)
    print(f"{'Metric':<30} {'Health-Aware':>15} {'Non-Health-Aware':>18}")
    print("-" * 65)
    print(f"{'SPMT Failures':<30} {analysis_ha['spmt_failures']:>15} {analysis_nha['spmt_failures']:>18}")
    print(f"{'Goliath Crane Failures':<30} {analysis_ha['goliath_failures']:>15} {analysis_nha['goliath_failures']:>18}")
    print(f"{'Total Failures':<30} {analysis_ha['total_failures']:>15} {analysis_nha['total_failures']:>18}")
    print()
    
    print("SPMT COMPONENT FAILURES:")
    print("-" * 65)
    all_spmt_components = set(analysis_ha['spmt_component_failures'].keys()) | set(analysis_nha['spmt_component_failures'].keys())
    if all_spmt_components:
        for comp in sorted(all_spmt_components):
            ha_count = analysis_ha['spmt_component_failures'].get(comp, 0)
            nha_count = analysis_nha['spmt_component_failures'].get(comp, 0)
            print(f"  {comp:<28} {ha_count:>15} {nha_count:>18}")
    else:
        print("  No SPMT component failures recorded")
    print()
    
    print("GOLIATH CRANE COMPONENT FAILURES:")
    print("-" * 65)
    all_crane_components = set(analysis_ha['goliath_component_failures'].keys()) | set(analysis_nha['goliath_component_failures'].keys())
    if all_crane_components:
        for comp in sorted(all_crane_components):
            ha_count = analysis_ha['goliath_component_failures'].get(comp, 0)
            nha_count = analysis_nha['goliath_component_failures'].get(comp, 0)
            print(f"  {comp:<28} {ha_count:>15} {nha_count:>18}")
    else:
        print("  No Goliath crane component failures recorded")
    print()
    
    print("RECOVERY TIME ANALYSIS (hours):")
    print("-" * 65)
    print(f"{'SPMT Avg Downtime':<30} {analysis_ha['spmt_avg_downtime']:>15.1f} {analysis_nha['spmt_avg_downtime']:>18.1f}")
    print(f"{'SPMT Max Downtime':<30} {analysis_ha['spmt_max_downtime']:>15.1f} {analysis_nha['spmt_max_downtime']:>18.1f}")
    print(f"{'Goliath Avg Downtime':<30} {analysis_ha['goliath_avg_downtime']:>15.1f} {analysis_nha['goliath_avg_downtime']:>18.1f}")
    print(f"{'Goliath Max Downtime':<30} {analysis_ha['goliath_max_downtime']:>15.1f} {analysis_nha['goliath_max_downtime']:>18.1f}")
    print(f"{'Unrecovered Failures':<30} {analysis_ha['unrecovered_failures']:>15} {analysis_nha['unrecovered_failures']:>18}")
    print()
    
    print("THROUGHPUT IMPACT:")
    print("-" * 65)
    print(f"{'Blocks Completed':<30} {results_ha['blocks_completed']:>15} {results_nha['blocks_completed']:>18}")
    print(f"{'Total Reward':<30} {results_ha['total_reward']:>15.1f} {results_nha['total_reward']:>18.1f}")
    print(f"{'Planned Maintenance':<30} {results_ha['planned_maintenance']:>15} {results_nha['planned_maintenance']:>18}")
    
    # Calculate throughput impact
    if results_nha['blocks_completed'] > 0:
        throughput_improvement = ((results_ha['blocks_completed'] - results_nha['blocks_completed']) / 
                                  results_nha['blocks_completed'] * 100)
    else:
        throughput_improvement = 0 if results_ha['blocks_completed'] == 0 else float('inf')
    
    total_failures_ha = analysis_ha['total_failures']
    total_failures_nha = analysis_nha['total_failures']
    
    if total_failures_nha > 0:
        failure_reduction = ((total_failures_nha - total_failures_ha) / total_failures_nha * 100)
    else:
        failure_reduction = 0 if total_failures_ha == 0 else -100
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total simulation time: {results_ha['final_time']:.1f} hours (health-aware)")
    print(f"                       {results_nha['final_time']:.1f} hours (non-health-aware)")
    print()
    print(f"Throughput improvement with health-aware scheduling: {throughput_improvement:+.1f}%")
    print(f"Failure reduction with health-aware scheduling: {failure_reduction:+.1f}%")
    print()
    
    if total_failures_ha < total_failures_nha:
        print("CONCLUSION: Health-aware scheduling significantly reduces equipment failures")
        print("            and improves overall throughput by preemptively maintaining equipment.")
    elif total_failures_ha > total_failures_nha:
        print("CONCLUSION: Unexpected result - health-aware policy had more failures.")
        print("            This may indicate the maintenance threshold needs tuning.")
    else:
        if total_failures_ha == 0:
            print("CONCLUSION: No failures occurred in either scenario.")
            print("            Degradation rates may need to be increased further.")
        else:
            print("CONCLUSION: Similar failure rates between policies.")
            print("            Both policies experienced equipment degradation equally.")
    
    print()
    print("=" * 80)
    print("DETAILED FAILURE LOG (Health-Aware Policy)")
    print("=" * 80)
    if tracker_ha.failures:
        for i, f in enumerate(tracker_ha.failures[:25]):  # Show first 25
            recovery_str = f"recovered at {f.recovery_time:.1f}h" if f.recovery_time else "unrecovered"
            downtime_str = f"(downtime: {f.downtime:.1f}h)" if f.downtime else ""
            print(f"  {i+1:2d}. {f.equipment_type} {f.equipment_id} failed at {f.failure_time:.1f}h "
                  f"({f.component_failed}) - {recovery_str} {downtime_str}")
        
        if len(tracker_ha.failures) > 25:
            print(f"  ... and {len(tracker_ha.failures) - 25} more failures")
    else:
        print("  No failures recorded")
    
    print()
    print("=" * 80)
    print("DETAILED FAILURE LOG (Non-Health-Aware Policy)")
    print("=" * 80)
    if tracker_nha.failures:
        for i, f in enumerate(tracker_nha.failures[:25]):  # Show first 25
            recovery_str = f"recovered at {f.recovery_time:.1f}h" if f.recovery_time else "unrecovered"
            downtime_str = f"(downtime: {f.downtime:.1f}h)" if f.downtime else ""
            print(f"  {i+1:2d}. {f.equipment_type} {f.equipment_id} failed at {f.failure_time:.1f}h "
                  f"({f.component_failed}) - {recovery_str} {downtime_str}")
        
        if len(tracker_nha.failures) > 25:
            print(f"  ... and {len(tracker_nha.failures) - 25} more failures")
    else:
        print("  No failures recorded")
    
    print()
    return {
        "health_aware": {"results": results_ha, "analysis": analysis_ha},
        "non_health_aware": {"results": results_nha, "analysis": analysis_nha}
    }


if __name__ == "__main__":
    main()
