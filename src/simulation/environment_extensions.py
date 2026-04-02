"""Environment extensions: spatial constraints, labor leveling, shifts, weather, duration uncertainty.

These are mixin methods injected into HHIShipyardEnv via _init_extensions()
and called from _advance_simulation(). Each feature is gated by a config flag.
All five extensions support stochastic modeling for realistic simulation.

Config keys:
  extensions:
    enable_spatial: true       # Spatial constraints (SPMT congestion, crane reach)
    enable_shifts: true        # Shift start/stop (day/night, breaks)
    enable_weather: true       # Weather effects on outdoor work
    labor_leveling: true       # Enforce crew requirements per stage
    duration_uncertainty: true  # Stochastic processing time variation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional
import numpy as np

if TYPE_CHECKING:
    from .shipyard_env import HHIShipyardEnv
    from .entities import Block


# =========================================================================
# 1. Spatial Constraints
# =========================================================================

# Maximum number of SPMTs that can use a route segment simultaneously
MAX_SPMTS_PER_ROUTE = 3

# Crane reach: each crane can only serve docks within its rail range
# Maps crane_id (0-8) to list of dock indices it can serve
CRANE_DOCK_ASSIGNMENTS = {
    0: [0],        # GC01 → Dock 1
    1: [0],        # GC02 → Dock 1
    2: [1],        # GC03 → Dock 2
    3: [2],        # GC04 → Dock 3 (MEGA)
    4: [2],        # GC05 → Dock 3 (MEGA)
    5: [3],        # GC06 → Dock 4
    6: [4],        # GC07 → Dock 5
    7: [7],        # GC08 → Dock 8
    8: [9],        # GC09 → H-Dock
}


def init_spatial(env: HHIShipyardEnv) -> None:
    """Initialize spatial constraint tracking."""
    env._route_usage: Dict[str, int] = {}  # route_key -> n_spmts currently using
    env._spmt_routes: Dict[str, str] = {}  # spmt_id -> current route_key


def check_route_available(env: HHIShipyardEnv, origin: str, destination: str) -> bool:
    """Check if a route segment has capacity for another SPMT."""
    ext = env.config.get("extensions", {})
    if not ext.get("enable_spatial", False):
        return True  # No spatial constraints
    route_key = f"{origin}->{destination}"
    current = getattr(env, "_route_usage", {}).get(route_key, 0)
    return current < MAX_SPMTS_PER_ROUTE


def apply_congestion_delay(env: HHIShipyardEnv, base_travel_time: float,
                           origin: str, destination: str) -> float:
    """Apply congestion-based delay to travel time.

    Each additional SPMT on the route adds 20% delay (queuing at
    loading/unloading points). Stochastic noise models variability in
    loading/unloading operations and route conditions.
    """
    ext = env.config.get("extensions", {})
    if not ext.get("enable_spatial", False):
        return base_travel_time  # No spatial constraints
    route_key = f"{origin}->{destination}"
    n_users = getattr(env, "_route_usage", {}).get(route_key, 0)
    congestion_factor = 1.0 + 0.2 * max(n_users - 1, 0)

    # Stochastic: ±10% log-normal noise on travel time
    rng = getattr(env, "_ext_rng", None)
    if rng is not None:
        noise = rng.lognormal(0.0, 0.1)  # mean=1.0, ~10% CV
    else:
        noise = 1.0

    return base_travel_time * congestion_factor * noise


def check_crane_reach(env: HHIShipyardEnv, crane_idx: int, dock_idx: int) -> bool:
    """Check if crane can reach the specified dock."""
    ext = env.config.get("extensions", {})
    if not ext.get("enable_spatial", False):
        return True  # No spatial constraints
    allowed = CRANE_DOCK_ASSIGNMENTS.get(crane_idx, list(range(10)))
    return dock_idx in allowed


# =========================================================================
# 2. Labor Resource Leveling
# =========================================================================

# Stochastic: per-step probability that any individual worker is absent
LABOR_ABSENCE_RATE = 0.05

# Minimum crew sizes per production stage (workers required simultaneously)
STAGE_CREW_REQUIREMENTS = {
    "STEEL_CUTTING": 8,
    "PART_FABRICATION": 6,
    "PANEL_ASSEMBLY": 10,
    "BLOCK_ASSEMBLY": 16,
    "BLOCK_OUTFITTING": 12,
    "PAINTING": 8,
    "PRE_ERECTION": 10,
    "ERECTION": 20,      # Crane ops need large crew
}

# Skill requirements per stage
STAGE_SKILL_NEEDS = {
    "STEEL_CUTTING": {"WELDER": 4, "FITTER": 4},
    "PART_FABRICATION": {"WELDER": 3, "FITTER": 3},
    "PANEL_ASSEMBLY": {"WELDER": 6, "FITTER": 4},
    "BLOCK_ASSEMBLY": {"WELDER": 8, "FITTER": 6, "ELECTRICIAN": 2},
    "BLOCK_OUTFITTING": {"ELECTRICIAN": 4, "FITTER": 4, "WELDER": 4},
    "PAINTING": {"PAINTER": 6, "FITTER": 2},
    "PRE_ERECTION": {"FITTER": 6, "CRANE_OPERATOR": 2, "WELDER": 2},
    "ERECTION": {"CRANE_OPERATOR": 4, "FITTER": 8, "WELDER": 8},
}


def check_labor_available(env: HHIShipyardEnv, stage_name: str) -> bool:
    """Check if enough workers are available for the given production stage."""
    ext = env.config.get("extensions", {})
    if not ext.get("labor_leveling", False):
        return True

    required = STAGE_CREW_REQUIREMENTS.get(stage_name, 1)
    # Count available workers across all pools
    labor_pools = env.entities.get("labor_pools", [])
    if not labor_pools:
        return True  # No labor pools defined — assume fully staffed
    total_available = sum(p.available_workers for p in labor_pools)
    return total_available >= required


def apply_labor_penalty(env: HHIShipyardEnv, base_time: float,
                        stage_name: str) -> float:
    """Apply processing time penalty when crew is understaffed.

    If fewer workers than optimal are available, processing takes longer
    proportionally (half crew = double time, capped at 3x).
    Stochastic: models random worker absences (5% absence rate per step).
    """
    ext = env.config.get("extensions", {})
    if not ext.get("labor_leveling", False):
        return base_time

    required = STAGE_CREW_REQUIREMENTS.get(stage_name, 1)

    # Get base available count from labor pools
    labor_pools = env.entities.get("labor_pools", [])
    if labor_pools:
        total_available = sum(p.available_workers for p in labor_pools)
    else:
        total_available = required  # Assume fully staffed if no pools defined

    # Stochastic absences: each worker has 5% chance of being absent
    rng = getattr(env, "_ext_rng", None)
    if rng is not None and total_available > 0:
        absent = rng.binomial(total_available, LABOR_ABSENCE_RATE)
        total_available = max(total_available - absent, 1)

    if total_available >= required:
        return base_time

    # Understaffed: time increases inversely with available ratio
    ratio = max(total_available / required, 0.33)  # Min 33% staffing
    return base_time / ratio


# =========================================================================
# 3. Shifts / Start-Stop
# =========================================================================

# Stochastic shift parameters
SHIFT_OVERTIME_PROB = 0.10   # 10% chance of working through a break
SHIFT_FATIGUE_STD = 0.05     # ±5% productivity variation from fatigue

# Shift schedule: 2 shifts per day (12h each), with 30-min handover gap
SHIFT_CONFIG = {
    "shift_duration_hours": 12.0,
    "handover_gap_hours": 0.5,       # No work during shift change
    "breaks_per_shift": 2,           # Two 30-min breaks per shift
    "break_duration_hours": 0.5,
    "weekend_slowdown": 0.5,         # 50% capacity on weekends (Sat/Sun)
    "hours_per_day": 24.0,
}


def is_work_active(env: HHIShipyardEnv) -> bool:
    """Check if work is currently active (not in shift change or break).

    Returns False during:
    - Shift change gaps (30 min at shift boundaries)
    - Scheduled breaks
    """
    ext = env.config.get("extensions", {})
    if not ext.get("enable_shifts", False):
        return True

    cfg = SHIFT_CONFIG
    hour_of_day = env.sim_time % cfg["hours_per_day"]
    shift_dur = cfg["shift_duration_hours"]
    gap = cfg["handover_gap_hours"]

    # Check shift boundaries (at 0h and 12h)
    for boundary in [0.0, shift_dur]:
        if boundary <= hour_of_day < boundary + gap:
            return False

    # Check breaks: at 1/3 and 2/3 through each shift
    shift_start = 0.0 if hour_of_day < shift_dur else shift_dur
    hours_into_shift = hour_of_day - shift_start
    break_dur = cfg["break_duration_hours"]
    shift_third = (shift_dur - gap) / 3

    for i in [1, 2]:
        break_start = gap + shift_third * i
        if break_start <= hours_into_shift < break_start + break_dur:
            return False

    return True


def get_shift_productivity(env: HHIShipyardEnv) -> float:
    """Get productivity multiplier based on shift schedule.

    Returns 0.0 during breaks/handovers, reduced on weekends.
    Stochastic: models fatigue-based productivity variation (±5%) and
    random overtime probability (10% chance of extended shift).
    """
    ext = env.config.get("extensions", {})
    if not ext.get("enable_shifts", False):
        return 1.0

    if not is_work_active(env):
        # Stochastic overtime: 10% chance of working through a break
        rng = getattr(env, "_ext_rng", None)
        if rng is not None and rng.random() < SHIFT_OVERTIME_PROB:
            return 0.8  # Overtime at reduced productivity
        return 0.0

    # Weekend slowdown (assuming day 0 = Monday, so days 5-6 = weekend)
    day_of_week = int(env.sim_time / 24.0) % 7
    base = SHIFT_CONFIG["weekend_slowdown"] if day_of_week >= 5 else 1.0

    # Stochastic fatigue: productivity varies ±5% within a shift
    rng = getattr(env, "_ext_rng", None)
    if rng is not None:
        fatigue_noise = rng.normal(1.0, SHIFT_FATIGUE_STD)
        base *= max(fatigue_noise, 0.7)  # Floor at 70%

    return base


# =========================================================================
# 4. Weather Effects
# =========================================================================

# Weather state machine (Markov chain)
WEATHER_STATES = ["clear", "cloudy", "rain", "storm"]

# Transition probabilities (per hour)
WEATHER_TRANSITIONS = {
    "clear":  {"clear": 0.95, "cloudy": 0.04, "rain": 0.01, "storm": 0.00},
    "cloudy": {"clear": 0.10, "cloudy": 0.80, "rain": 0.08, "storm": 0.02},
    "rain":   {"clear": 0.02, "cloudy": 0.15, "rain": 0.75, "storm": 0.08},
    "storm":  {"clear": 0.00, "cloudy": 0.05, "rain": 0.25, "storm": 0.70},
}

# Impact on processing: multiplier on outdoor work speed
WEATHER_SPEED_MULTIPLIER = {
    "clear": 1.0,
    "cloudy": 1.0,
    "rain": 0.6,     # Rain slows outdoor work by 40%
    "storm": 0.0,    # Storm halts outdoor work entirely
}

# Which stages are outdoor (affected by weather)
OUTDOOR_STAGES = {
    "PRE_ERECTION", "ERECTION",  # Staging and crane ops are outdoors
}

# Outdoor facilities
OUTDOOR_FACILITIES = {
    "grand_block_staging_north", "grand_block_staging_south",
    "outfitting_quay_1", "outfitting_quay_2", "outfitting_quay_3",
}


def init_weather(env: HHIShipyardEnv) -> None:
    """Initialize weather state."""
    env._weather_state = "clear"
    env._weather_rng = np.random.RandomState(42)


def update_weather(env: HHIShipyardEnv, dt: float) -> None:
    """Advance weather state by one timestep using Markov chain."""
    ext = env.config.get("extensions", {})
    if not ext.get("enable_weather", False):
        return

    if not hasattr(env, "_weather_state"):
        init_weather(env)

    current = env._weather_state
    probs = WEATHER_TRANSITIONS[current]
    states = list(probs.keys())
    probabilities = [probs[s] for s in states]

    # Transition (probabilistic per timestep)
    env._weather_state = env._weather_rng.choice(states, p=probabilities)

    # Track in metrics
    if not hasattr(env, "_weather_history"):
        env._weather_history: List[str] = []
    env._weather_history.append(env._weather_state)


def get_weather_multiplier(env: HHIShipyardEnv, facility_name: str) -> float:
    """Get processing speed multiplier based on weather and facility location.

    Only outdoor facilities are affected. Indoor facilities always return 1.0.
    """
    ext = env.config.get("extensions", {})
    if not ext.get("enable_weather", False):
        return 1.0

    if not hasattr(env, "_weather_state"):
        return 1.0

    # Check if facility is outdoor
    is_outdoor = facility_name in OUTDOOR_FACILITIES
    if not is_outdoor:
        # Also check if the stage associated with this facility is outdoor
        stage_map = getattr(env, "_STAGE_MAP", {})
        stage = stage_map.get(facility_name)
        if stage is not None:
            stage_name = stage.name if hasattr(stage, "name") else str(stage)
            is_outdoor = stage_name in OUTDOOR_STAGES

    if not is_outdoor:
        return 1.0

    return WEATHER_SPEED_MULTIPLIER.get(env._weather_state, 1.0)


def get_weather_state(env: HHIShipyardEnv) -> str:
    """Get current weather state."""
    return getattr(env, "_weather_state", "clear")


# =========================================================================
# 5. Duration Uncertainty
# =========================================================================

# Per-stage coefficient of variation for processing time noise
# Higher values = more variable stages (block assembly is most variable)
DURATION_CV = {
    "STEEL_CUTTING": 0.10,
    "PART_FABRICATION": 0.12,
    "PANEL_ASSEMBLY": 0.15,
    "BLOCK_ASSEMBLY": 0.20,      # Most complex, most variable
    "BLOCK_OUTFITTING": 0.15,
    "PAINTING": 0.10,
    "PRE_ERECTION": 0.12,
    "ERECTION": 0.18,            # Weather-dependent, variable
    "QUAY_OUTFITTING": 0.15,
    "SEA_TRIALS": 0.25,          # Highly variable (rework, retesting)
    "DELIVERY": 0.05,
}

# Default CV for stages not explicitly listed
DEFAULT_DURATION_CV = 0.15


def apply_duration_noise(env: HHIShipyardEnv, base_duration: float,
                         stage_name: str = "") -> float:
    """Apply stochastic noise to a processing duration.

    Uses log-normal distribution to ensure positive durations.
    The coefficient of variation (CV = std/mean) is stage-dependent,
    modeling that some operations are inherently more variable than others.

    Returns the noisy duration (always positive, mean-preserving).
    """
    ext = env.config.get("extensions", {})
    if not ext.get("duration_uncertainty", False):
        return base_duration

    rng = getattr(env, "_ext_rng", None)
    if rng is None:
        return base_duration

    cv = DURATION_CV.get(stage_name, DEFAULT_DURATION_CV)
    if cv <= 0 or base_duration <= 0:
        return base_duration

    # Log-normal parameterization: mean = base_duration, CV = sigma
    # sigma² = ln(1 + CV²), mu = ln(base_duration) - sigma²/2
    sigma_sq = np.log(1 + cv ** 2)
    sigma = np.sqrt(sigma_sq)
    mu = np.log(base_duration) - sigma_sq / 2

    noisy = rng.lognormal(mu, sigma)

    # Clamp to [0.5x, 2.0x] of base to prevent extreme outliers
    return float(np.clip(noisy, base_duration * 0.5, base_duration * 2.0))


def get_duration_noise_factor(env: HHIShipyardEnv, stage_name: str = "") -> float:
    """Get a multiplicative noise factor for per-step processing rate.

    Unlike apply_duration_noise (which is applied once at task start),
    this provides per-step variation modeling micro-level variability
    (worker skill, tool availability, rework within a step).
    """
    ext = env.config.get("extensions", {})
    if not ext.get("duration_uncertainty", False):
        return 1.0

    rng = getattr(env, "_ext_rng", None)
    if rng is None:
        return 1.0

    cv = DURATION_CV.get(stage_name, DEFAULT_DURATION_CV)
    # Per-step noise is smaller than per-task noise (sqrt reduction)
    step_cv = cv * 0.3  # 30% of task-level CV for step-level noise
    noise = rng.lognormal(0.0, step_cv)
    return float(np.clip(noise, 0.7, 1.3))


# =========================================================================
# 6. Combined processing time modifier
# =========================================================================

def get_effective_dt(env: HHIShipyardEnv, facility_name: str,
                     stage_name: str = "") -> float:
    """Get effective processing progress for one timestep.

    Combines all stochastic modifiers:
    - Shift productivity (0.0 during breaks, 0.5 on weekends, ±5% fatigue)
    - Weather (0.0 during storms for outdoor facilities, Markov chain)
    - Duration uncertainty (per-step log-normal noise on processing rate)

    Returns the effective dt to apply to processing (0.0 = no progress).
    """
    dt = 1.0  # Base timestep

    # Shift productivity (0.0 during breaks, 0.5 on weekends, stochastic fatigue)
    dt *= get_shift_productivity(env)

    # Weather (0.0 during storms for outdoor facilities)
    dt *= get_weather_multiplier(env, facility_name)

    # Duration uncertainty (per-step noise on processing rate)
    dt *= get_duration_noise_factor(env, stage_name)

    return dt


# =========================================================================
# Extension initialization and per-step update
# =========================================================================

def init_extensions(env: HHIShipyardEnv) -> None:
    """Initialize all enabled extensions. Call from __init__."""
    ext = env.config.get("extensions", {})

    # Shared RNG for all stochastic extensions (seeded for reproducibility)
    seed = env.config.get("seed", 42)
    env._ext_rng = np.random.RandomState(seed)

    if ext.get("enable_spatial", False):
        init_spatial(env)

    if ext.get("enable_weather", False):
        init_weather(env)

    # Store extension flags for quick access
    env._ext_spatial = ext.get("enable_spatial", False)
    env._ext_shifts = ext.get("enable_shifts", False)
    env._ext_weather = ext.get("enable_weather", False)
    env._ext_labor = ext.get("labor_leveling", False)
    env._ext_duration = ext.get("duration_uncertainty", False)


def update_extensions(env: HHIShipyardEnv, dt: float) -> None:
    """Per-step update for all extensions. Call from _advance_simulation."""
    if getattr(env, "_ext_weather", False):
        update_weather(env, dt)
