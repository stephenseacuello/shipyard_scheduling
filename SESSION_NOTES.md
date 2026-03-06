# Session Notes — Shipyard Scheduling RL

## Project Overview

ISE 572 Graduate Course Project — Reinforcement Learning for Shipyard Block Scheduling.

**Partners:** Stephen Eacuello, Steven Blum

**Objective:** Develop an RL-based scheduling system for shipyard block production that outperforms or matches traditional optimization baselines (PuLP MIP, MPC, GA) while providing real-time, adaptive scheduling decisions.

**Key Components:**
- Heterogeneous GNN encoder (block/SPMT/crane/facility nodes)
- Multi-head actor-critic policy (PPO, SAC, DAgger)
- Plate-informed block scheduling (11-phase HHI production pipeline)
- MES dashboard (Dash/Plotly, 11 tabs)
- Multiple baselines: Expert (EDD), PuLP MIP, GA, MPC (CP-SAT)

---

## Session Log

### Session 1: Foundation
- Built core simulation environment (`ShipyardEnv`, `HHIShipyardEnv`)
- Implemented GNN encoder with HeteroConv message passing
- Created PPO and DAgger training pipelines
- Integrated plate-level decomposition (11 HHI production phases)
- 153+ tests passing

### Session 2: Baselines & Integration
- Added PuLP MIP scheduler (CBC solver)
- Added GA scheduler (chromosome-based priority evolution)
- Added MPC scheduler (CP-SAT rolling horizon)
- Fixed GA division-by-zero bug, tuned PuLP parameters
- Created ISE 572 project proposal
- Added readiness gate logistic regression classifier to policy

### Session 3: Fixes & Scalability Analysis
- Fixed ship delivery termination bug (episodes now run through outfitting + sea trials)
- Fixed GA baseline (added `decide()` method for step-level dispatch)
- Fixed comparison pipeline (removed broken GA routing)
- Committed all fixes: `3791b9a`

**DAgger Results:**
- Small instance (50 blocks): 99.7% of expert throughput
- Medium HHI (200 blocks): **0% throughput** (complete failure to scale)
- Root cause: distribution shift as beta decreases, loss increases 3.14 -> 3.70

**Calibration Results:**
- Painting: R^2 = 0.88
- Block assembly: R^2 = -0.03 (bug: `p0` had 5 values for 6-parameter model)

**Comparison Results (medium_hhi, 200 blocks, 9 ships):**
| Agent   | Blocks | Ships | Throughput |
|---------|--------|-------|------------|
| Expert  | 900    | 9     | 0.1106     |
| MPC     | 76     | 0     | 0.0093     |
| GA      | 3-20   | 0     | ~0.002     |
| DAgger  | 0      | 0     | 0.0000     |

### Session 4: Comprehensive Improvements
- Created 8-phase improvement plan
- **Phase 1 (DAgger Curriculum Learning):** Added `RunningMeanStd` observation normalization, per-stage beta reset (0.8->0.2 per stage instead of global decay), per-stage max_steps scaling, dataset retention across stages (keep 50%), increased default hyperparameters (10 iterations, 15 init episodes, 8 DAgger episodes, 20 epochs), fixed hardcoded paths
- **Phase 2 (MPC Baseline):** Added adaptive horizon (cap decision variables at ~5000), request prioritization (top-k=20 most urgent), reduced solver time limit (5.0->2.0s), reduced control/replanning horizons
- **Phase 3 (Calibration Fix):** Fixed `p0` bug (5->6 values), added Ridge regression fallback before simple linear fallback
- **Phase 4 (Statistical Tests):** Created `experiments/statistical_comparison.py` with Mann-Whitney U, Cohen's d, 95% CI
- **Phase 1 Results (Curriculum DAgger, 19.8h training):**
  - Tiny (10 blocks): 100.6% of expert (loss converging)
  - Small (50 blocks): 100.0% of expert (loss converging)
  - Medium (200 blocks): **34.3% of expert** (up from 0% — loss diverging but throughput emerging)
  - HHI Ulsan (1600 blocks): still 0% — medium curriculum not sufficient for full scale
- 157 tests passing after all changes

---

## Key Findings

1. **Small instance (50 blocks) is too easy** — all agents achieve 100% block completion
2. **Expert (EDD) is the only baseline that scales** to 200+ blocks reliably
3. **DAgger has a critical scalability gap** (partially resolved):
   - Direct DAgger: 99.7% on small, 0% on medium
   - Curriculum DAgger: 100% on small, **34.3% on medium** (0% → 34.3%)
   - Loss still diverges at scale (1.79 → 2.77) — architectural capacity limit
   - Further improvement needs larger policy network or attention-based action heads
4. **Calibration p0 bug** caused block assembly R^2 = -0.03 (now fixed)
5. **MPC decision variable explosion** at scale: 200 reqs x 24 SPMTs x 50 horizon = 240K binary variables (fixed with adaptive horizon + request prioritization)
6. **GA chromosome evolution** too expensive for 200+ blocks (30 generations x 50 population = 1500 episode simulations per step)

## Architecture Decisions

- **Plates as metadata** (not independent entities): correct abstraction level. Full plate simulation would be 10-30x training cost
- **GNN encoder** produces fixed-size embedding regardless of instance size (hidden_dim x 4 = 512)
- **Policy heads** sized for largest environment (24 SPMTs, 9 cranes, 1600 blocks)
- **Ship delivery pipeline**: blocks erected -> ship AFLOAT -> quay outfitting (~200h) -> sea trials (~168h) -> DELIVERED

## Partner Integration Status

- **Awaiting:** 3D model decomposition JSON, PuLP model code, processing time observations
- **Ready:** JSON schema defined, plate loader implemented, synthetic fallback working

## File Structure (Key Files)

```
src/
  simulation/
    shipyard_env.py    # HHI plate-decomposition environment
    environment.py     # Base shipyard environment
    calibration.py     # Processing time coefficient fitting
  agent/
    gnn_encoder.py     # Heterogeneous GNN encoder
    policy.py          # ActorCriticPolicy with multi-head output
    ppo.py, sac.py     # RL algorithms
  baselines/
    rule_based.py      # Expert (EDD) scheduler
    pulp_scheduler.py  # PuLP MIP scheduler
    ga_scheduler.py    # Genetic Algorithm scheduler
    mpc_scheduler.py   # MPC (CP-SAT) scheduler
  mes/
    app.py             # Dash MES dashboard (11 tabs)
experiments/
    curriculum_dagger.py          # Curriculum DAgger training
    compare_pulp_rl.py            # Multi-agent comparison
    statistical_comparison.py     # Statistical significance tests
    calibrate_processing_times.py # Calibration pipeline
```
