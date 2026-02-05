# Problem Formulation: Health-Aware Shipyard Scheduling

This document provides the formal mathematical formulation of the health-aware shipyard block scheduling problem as a Markov Decision Process (MDP).

## 1. Problem Overview

We consider an integrated shipyard scheduling problem where ship blocks must progress through multiple production stages before final assembly on the dock. The problem combines:
- **Production scheduling**: Routing blocks through facilities
- **Vehicle routing**: Dispatching SPMTs (Self-Propelled Modular Transporters) for block transport
- **Crane scheduling**: Coordinating crane lifts for dock placement
- **Predictive maintenance**: Managing equipment health to minimize breakdowns

## 2. Markov Decision Process Formulation

### 2.1 State Space S

The state at time $t$ is represented as a heterogeneous graph $G_t = (V, E)$ where:

**Node Types:**
- **Block nodes** $V_B = \{b_1, ..., b_n\}$: Ship blocks with features
  - $s_i^{(stage)} \in \{0, ..., 6\}$: Current production stage
  - $s_i^{(loc)} \in \mathcal{L}$: Current location
  - $s_i^{(comp)} \in [0, 1]$: Completion percentage
  - $s_i^{(due)} \in \mathbb{R}^+$: Time to due date
  - $s_i^{(pred)} \in \{0, 1\}$: Predecessor completion flag
  - $s_i^{(weight)}$: Normalized block weight
  - $s_i^{(transit)}, s_i^{(waiting)} \in \{0, 1\}$: Status flags

- **SPMT nodes** $V_S = \{v_1, ..., v_m\}$: Transport vehicles with features
  - $s_j^{(status)} \in \{idle, traveling\_empty, traveling\_loaded, maintenance, broken\}$
  - $s_j^{(loc)} \in \mathcal{L}$: Current location
  - $s_j^{(load)} \in [0, 1]$: Load ratio
  - $\mathbf{h}_j = [h_j^{(hyd)}, h_j^{(tire)}, h_j^{(eng)}] \in [0, 1]^3$: Health vector

- **Crane nodes** $V_C = \{c_1, ..., c_k\}$: Dock cranes with features
  - $s_l^{(pos)} \in \mathbb{R}$: Position on rail
  - $s_l^{(status)} \in \{idle, lifting, positioning, maintenance, broken\}$
  - $\mathbf{h}_l = [h_l^{(cable)}, h_l^{(motor)}] \in [0, 1]^2$: Health vector

- **Facility nodes** $V_F = \{f_1, ..., f_p\}$: Production facilities
  - $s_q^{(queue)} \in \mathbb{Z}^+$: Queue length
  - $s_q^{(util)} \in [0, 1]$: Utilization rate
  - $s_q^{(wait)} \in \mathbb{R}^+$: Average wait time

**Edge Types:**
- $(b, needs\_transport, v)$: Block-to-SPMT transport needs
- $(v, can\_transport, b)$: SPMT-to-block transport capability
- $(b, needs\_lift, c)$: Block-to-crane lift needs
- $(c, can\_lift, b)$: Crane-to-block lift capability
- $(b, at, f)$: Block location at facility
- $(b, precedes, b')$: Precedence constraints
- $(v, at, f)$: SPMT location at facility
- $(c, at, f)$: Crane location at facility (dock)

### 2.2 Action Space A

The action space is hierarchical with four action types:

$$A = A_{spmt} \cup A_{crane} \cup A_{maint} \cup A_{hold}$$

**Action Type 0 - Dispatch SPMT:**
$$a^{(0)} = (spmt\_idx, request\_idx) \in \{0, ..., m-1\} \times \{0, ..., |R_t|-1\}$$

**Action Type 1 - Dispatch Crane:**
$$a^{(1)} = (crane\_idx, lift\_idx) \in \{0, ..., k-1\} \times \{0, ..., |L_t|-1\}$$

**Action Type 2 - Trigger Maintenance:**
$$a^{(2)} = (equipment\_idx) \in \{0, ..., m+k-1\}$$

**Action Type 3 - Hold:**
$$a^{(3)} = \emptyset$$

### 2.3 Action Masking

Invalid actions are masked based on:

**SPMT Dispatch Mask:**
$$M_{spmt}[i,j] = \mathbb{1}[status_i = idle] \cdot \mathbb{1}[h_i^{min} > H_{fail}] \cdot \mathbb{1}[w_j \leq cap_i]$$

**Crane Dispatch Mask:**
$$M_{crane}[i,j] = \mathbb{1}[status_i = idle] \cdot \mathbb{1}[h_i^{min} > H_{fail}] \cdot \mathbb{1}[w_j \leq cap_i] \cdot \mathbb{1}[pred\_complete(b_j)]$$

**Maintenance Mask:**
$$M_{maint}[i] = \mathbb{1}[status_i = idle] \cdot \mathbb{1}[h_i^{min} < H_{pm}]$$

### 2.4 Transition Dynamics P(s'|s,a)

The transition dynamics follow a discrete-event simulation:

**Facility Processing:**
- Blocks enter facility queues and are processed FIFO
- Processing time: $\tau_{proc} \sim LogNormal(\mu_f, \sigma_f)$
- Completed blocks generate transport/lift requests

**SPMT Transport:**
- Travel time: $\tau_{travel} = d(loc_{src}, loc_{dst}) / speed$
- Load/unload time: $\tau_{load} = const$

**Crane Operations:**
- Positioning time: $\tau_{pos} = |pos_{crane} - pos_{block}| / speed_{rail}$
- Lift duration: $\tau_{lift} = const$

**Equipment Degradation (Wiener Process):**
$$dH_i(t) = -\mu(load_i) dt + \sigma dW_t$$

where:
- $\mu(load) = \mu_{base} + \mu_{load} \cdot load\_ratio$
- $\sigma$ = volatility parameter
- Failure occurs when $H_i < H_{fail}$

### 2.5 Reward Function R(s, a, s')

The reward function is a weighted sum of multiple objectives:

$$R(s, a, s') = R_{comp} + R_{tardy} + R_{breakdown} + R_{maint} + R_{empty}$$

**Components:**

1. **Completion Reward:**
$$R_{comp} = w_{comp} \cdot \mathbb{1}[block\_placed\_on\_dock]$$

2. **Tardiness Penalty:**
$$R_{tardy} = -w_{tardy} \cdot \sum_{b \in B_{active}} \max(0, t - due_b) \cdot \Delta t$$

3. **Breakdown Penalty:**
$$R_{breakdown} = -w_{breakdown} \cdot |\{e : H_e < H_{fail}\}|$$

4. **Maintenance Cost:**
$$R_{maint} = -w_{maint} \cdot \mathbb{1}[maintenance\_triggered]$$

5. **Empty Travel Penalty:**
$$R_{empty} = -w_{empty} \cdot d_{empty}$$

**Default Weights:**
| Weight | Value | Description |
|--------|-------|-------------|
| $w_{comp}$ | 1.0 | Block completion reward |
| $w_{tardy}$ | 10.0 | Tardiness penalty per time unit |
| $w_{breakdown}$ | 100.0 | Equipment breakdown penalty |
| $w_{maint}$ | 5.0 | Maintenance cost |
| $w_{empty}$ | 0.1 | Empty travel penalty |

### 2.6 Discount Factor

$$\gamma = 0.99$$

## 3. Constraints

### 3.1 Precedence Constraints

Block $b_i$ can only be placed on dock after all predecessors:
$$\forall b_j \in Pred(b_i): status_{b_j} = PLACED\_ON\_DOCK$$

### 3.2 Capacity Constraints

**Facility Capacity:**
$$|B_f^{processing}| \leq C_f \quad \forall f \in F$$

**SPMT Capacity:**
$$weight_{load} \leq capacity_{spmt}$$

**Crane Capacity:**
$$weight_{block} \leq lift\_capacity_{crane}$$

### 3.3 Equipment Availability

Only IDLE equipment can be dispatched:
$$status_e = IDLE \land status_e \neq BROKEN\_DOWN$$

## 4. Equipment Degradation Model

### 4.1 Wiener Process with Load-Dependent Drift

The health of equipment component $i$ evolves according to:

$$H_i(t + \Delta t) = H_i(t) - \mu_i \cdot \Delta t + \sigma \cdot \sqrt{\Delta t} \cdot Z$$

where $Z \sim \mathcal{N}(0, 1)$

**Drift Rate:**
$$\mu_i = \mu_{base} + \mu_{load} \cdot \frac{weight_{current}}{capacity}$$

**Parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| $\mu_{base}$ | 0.05 | Base drift rate (per hour) |
| $\mu_{load}$ | 0.1 | Load-dependent drift factor |
| $\sigma$ | 0.02 | Volatility |
| $H_{fail}$ | 20.0 | Failure threshold |
| $H_{pm}$ | 40.0 | Preventive maintenance threshold |
| $H_{restore}$ | 95.0 | Post-maintenance health |

### 4.2 Remaining Useful Life (RUL) Estimation

Estimated RUL based on current health and degradation rate:

$$RUL_i = \frac{H_i - H_{fail}}{\mu_i}$$

## 5. Production Stages

Blocks progress through 7 stages:

| Stage | Index | Typical Duration | Capacity |
|-------|-------|------------------|----------|
| CUTTING | 0 | 8.0 ± 2.0 hrs | 2 |
| PANEL | 1 | 6.0 ± 1.5 hrs | 2 |
| ASSEMBLY | 2 | 10.0 ± 3.0 hrs | 2 |
| OUTFITTING | 3 | 12.0 ± 4.0 hrs | 2 |
| PAINTING | 4 | 8.0 ± 2.0 hrs | 2 |
| PRE_ERECTION | 5 | - | - |
| DOCK | 6 | - | 32 positions |

## 6. Complexity Analysis

### 6.1 State Space Size

$$|S| = O(n \cdot |Stages| \cdot |Locations| \cdot 2^{features} \cdot m \cdot |Status| \cdot H_{bins}^3 \cdot k \cdot H_{bins}^2)$$

For typical instance (50 blocks, 6 SPMTs, 2 cranes): $|S| \approx 10^{15}$

### 6.2 Action Space Size

$$|A| = O(m \cdot n + k \cdot n + m + k + 1)$$

For typical instance: $|A| \approx 400$

### 6.3 GNN Forward Pass Complexity

$$O(|E| \cdot d + |V| \cdot d^2)$$

where $d$ = hidden dimension, $|E|$ = edges, $|V|$ = nodes

## 7. Instance Sizes

| Instance | Blocks | SPMTs | Cranes | Max Time | State Dim |
|----------|--------|-------|--------|----------|-----------|
| Small | 50 | 6 | 2 | 5,000 | 465 |
| Medium | 150 | 9 | 3 | 15,000 | 1,371 |
| Large | 300 | 12 | 4 | 30,000 | 2,716 |

## 8. Objective

Find policy $\pi^*$ that maximizes expected cumulative discounted reward:

$$\pi^* = \arg\max_\pi \mathbb{E}\left[\sum_{t=0}^{T} \gamma^t R(s_t, a_t, s_{t+1}) \mid \pi\right]$$

subject to:
- Precedence constraints
- Capacity constraints
- Equipment availability constraints

## References

- Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347.
- Kipf, T. N., & Welling, M. (2017). "Semi-Supervised Classification with Graph Convolutional Networks." ICLR.
- Si, X. S., et al. (2011). "Remaining useful life estimation – A review on the statistical data driven approaches." European Journal of Operational Research.
