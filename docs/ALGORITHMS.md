# Algorithm Descriptions

This document provides pseudocode for the key algorithms used in the health-aware shipyard scheduling system.

## Algorithm 1: Health-Aware Shipyard Scheduling with GNN-PPO

```
Algorithm 1: GNN-PPO for Health-Aware Shipyard Scheduling
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Environment E, GNN encoder φ, Actor-Critic policy π
       Learning rate α, Discount γ, GAE parameter λ
       Clip ratio ε, Number of epochs K, Batch size B

Output: Trained policy π*

1:  Initialize encoder φ and policy π with random weights
2:  for epoch = 1 to N_epochs do
3:      // Collect rollout
4:      τ ← empty trajectory
5:      s ← E.reset()
6:      for t = 1 to T_horizon do
7:          G ← E.get_graph_data()           // Heterogeneous graph
8:          z ← φ(G)                          // GNN encoding
9:          M ← E.get_action_mask()           // Valid action mask
10:         a, log_π(a) ← π.sample(z, M)      // Masked sampling
11:         s', r, done ← E.step(a)
12:         τ.append(G, a, r, done, log_π(a), π.value(z))
13:         s ← s'
14:         if done then break
15:     end for
16:
17:     // Compute advantages using GAE
18:     A ← compute_gae(τ.rewards, τ.values, γ, λ)
19:     R ← A + τ.values                      // Returns
20:
21:     // PPO update
22:     for k = 1 to K do
23:         for batch in random_batches(τ, B) do
24:             G_batch ← batch_graphs(batch.graphs)
25:             z_batch ← φ(G_batch)          // Re-encode for gradients
26:             M_batch ← batch_masks(batch.masks)
27:
28:             // Policy loss
29:             log_π_new ← π.log_prob(z_batch, batch.actions, M_batch)
30:             ratio ← exp(log_π_new - batch.log_probs)
31:             L_clip ← min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
32:             L_policy ← -mean(L_clip)
33:
34:             // Value loss
35:             V ← π.value(z_batch)
36:             L_value ← mean((V - R)²)
37:
38:             // Entropy bonus
39:             H ← π.entropy(z_batch, M_batch)
40:
41:             // Total loss
42:             L ← L_policy + c_v * L_value - c_e * H
43:
44:             // Update
45:             ∇L ← backprop(L)
46:             clip_grad_norm(∇L, max_norm)
47:             optimizer.step(∇L)
48:         end for
49:     end for
50: end for
51: return π
```

## Algorithm 2: Heterogeneous Graph Attention Encoding

```
Algorithm 2: Heterogeneous GNN Encoder
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Heterogeneous graph G = (V, E) with node types
       {block, spmt, crane, facility} and 8 edge types
       Hidden dimension d, Number of layers L, Number of heads H

Output: Graph embedding z ∈ ℝ^(4d)

1:  // Input projection for each node type
2:  x_block ← W_block · X_block + b_block
3:  x_spmt ← W_spmt · X_spmt + b_spmt
4:  x_crane ← W_crane · X_crane + b_crane
5:  x_facility ← W_facility · X_facility + b_facility
6:
7:  x_dict ← {block: x_block, spmt: x_spmt,
8:            crane: x_crane, facility: x_facility}
9:
10: // Message passing layers
11: for l = 1 to L do
12:     x_dict_new ← {}
13:     for node_type in {block, spmt, crane, facility} do
14:         messages ← []
15:         for edge_type in E.edge_types_to(node_type) do
16:             // Multi-head attention for this edge type
17:             src_type ← edge_type.source
18:             e_ij ← E[edge_type]
19:
20:             // Attention computation
21:             Q ← W_Q^(l,edge_type) · x_dict[node_type]
22:             K ← W_K^(l,edge_type) · x_dict[src_type]
23:             V ← W_V^(l,edge_type) · x_dict[src_type]
24:
25:             α_ij ← softmax(Q · K^T / √(d/H))
26:             m_ij ← α_ij · V
27:             messages.append(aggregate(m_ij, e_ij))
28:         end for
29:
30:         // Aggregate messages from all edge types
31:         x_dict_new[node_type] ← mean(messages)
32:     end for
33:
34:     // Residual connection and normalization
35:     for node_type in x_dict do
36:         x_dict[node_type] ← LayerNorm(
37:             Dropout(ReLU(x_dict_new[node_type])) + x_dict[node_type]
38:         )
39:     end for
40: end for
41:
42: // Global pooling per node type
43: z_block ← mean_pool(x_dict[block])
44: z_spmt ← mean_pool(x_dict[spmt])
45: z_crane ← mean_pool(x_dict[crane])
46: z_facility ← mean_pool(x_dict[facility])
47:
48: // Concatenate for final embedding
49: z ← concat(z_block, z_spmt, z_crane, z_facility)
50: return z
```

## Algorithm 3: Hierarchical Action Masking

```
Algorithm 3: Hierarchical Action Masking for Policy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: State embedding z, Raw masks M_env from environment
       Policy heads {action_type, spmt, request, crane, lift, equipment}

Output: Valid action a, Log probability log_π(a)

1:  // Flatten 2D dispatch masks to 1D per-head masks
2:  M_spmt ← any(M_env.spmt_dispatch, axis=1)      // SPMT available
3:  M_request ← any(M_env.spmt_dispatch, axis=0)   // Request serviceable
4:  M_crane ← any(M_env.crane_dispatch, axis=1)    // Crane available
5:  M_lift ← any(M_env.crane_dispatch, axis=0)     // Lift serviceable
6:
7:  // Compute masked logits for action type
8:  logits_type ← head_action_type(z)
9:  logits_type[~M_env.action_type] ← -∞
10: dist_type ← Categorical(softmax(logits_type))
11: action_type ← dist_type.sample()
12:
13: // Sample sub-actions based on action type
14: if action_type == 0 then  // Dispatch SPMT
15:     logits_spmt ← head_spmt(z)
16:     logits_spmt[~M_spmt] ← -∞
17:     spmt_idx ← Categorical(softmax(logits_spmt)).sample()
18:
19:     // Mask requests valid for selected SPMT
20:     M_req_given_spmt ← M_env.spmt_dispatch[spmt_idx, :]
21:     logits_req ← head_request(z)
22:     logits_req[~M_req_given_spmt] ← -∞
23:     request_idx ← Categorical(softmax(logits_req)).sample()
24:
25:     a ← {type: 0, spmt: spmt_idx, request: request_idx}
26:     relevant_heads ← [action_type, spmt, request]
27:
28: else if action_type == 1 then  // Dispatch Crane
29:     logits_crane ← head_crane(z)
30:     logits_crane[~M_crane] ← -∞
31:     crane_idx ← Categorical(softmax(logits_crane)).sample()
32:
33:     M_lift_given_crane ← M_env.crane_dispatch[crane_idx, :]
34:     logits_lift ← head_lift(z)
35:     logits_lift[~M_lift_given_crane] ← -∞
36:     lift_idx ← Categorical(softmax(logits_lift)).sample()
37:
38:     a ← {type: 1, crane: crane_idx, lift: lift_idx}
39:     relevant_heads ← [action_type, crane, lift]
40:
41: else if action_type == 2 then  // Maintenance
42:     logits_equip ← head_equipment(z)
43:     logits_equip[~M_env.maintenance] ← -∞
44:     equip_idx ← Categorical(softmax(logits_equip)).sample()
45:
46:     a ← {type: 2, equipment: equip_idx}
47:     relevant_heads ← [action_type, equipment]
48:
49: else  // Hold
50:     a ← {type: 3}
51:     relevant_heads ← [action_type]
52: end if
53:
54: // Compute log probability (only relevant heads)
55: log_π(a) ← Σ_{head ∈ relevant_heads} log_prob(head, a)
56:
57: return a, log_π(a)
```

## Algorithm 4: Wiener Process Degradation

```
Algorithm 4: Equipment Health Degradation Step
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Current health H, Time step Δt, Load ratio L
       Operating flag, Parameters (μ_base, μ_load, σ, H_fail)

Output: New health H', Failure flag

1:  if not operating then
2:      return H, False           // No degradation when idle
3:  end if
4:
5:  // Load-dependent drift rate
6:  μ ← μ_base + μ_load × L
7:
8:  // Wiener process step
9:  Z ← sample_normal(0, 1)
10: ΔH ← -μ × Δt + σ × √Δt × Z
11:
12: // Update health (bounded)
13: H' ← max(0, H + ΔH)
14:
15: // Check for failure
16: failed ← (H' < H_fail)
17:
18: return H', failed
```

## Algorithm 5: Preventive Maintenance Decision

```
Algorithm 5: Maintenance Trigger Logic
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Equipment health vector H, PM threshold H_pm
       Equipment status, Maintenance restore level H_restore

Output: Maintenance mask M, Updated health (if triggered)

1:  // Determine which equipment qualifies for PM
2:  M ← zeros(n_equipment)
3:  for i = 1 to n_equipment do
4:      H_min ← min(H[i])           // Minimum component health
5:      if status[i] == IDLE and H_min < H_pm then
6:          M[i] ← True              // Can perform maintenance
7:      end if
8:  end for
9:
10: return M

---

Algorithm 5b: Perform Maintenance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Equipment index i, Health vector H[i]
       Maintenance restore level H_restore

Output: Updated health H'[i]

1:  for component in H[i] do
2:      component ← H_restore      // Restore to near-new condition
3:  end for
4:  status[i] ← IDLE              // Equipment available again
5:  return H[i]
```

## Algorithm 6: Precedence Constraint Checking

```
Algorithm 6: Check Predecessor Completion
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Block b, Set of placed blocks P

Output: Boolean (can block be placed on dock?)

1:  for pred_id in b.predecessors do
2:      if pred_id not in P then
3:          return False           // Predecessor not yet placed
4:      end if
5:      pred_block ← P[pred_id]
6:      if pred_block.status ≠ PLACED_ON_DOCK then
7:          return False           // Predecessor not on dock
8:      end if
9:  end for
10: return True                    // All predecessors satisfied
```

## Algorithm 7: Generalized Advantage Estimation (GAE)

```
Algorithm 7: Compute GAE Advantages
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Rewards r[1:T], Values V[1:T], γ, λ

Output: Advantages A[1:T]

1:  A ← zeros(T)
2:  last_gae ← 0
3:
4:  for t = T down to 1 do
5:      if t == T then
6:          next_value ← 0
7:      else
8:          next_value ← V[t+1]
9:      end if
10:
11:     // TD error
12:     δ_t ← r[t] + γ × next_value - V[t]
13:
14:     // GAE recursion
15:     A[t] ← δ_t + γ × λ × last_gae
16:     last_gae ← A[t]
17: end for
18:
19: return A
```

## Algorithm 8: Rule-Based Baseline (EDD + Nearest Vehicle)

```
Algorithm 8: Rule-Based Scheduling Decision
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input: Environment state, Transport requests R, Lift requests L

Output: Action a

1:  // Check for maintenance needs first
2:  for equipment in all_equipment do
3:      if equipment.status == IDLE and equipment.min_health < 40 then
4:          return {type: 2, equipment: equipment.idx}
5:      end if
6:  end for
7:
8:  // Try SPMT dispatch (EDD priority, nearest vehicle)
9:  if R not empty then
10:     // Sort requests by due date (earliest first)
11:     R_sorted ← sort(R, key=due_date)
12:     for request in R_sorted do
13:         block ← get_block(request.block_id)
14:         best_spmt ← None
15:         best_time ← ∞
16:
17:         for spmt in spmts do
18:             if spmt.status == IDLE and block.weight ≤ spmt.capacity then
19:                 travel_time ← get_travel_time(spmt.location, block.location)
20:                 if travel_time < best_time then
21:                     best_time ← travel_time
22:                     best_spmt ← spmt
23:                 end if
24:             end if
25:         end for
26:
27:         if best_spmt ≠ None then
28:             return {type: 0, spmt: best_spmt.idx, request: request.idx}
29:         end if
30:     end for
31: end if
32:
33: // Try crane dispatch
34: if L not empty then
35:     for request in L do
36:         block ← get_block(request.block_id)
37:         if predecessors_complete(block) then
38:             for crane in cranes do
39:                 if crane.status == IDLE and block.weight ≤ crane.capacity then
40:                     return {type: 1, crane: crane.idx, lift: request.idx}
41:                 end if
42:             end for
43:         end if
44:     end for
45: end if
46:
47: // Default: Hold
48: return {type: 3}
```

## Complexity Analysis

| Algorithm | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| GNN-PPO (per epoch) | O(T · (|E|d + |V|d²) + K · B · d²) | O(T · (|V| + |E|) + d²) |
| GNN Encoder | O(L · |E| · d + |V| · d²) | O(|V| · d + |E|) |
| Hierarchical Masking | O(n_spmts · n_requests + n_cranes · n_lifts) | O(n_equipment) |
| Degradation Step | O(n_components) | O(1) |
| GAE | O(T) | O(T) |
| Rule-Based | O(n_requests · n_spmts + n_lifts · n_cranes) | O(1) |

Where:
- T = trajectory length
- |V| = number of nodes
- |E| = number of edges
- d = hidden dimension
- L = number of GNN layers
- K = PPO epochs
- B = batch size
