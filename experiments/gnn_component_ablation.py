#!/usr/bin/env python3
"""GNN Component Ablation Study for Shipyard Scheduling.

Tests 9 architecture variants under DAgger training to isolate each component's
contribution (heterogeneous typing, GAT attention, GNN message passing,
readiness gate, attention action heads, depth, width).

Usage:
    PYTHONPATH=src python experiments/gnn_component_ablation.py \
        --config config/small_instance.yaml --seeds 3
"""
import argparse, csv, os, random, sys, time
from collections import defaultdict
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, yaml
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from torch.distributions import Categorical

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
from simulation.shipyard_env import HHIShipyardEnv
from agent.gnn_encoder import HeterogeneousGNNEncoder
from agent.policy import ActorCriticPolicy, _apply_mask
from baselines.rule_based import RuleBasedScheduler

def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg.get("inherit_from"):
        base = load_config(os.path.join(os.path.dirname(path), cfg["inherit_from"]))
        base.update({k: v for k, v in cfg.items() if k != "inherit_from"})
        return base
    return cfg

# ── Ablation encoders ──────────────────────────────────────────────────
_BASE_NT = ["block", "spmt", "crane", "facility"]
_DIM_KEYS = {"block": "block_dim", "spmt": "spmt_dim", "crane": "crane_dim",
             "facility": "facility_dim", "supplier": "supplier_dim",
             "inventory": "inventory_dim", "labor": "labor_dim"}

def _node_types(sd, ind, ld):
    nt = list(_BASE_NT)
    if sd > 0: nt.append("supplier")
    if ind > 0: nt.append("inventory")
    if ld > 0: nt.append("labor")
    return nt

class _PoolEncoder(nn.Module):
    """Base class for ablation encoders that project + pool per node type."""
    def __init__(self, dims, hidden_dim, node_types):
        super().__init__()
        self.node_types, self.hd = node_types, hidden_dim
        self.projs = nn.ModuleDict({nt: nn.Linear(dims[nt], hidden_dim) for nt in node_types})
        self.output_norm = nn.LayerNorm(hidden_dim * len(node_types))

    def _pool(self, x_dict, data):
        pooled = []
        dev = next(self.parameters()).device
        for nt in self.node_types:
            if nt in x_dict and nt in data.node_types and data[nt].x.shape[0] > 0:
                pooled.append(global_mean_pool(x_dict[nt], data[nt].batch))
            else:
                pooled.append(torch.zeros(1, self.hd, device=dev))
        return self.output_norm(torch.cat(pooled, dim=-1))

class MLPEncoder(_PoolEncoder):
    """No-GNN: per-type 2-layer MLP, then mean-pool."""
    def __init__(self, dims, hidden_dim, node_types):
        super().__init__(dims, hidden_dim, node_types)
        self.mlps = nn.ModuleDict({
            nt: nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            for nt in node_types})
    def forward(self, data):
        x = {nt: F.relu(self.projs[nt](data[nt].x)) for nt in self.node_types
             if nt in data.node_types and data[nt].x.shape[0] > 0}
        x = {nt: self.mlps[nt](x[nt]) for nt in x}
        return self._pool(x, data)

class HomogeneousEncoder(_PoolEncoder):
    """No-heterogeneity: shared MLP layers across all types (no HeteroConv)."""
    def __init__(self, dims, hidden_dim, node_types, num_layers=2):
        super().__init__(dims, hidden_dim, node_types)
        layers = []
        for _ in range(num_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)]
        self.shared = nn.Sequential(*layers)
    def forward(self, data):
        x = {nt: self.shared(F.relu(self.projs[nt](data[nt].x)))
             for nt in self.node_types if nt in data.node_types and data[nt].x.shape[0] > 0}
        return self._pool(x, data)

class NoAttnEncoder(_PoolEncoder):
    """No-attention: per-node Linear+ReLU with residual (no message passing)."""
    def __init__(self, dims, hidden_dim, node_types, num_layers=2):
        super().__init__(dims, hidden_dim, node_types)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
    def forward(self, data):
        x = {nt: F.relu(self.projs[nt](data[nt].x)) for nt in self.node_types
             if nt in data.node_types and data[nt].x.shape[0] > 0}
        for layer, norm in zip(self.layers, self.norms):
            x = {nt: norm(F.relu(layer(x[nt])) + x[nt]) for nt in x}
        return self._pool(x, data)

# ── Variant definitions ────────────────────────────────────────────────
VARIANTS = {
    "full": {}, "no_hetero": {"no_hetero": True}, "no_attention": {"no_attention": True},
    "no_gnn": {"no_gnn": True}, "no_readiness_gate": {"no_readiness_gate": True},
    "no_attn_heads": {"no_attn_heads": True}, "depth_4": {"num_layers": 4},
    "width_256": {"hidden_dim": 256}, "width_64": {"hidden_dim": 64},
}

def _env_dims(env):
    sd = env.supplier_features if env.n_suppliers > 0 else 0
    ind = env.inventory_features if env.n_inventory_nodes > 0 else 0
    ld = env.labor_features if env.n_labor_pools > 0 else 0
    raw = dict(block_dim=env.block_features, spmt_dim=env.spmt_features,
               crane_dim=env.crane_features, facility_dim=env.facility_features,
               supplier_dim=sd, inventory_dim=ind, labor_dim=ld)
    nt = _node_types(sd, ind, ld)
    dims = {k.replace("_dim", ""): v for k, v in raw.items()}
    return raw, dims, nt

def create_variant(env, vcfg):
    raw, dims, nt = _env_dims(env)
    hd = vcfg.get("hidden_dim", 128)
    nl = vcfg.get("num_layers", 2)
    if vcfg.get("no_gnn"):
        encoder = MLPEncoder(dims, hd, nt)
    elif vcfg.get("no_hetero"):
        encoder = HomogeneousEncoder(dims, hd, nt, nl)
    elif vcfg.get("no_attention"):
        encoder = NoAttnEncoder(dims, hd, nt, nl)
    else:
        encoder = HeterogeneousGNNEncoder(**raw, hidden_dim=hd, num_layers=nl)
    state_dim = hd * len(nt)
    policy = ActorCriticPolicy(
        state_dim=state_dim, n_action_types=env.n_action_types, n_spmts=env.n_spmts,
        n_cranes=getattr(env, "n_goliath_cranes", getattr(env, "n_cranes", 2)),
        max_requests=env.n_blocks, hidden_dim=256, epsilon=0.0,
        n_suppliers=env.n_suppliers, n_inventory=env.n_inventory_nodes,
        n_labor_pools=env.n_labor_pools)
    # Policy-level ablations
    if vcfg.get("no_readiness_gate"):
        def _fwd_no_gate(state, mask=None, entity_embeddings=None):
            feat = policy.shared(state)
            policy._last_readiness_prob = torch.tensor(0.5)
            heads = ["action_type", "spmt", "request", "crane", "lift",
                     "equipment", "supplier", "material", "labor_pool", "target_block"]
            lg = {h: getattr(policy, f"{h}_head")(feat) for h in heads}
            if mask:
                lg = {k: _apply_mask(v, mask.get(k)) if mask.get(k) is not None else v for k, v in lg.items()}
            if policy.temperature != 1.0:
                lg = {k: v / policy.temperature for k, v in lg.items()}
            return {k: Categorical(logits=v) for k, v in lg.items()}, policy.critic(feat)
        policy.forward = _fwd_no_gate
    if vcfg.get("no_attn_heads"):
        _orig = policy.forward
        policy.forward = lambda s, m=None, e=None, _o=_orig: _o(s, m, entity_embeddings=None)
    return encoder, policy

# ── Compact DAgger loop ────────────────────────────────────────────────
AK = {"action_type": "action_type", "spmt": "spmt_idx", "request": "request_idx",
      "crane": "crane_idx", "lift": "lift_idx", "equipment": "equipment_idx",
      "supplier": "supplier_idx", "material": "material_idx",
      "labor_pool": "labor_pool_idx", "target_block": "target_block_idx"}

def run_dagger(env, enc, pol, expert, n_iters=5, n_epochs=10,
               init_eps=5, dag_eps=3, max_steps=500, lr=3e-4):
    enc.to("cpu"); pol.to("cpu")
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pol.parameters()), lr=lr, weight_decay=1e-5)
    gl, ea = [], []
    def collect(n, beta=1.0):
        for _ in range(n):
            env.reset()
            for _ in range(max_steps):
                gd = env.get_graph_data().cpu(); act = expert.decide(env)
                gl.append(gd); ea.append(act)
                if random.random() >= beta:
                    with torch.no_grad():
                        a, _, _ = pol.get_action(enc(gd))
                    act = {k: int(v.item()) for k, v in a.items()}
                _, _, t, tr, _ = env.step(act)
                if t or tr: break
    def train():
        idx = list(range(len(gl))); random.shuffle(idx)
        for i in range(0, len(gl), 64):
            bi = idx[i:i+64]
            if len(bi) < 2: continue
            bg = Batch.from_data_list([gl[j] for j in bi])
            ad, _ = pol.forward(enc(bg))
            loss = sum((2.0 if h == "action_type" else 1.0) * F.cross_entropy(
                ad[h].logits, torch.tensor([ea[j].get(a, 0) for j in bi]).clamp(0, ad[h].probs.shape[-1]-1))
                for h, a in AK.items())
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(pol.parameters()) + list(enc.parameters()), 1.0)
            opt.step()
    collect(init_eps)
    for _ in range(n_epochs): train()
    for it in range(n_iters):
        collect(dag_eps, beta=1.0 - 0.9 * it / max(n_iters - 1, 1))
        for _ in range(n_epochs): train()
    # Evaluate
    tps = []
    for _ in range(5):
        env.reset()
        for _ in range(max_steps):
            with torch.no_grad():
                a, _, _ = pol.get_action(enc(env.get_graph_data()), deterministic=True)
            _, _, t, tr, _ = env.step({k: int(v.item()) for k, v in a.items()})
            if t or tr: break
        if env.sim_time > 0:
            tps.append(env.metrics.get("blocks_erected", env.metrics.get("blocks_completed", 0)) / env.sim_time)
    return float(np.mean(tps)) if tps else 0.0

def measure_expert(env, expert, n=10, ms=500):
    tps = []
    for _ in range(n):
        env.reset()
        for _ in range(ms):
            _, _, t, tr, _ = env.step(expert.decide(env))
            if t or tr: break
        if env.sim_time > 0:
            tps.append(env.metrics.get("blocks_erected", env.metrics.get("blocks_completed", 0)) / env.sim_time)
    return float(np.mean(tps)) if tps else 0.0

def main():
    ap = argparse.ArgumentParser(description="GNN Component Ablation")
    ap.add_argument("--config", default="config/small_instance.yaml")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--dagger-iters", type=int, default=5)
    ap.add_argument("--train-epochs", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--output", default="data/ablation_results.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    env = HHIShipyardEnv(cfg)
    expert = RuleBasedScheduler()
    print("Measuring expert baseline ...")
    expert_tp = measure_expert(env, expert, ms=args.max_steps)
    print(f"Expert throughput: {expert_tp:.6f}\n")
    results = []
    for vname, vcfg in VARIANTS.items():
        for seed in range(args.seeds):
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            enc, pol = create_variant(env, vcfg)
            print(f"[{vname}] seed={seed} ...", end=" ", flush=True)
            t0 = time.time()
            tp = run_dagger(env, enc, pol, expert, n_iters=args.dagger_iters,
                            n_epochs=args.train_epochs, max_steps=args.max_steps)
            elapsed = time.time() - t0
            pct = 100.0 * tp / expert_tp if expert_tp > 0 else 0.0
            print(f"throughput={tp:.6f}  vs_expert={pct:.1f}%  time={elapsed:.1f}s")
            results.append(dict(variant=vname, seed=seed, throughput=round(tp, 6),
                                vs_expert_pct=round(pct, 1), training_time=round(elapsed, 1)))
    # Save CSV
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant","seed","throughput","vs_expert_pct","training_time"])
        w.writeheader(); w.writerows(results)
    print(f"\nResults saved to {args.output}")
    # Summary
    print("\n" + "=" * 65)
    print(f"{'Variant':<20} {'Mean %Expert':>12} {'Std':>8} {'Time(s)':>8}")
    print("-" * 65)
    by_v, by_t = defaultdict(list), defaultdict(list)
    for r in results:
        by_v[r["variant"]].append(r["vs_expert_pct"])
        by_t[r["variant"]].append(r["training_time"])
    for v in VARIANTS:
        print(f"{v:<20} {np.mean(by_v[v]):>11.1f}% {np.std(by_v[v]):>7.1f} {np.mean(by_t[v]):>8.1f}")
    print("=" * 65)

if __name__ == "__main__":
    main()
