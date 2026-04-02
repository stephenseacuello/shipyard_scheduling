#!/usr/bin/env python3
"""DAgger training for supply chain procurement (PLACE_ORDER, ASSIGN_WORKER).

Reuses DAggerTrainer from train_dagger.py with the supply chain config and
evaluates procurement metrics: stockout events, order fulfillment rate,
material holding cost alongside block scheduling throughput.

Usage:
    PYTHONPATH=src python experiments/train_dagger_supply_chain.py \
        --config config/hhi_supply_chain.yaml --iterations 5
"""
import argparse, os, sys, random
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from simulation.shipyard_env import HHIShipyardEnv
from agent.gnn_encoder import HeterogeneousGNNEncoder
from agent.policy import ActorCriticPolicy
from baselines.rule_based import RuleBasedScheduler
from experiments.train_dagger import DAggerTrainer, load_config

SC_KEYS = ("stockout_events", "holding_cost", "orders_placed",
           "deliveries_received", "procurement_cost", "labor_cost",
           "blocks_erected", "blocks_completed")


def evaluate_sc(env, actor, n_ep=5, max_steps=1000):
    """Run episodes, return averaged scheduling + supply chain metrics."""
    totals = {k: 0.0 for k in SC_KEYS}
    for _ in range(n_ep):
        env.reset()
        for _ in range(max_steps):
            _, _, t, tr, _ = env.step(actor(env))
            if t or tr:
                break
        for k in SC_KEYS:
            totals[k] += env.metrics.get(k, 0.0)
    avg = {k: v / n_ep for k, v in totals.items()}
    p, d = avg["orders_placed"], avg["deliveries_received"]
    avg["order_fulfillment_rate"] = d / p if p > 0 else 0.0
    return avg


def print_metrics(label, m):
    blk = m.get("blocks_erected", m.get("blocks_completed", 0))
    print(f"  [{label}]  stockouts={m['stockout_events']:.0f}  "
          f"fulfillment={m['order_fulfillment_rate']:.1%}  "
          f"holding={m['holding_cost']:.1f}  blocks={blk:.0f}")


def make_policy_actor(encoder, policy, device):
    def actor(env):
        with torch.no_grad():
            emb = encoder(env.get_graph_data().to(device))
            act, _, _ = policy.get_action(emb, deterministic=True)
        return {k: int(v.item()) for k, v in act.items()}
    return actor


def main():
    a = argparse.ArgumentParser()
    aa = a.add_argument
    aa("--config", default="config/hhi_supply_chain.yaml")
    aa("--iterations", type=int, default=5);  aa("--init-episodes", type=int, default=15)
    aa("--dagger-episodes", type=int, default=8); aa("--train-epochs", type=int, default=20)
    aa("--max-steps", type=int, default=1000)
    aa("--beta-start", type=float, default=1.0); aa("--beta-end", type=float, default=0.1)
    aa("--lr", type=float, default=3e-4); aa("--device", default="cpu")
    aa("--save", default="data/checkpoints/dagger_supply_chain/")
    aa("--seed", type=int, default=42); aa("--eval-episodes", type=int, default=5)
    P = a.parse_args()

    random.seed(P.seed); np.random.seed(P.seed); torch.manual_seed(P.seed)
    cfg = load_config(P.config)
    sc = cfg.get("supply_chain", {})
    assert sc.get("enable_suppliers"), "Config must enable supply_chain.enable_suppliers"
    assert sc.get("enable_inventory"), "Config must enable supply_chain.enable_inventory"
    env = HHIShipyardEnv(cfg)

    H = 128
    sd = env.supplier_features if env.n_suppliers > 0 else 0
    id_ = env.inventory_features if env.n_inventory_nodes > 0 else 0
    ld = env.labor_features if env.n_labor_pools > 0 else 0
    encoder = HeterogeneousGNNEncoder(
        block_dim=env.block_features, spmt_dim=env.spmt_features,
        crane_dim=env.crane_features, facility_dim=env.facility_features,
        hidden_dim=H, num_layers=2, supplier_dim=sd, inventory_dim=id_, labor_dim=ld)
    n_types = 4 + (sd > 0) + (id_ > 0) + (ld > 0)
    policy = ActorCriticPolicy(
        state_dim=H * n_types, n_action_types=env.n_action_types, n_spmts=env.n_spmts,
        n_cranes=getattr(env, "n_goliath_cranes", getattr(env, "n_cranes", 2)),
        max_requests=env.n_blocks, hidden_dim=256, epsilon=0.0,
        n_suppliers=env.n_suppliers, n_inventory=env.n_inventory_nodes,
        n_labor_pools=env.n_labor_pools)
    expert = RuleBasedScheduler()
    trainer = DAggerTrainer(env=env, encoder=encoder, policy=policy,
                            expert=expert, device=P.device, lr=P.lr)

    print("=" * 60 + "\nDAgger Training -- Supply Chain Procurement")
    print(f"  suppliers={env.n_suppliers}  inventory={env.n_inventory_nodes}  "
          f"labor={env.n_labor_pools}  iters={P.iterations}\n" + "=" * 60)

    # Phase 1: expert demonstrations + initial BC
    trainer.collect_expert_demos(P.init_episodes, max_steps=P.max_steps)
    for ep in range(P.train_epochs):
        loss = trainer.train_epoch()
        if (ep + 1) % 5 == 0: print(f"  BC epoch {ep+1}/{P.train_epochs}  loss={loss:.4f}")

    # Phase 2: DAgger iterations -- track best by stockout count
    best_stockouts, best_state = float("inf"), None
    actor_fn = make_policy_actor(encoder, policy, P.device)
    for it in range(P.iterations):
        beta = P.beta_start - (P.beta_start - P.beta_end) * it / max(P.iterations - 1, 1)
        print(f"\n--- Iteration {it+1}/{P.iterations}  beta={beta:.2f} ---")
        trainer.collect_dagger_data(P.dagger_episodes, beta=beta, max_steps=P.max_steps)
        for _ in range(P.train_epochs):
            loss = trainer.train_epoch()
        m = evaluate_sc(env, actor_fn, n_ep=3, max_steps=P.max_steps)
        print(f"  loss={loss:.4f}"); print_metrics("policy", m)
        if m["stockout_events"] < best_stockouts:
            best_stockouts = m["stockout_events"]
            best_state = {"encoder": encoder.state_dict(), "policy": policy.state_dict()}
            print("  *** new best ***")

    # Restore best and run final comparison
    if best_state:
        encoder.load_state_dict(best_state["encoder"])
        policy.load_state_dict(best_state["policy"])
    pm = evaluate_sc(env, actor_fn, P.eval_episodes, P.max_steps)
    em = evaluate_sc(env, lambda e: expert.decide(e), P.eval_episodes, P.max_steps)

    print("\n" + "=" * 60 + "\nFinal Results\n" + "=" * 60)
    print_metrics("DAgger Policy", pm); print_metrics("Rule-Based Expert", em)
    es, ps = em["stockout_events"], pm["stockout_events"]
    print(f"\n  Stockout reduction: {es:.0f} -> {ps:.0f} ({100*(1-ps/es) if es else 0:.1f}%)")
    print(f"  Fulfillment:  expert={em['order_fulfillment_rate']:.1%}  "
          f"dagger={pm['order_fulfillment_rate']:.1%}")
    print(f"  Holding cost: expert={em['holding_cost']:.1f}  dagger={pm['holding_cost']:.1f}")

    os.makedirs(P.save, exist_ok=True)
    ckpt = {"policy_metrics": pm, "expert_metrics": em, "args": vars(P)}
    if best_state: ckpt.update(best_state)
    torch.save(ckpt, os.path.join(P.save, "dagger_supply_chain.pt"))
    print(f"\nSaved to {P.save}")


if __name__ == "__main__":
    main()
