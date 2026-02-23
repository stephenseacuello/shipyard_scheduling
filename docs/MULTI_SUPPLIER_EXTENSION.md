# Multi-Supplier Extension Design

## Overview

This document outlines the design for extending the shipyard scheduling framework to handle:
1. **Multiple suppliers** for blocks and components
2. **Resource allocation** for machines, materials, and labor

## Motivation

Real shipyards like HHI Ulsan source materials and components from multiple suppliers with varying:
- Lead times and reliability
- Quality levels and defect rates
- Pricing and capacity constraints
- Geographic locations (affecting transport time)

Current framework assumes all blocks are internally produced, which is unrealistic.

---

## 1. Multi-Supplier Model

### 1.1 Supplier Entity

```python
@dataclass
class Supplier:
    id: str
    name: str
    location: Tuple[float, float]  # Geo coordinates

    # Capacity
    daily_capacity: int  # Max items per day
    current_backlog: int

    # Performance
    lead_time_mean: float  # Days
    lead_time_std: float
    on_time_rate: float  # Historical reliability
    defect_rate: float  # Quality metric

    # Pricing
    base_price: float
    rush_multiplier: float  # For expedited orders

    # Specialization
    component_types: List[str]  # What they can supply
    certifications: List[str]  # Quality certifications
```

### 1.2 Component Types

```python
class ComponentType(Enum):
    STEEL_PLATE = "steel_plate"
    PIPE_SECTION = "pipe_section"
    ELECTRICAL_PANEL = "electrical_panel"
    HVAC_UNIT = "hvac_unit"
    ENGINE_COMPONENT = "engine_component"
    INSULATION = "insulation"
    TANK_MEMBRANE = "tank_membrane"  # LNG-specific
    CARGO_PUMP = "cargo_pump"
```

### 1.3 Procurement Decision

New action type for procurement:

```python
class ProcurementAction:
    component_type: ComponentType
    supplier_id: str
    quantity: int
    priority: str  # "normal", "rush", "critical"
    required_by: datetime
```

---

## 2. Resource Allocation Model

### 2.1 Machine Resources

```python
@dataclass
class MachineResource:
    id: str
    type: str  # "welder", "crane", "cutter", "painter"
    location: str  # Facility

    # Capacity
    max_hours_per_day: float
    scheduled_maintenance: List[Tuple[datetime, datetime]]

    # Health (existing Wiener model)
    health: float
    degradation_rate: float

    # Operator requirement
    requires_operator: bool
    operator_skill_level: str  # "basic", "certified", "expert"
```

### 2.2 Material Resources

```python
@dataclass
class MaterialInventory:
    material_type: str
    quantity: float
    unit: str  # "tons", "meters", "units"
    location: str  # Warehouse/facility

    # Tracking
    reorder_point: float
    economic_order_quantity: float
    lead_time: float

    # Quality
    batch_id: str
    expiry_date: Optional[datetime]
    quality_grade: str
```

### 2.3 Labor Resources

```python
@dataclass
class LaborPool:
    skill_type: str  # "welder", "electrician", "fitter", "painter"

    # Availability
    total_workers: int
    available_today: int
    shift_schedule: Dict[str, int]  # shift -> count

    # Constraints
    max_overtime_hours: float
    union_rules: Dict[str, Any]

    # Cost
    hourly_rate: float
    overtime_rate: float
```

---

## 3. Extended State Representation

### 3.1 New Graph Node Types

Add to existing 6 node types:

```python
# Supplier nodes
V_supplier = {
    "features": [
        "current_backlog",
        "lead_time_estimate",
        "reliability_score",
        "price_competitiveness",
        "geographic_distance"
    ]
}

# Inventory nodes
V_inventory = {
    "features": [
        "quantity_available",
        "days_until_stockout",
        "quality_grade",
        "location"
    ]
}

# Labor pool nodes
V_labor = {
    "features": [
        "available_workers",
        "skill_level_distribution",
        "overtime_capacity",
        "current_assignments"
    ]
}
```

### 3.2 New Edge Types

```python
# Supplier edges
(component, "sourced_from", supplier)
(supplier, "delivers_to", facility)

# Resource edges
(block, "requires_material", material)
(block, "requires_machine", machine)
(block, "requires_labor", labor_pool)

# Assignment edges
(worker, "assigned_to", task)
(machine, "allocated_to", block)
```

---

## 4. Extended Action Space

### 4.1 New Action Types

```python
class ExtendedActionType(Enum):
    # Existing
    DISPATCH_SPMT = 0
    DISPATCH_CRANE = 1
    MAINTENANCE = 2
    HOLD = 3

    # New: Procurement
    PLACE_ORDER = 4  # Order from supplier
    EXPEDITE_ORDER = 5  # Rush existing order

    # New: Resource allocation
    ASSIGN_WORKER = 6
    REASSIGN_MACHINE = 7
    REQUEST_OVERTIME = 8

    # New: Inventory
    TRANSFER_MATERIAL = 9
    EMERGENCY_PROCUREMENT = 10
```

### 4.2 Action Masking Extensions

```python
def extended_action_mask(state):
    mask = base_action_mask(state)

    # Procurement masking
    for supplier in state.suppliers:
        if supplier.current_backlog >= supplier.daily_capacity:
            mask["place_order"][supplier.id] = False

    # Labor masking
    for pool in state.labor_pools:
        if pool.available_today == 0:
            mask["assign_worker"][pool.skill_type] = False

    # Inventory masking
    for material in state.inventory:
        if material.quantity < minimum_order_quantity:
            mask["transfer_material"][material.id] = False

    return mask
```

---

## 5. Extended Reward Function

```python
def extended_reward(state, action, next_state):
    base_reward = original_reward(state, action, next_state)

    # Procurement costs
    procurement_cost = sum(
        order.quantity * order.unit_price * (1 + order.rush_multiplier)
        for order in new_orders
    )

    # Inventory holding cost
    holding_cost = sum(
        inv.quantity * inv.holding_cost_per_unit
        for inv in state.inventory
    )

    # Stockout penalty
    stockout_penalty = sum(
        STOCKOUT_COST * stockout_duration
        for stockout in stockouts
    )

    # Labor cost
    labor_cost = (
        regular_hours * regular_rate +
        overtime_hours * overtime_rate
    )

    # Supplier reliability bonus
    reliability_bonus = sum(
        RELIABILITY_WEIGHT * supplier.on_time_rate
        for supplier in used_suppliers
    )

    return (
        base_reward
        - w_proc * procurement_cost
        - w_hold * holding_cost
        - w_stock * stockout_penalty
        - w_labor * labor_cost
        + w_rel * reliability_bonus
    )
```

---

## 6. Implementation Phases

### Phase 1: Supplier Model (2 weeks)
- [ ] Add Supplier entity and database schema
- [ ] Implement procurement action type
- [ ] Add supplier nodes to GNN
- [ ] Basic supplier selection policy

### Phase 2: Inventory Management (2 weeks)
- [ ] Add MaterialInventory entity
- [ ] Implement inventory tracking
- [ ] Add reorder point logic
- [ ] Stockout detection and penalty

### Phase 3: Labor Allocation (2 weeks)
- [ ] Add LaborPool entity
- [ ] Implement worker assignment action
- [ ] Shift scheduling constraints
- [ ] Overtime logic

### Phase 4: Integration (2 weeks)
- [ ] Extended state representation
- [ ] Extended action masking
- [ ] Reward function integration
- [ ] End-to-end training

### Phase 5: Validation (1 week)
- [ ] Unit tests for new components
- [ ] Integration tests
- [ ] Performance benchmarking
- [ ] Ablation studies

---

## 7. Research Questions

1. **Supplier selection under uncertainty**: How does the agent learn to balance price, reliability, and lead time?

2. **Dynamic resource allocation**: Can the agent learn to reallocate resources in response to disruptions?

3. **Multi-objective extension**: How do we balance throughput vs. cost vs. reliability?

4. **Hierarchical decisions**: Should procurement be a separate hierarchical level?

5. **Transfer learning**: Does a policy trained with single supplier transfer to multi-supplier?

---

## 8. Expected Impact

| Metric | Current | With Extension | Improvement |
|--------|---------|----------------|-------------|
| Cost modeling | Basic | Full procurement + labor | More realistic |
| Disruption handling | Limited | Supplier backup, overtime | More robust |
| Decision scope | Operations | Strategic + Operations | Broader applicability |
| Industry relevance | High | Very High | Publication impact |

---

## 9. Publication Angle

**Title idea**: "End-to-End Supply Chain Optimization for Shipbuilding: Integrating Procurement, Scheduling, and Resource Allocation via Hierarchical Reinforcement Learning"

**Venues**:
- Manufacturing & Service Operations Management (MSOM)
- Production and Operations Management (POM)
- INFORMS Journal on Computing
- European Journal of Operational Research (EJOR)

**Key contributions**:
1. First integrated RL framework for shipyard supply chain
2. Novel multi-supplier action space with dynamic masking
3. Practical labor and machine allocation under constraints
4. Industrial validation on HHI Ulsan extended model
