# Scheduling Benchmark Instances

Standard benchmark instances for multi-domain entropy-collapse validation.
These replace the randomly generated instances in `experiments/entropy_collapse_domains.py`
with deterministic, published benchmarks so that results are reproducible and
comparable to the operations-research literature.

## Included Benchmarks

### JSSP -- Taillard / Fisher-Thompson Instances

| Instance | Size | Optimal Makespan | Reference |
|----------|------|-------------------|-----------|
| ft06 | 6 jobs x 6 machines | 55 | Fisher & Thompson, 1963 |
| ft10 | 10 jobs x 10 machines | 930 | Fisher & Thompson, 1963 |

**Module:** `taillard_jssp.py`

```python
from data.benchmarks.taillard_jssp import get_instance
inst = get_instance("ft06")
# inst["processing_times"]  -- ndarray (6, 6)
# inst["machine_order"]     -- ndarray (6, 6)
```

**Source:** http://mistic.heig-vd.ch/taillard/problemes.dir/ordonnancement.dir/jobshop.dir/

**Original reference:** J.F. Muth and G.L. Thompson, *Industrial Scheduling*,
Prentice Hall, 1963.

### VRPTW -- Solomon Instances

| Instance | Customers | Capacity | Distribution | Time Windows |
|----------|-----------|----------|--------------|--------------|
| R101_25 | 25 | 200 | Random | Tight |
| R101_100 | 100 | 200 | Random | Tight |

**Module:** `solomon_vrptw.py`

```python
from data.benchmarks.solomon_vrptw import get_instance
inst = get_instance("r101_25")
# inst["depot"]      -- [x, y]
# inst["customers"]  -- [[x, y, demand, ready, due, service], ...]
```

**Source:** https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/

**Original reference:** M.M. Solomon, "Algorithms for the Vehicle Routing and
Scheduling Problems with Time Window Constraints," *Operations Research*,
35(2), 1987, pp. 254--265.

## Usage with Entropy Collapse Experiment

These instances can be loaded into `JobShopEnv` and `VRPTWEnv` from
`experiments/entropy_collapse_domains.py` to replace the random instance
generation, ensuring deterministic, literature-standard problem data:

```python
from data.benchmarks.taillard_jssp import get_instance as get_jssp
from data.benchmarks.solomon_vrptw import get_instance as get_vrptw

jssp = get_jssp("ft10")
vrptw = get_vrptw("r101_25")
```

## Data Format Details

### JSSP

- `machine_order[j, k]` = machine that job `j` visits at its `k`-th operation
- `processing_times[j, k]` = time job `j` spends at its `k`-th operation
- Row `j` of each array describes the full route for job `j`

### VRPTW

- `depot` = `[x, y]` coordinates of the depot
- `customers[i]` = `[x, y, demand, ready_time, due_date, service_time]`
- Vehicles start and end at the depot
- A vehicle must arrive at customer `i` no later than `due_date`
- If it arrives before `ready_time`, it waits
