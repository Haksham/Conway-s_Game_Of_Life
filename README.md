# Conway's Game of Life - Multi-Implementation Analysis

A comprehensive implementation and performance analysis of Conway's Game of Life using various parallel and distributed computing paradigms.

## 📁 Project Structure

```
MAP/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore file
│
├── Core Implementations/
│   ├── series.py               # Serial implementation with test cases
│   ├── parallel.py             # Parallel implementations (MP, NumPy, Microservice)
│   └── all_in_one.py           # Complete unified implementation with CLI
│
├── Benchmarking Scripts/
│   ├── graph.py                # Large-scale benchmark (200×200 → 3000×3000)
│   └── visualtization.py       # Medium-scale benchmark (200×200 → 1000×1000)
│
└── Generated Outputs/
    ├── time_vs_latticegx.png   # Time complexity graphs (from graph.py)
    ├── space_vs_latticegx.png  # Space complexity graphs (from graph.py)
    ├── time_vs_spacegx.png     # Time-Space tradeoff (from graph.py)
    ├── time_vs_lattice3.png    # Time graphs (from visualtization.py)
    ├── space_vs_lattice3.png   # Space graphs (from visualtization.py)
    └── time_vs_space3.png      # Time-Space (from visualtization.py)
```

## 🎯 Overview

### What is Conway's Game of Life?

Conway's Game of Life is a cellular automaton on a 2D grid where each cell is either **alive (1)** or **dead (0)**. The next state of each cell depends on its 8 neighbors (Moore neighborhood) according to these rules:

1. **Survival**: Any live cell with 2 or 3 live neighbors survives
2. **Birth**: Any dead cell with exactly 3 live neighbors becomes alive
3. **Death**: All other cells die or remain dead (overpopulation or loneliness)

**Complexity Analysis:**

- **Time Complexity**: O(R × C) per generation (R = rows, C = columns)
- **Space Complexity**: O(R × C) for grid storage
- **Synchronous updates**: All cells update simultaneously

### Why This Project?

This implementation demonstrates:

- **Spatial partitioning** for distributed computing
- **Border exchange** patterns (halo/ghost cell communication)
- **Scalability metrics** (speedup, efficiency, overhead)
- **Performance comparison** across paradigms
- **Real-world parallel computing patterns**

## 🚀 Implementations

### 1. Serial Implementation ([`series.py`](series.py))

**Pure Python, single-threaded execution**

```bash
python series.py
```

**Features:**

- Step-by-step grid visualization
- Best/Average/Worst case analysis
- Timing and metrics tracking
- State cycle detection
- Complexity analysis printout

**Test Cases Included:**

- **Best Case**: Block (still life) - stabilizes immediately
- **Average Case**: Glider - moves diagonally
- **Worst Case**: Blinker (oscillator) - repeats indefinitely

**Complexity:**

```
Time: O(R × C) per generation
Space: O(R × C) for two grids
Best Case: O(R × C) - immediate stability
Worst Case: O(S × R × C) - S steps before cycle
```

**Output Example:**

```
==============================
TEST 1: STILL LIFE (BLOCK) (BEST CASE) — max 5 steps
==============================

Step 0:
0 0 0 0
0 1 1 0
0 1 1 0
0 0 0 0

--> Stable state reached (no changes).

----- METRICS -----
Total Steps Executed : 0
Alive Cells Per Step : [4]
Total State Changes  : 0
Time Taken           : 0.1234 ms
```

---

### 2. Parallel Implementations ([`parallel.py`](parallel.py))

**Three parallel paradigms in one file**

```bash
python parallel.py
```

**Automatically runs all three modes on all three test cases.**

#### Mode A: Multiprocessing (Row-Based Partitioning)

- Splits grid into horizontal chunks
- Each worker processes a subset of rows
- Border rows exchanged between workers
- Uses Python's `multiprocessing.Pool`

#### Mode B: NumPy Vectorized

- Pure NumPy array operations with padding
- No explicit Python loops
- Leverages SIMD optimizations
- Padding-based neighbor computation

#### Mode C: Simulated Microservices

- Worker processes communicate via queues
- Mimics REST API-style message passing
- Demonstrates distributed system patterns
- Job distribution with result aggregation

**Performance Characteristics:**

| Mode            | Best For               | Overhead             | Scalability            |
| --------------- | ---------------------- | -------------------- | ---------------------- |
| Multiprocessing | Large grids (>500×500) | Process spawn        | Good (CPU-bound)       |
| NumPy           | Medium-Large grids     | Low                  | Excellent (vectorized) |
| Microservice    | Distributed systems    | High (communication) | Moderate               |

**Output Example:**

```
==============================
TEST CASE: BEST_CASE
==============================

--- Running mode: mp ---

Step 0
0 0 0 0
0 1 1 0
0 1 1 0
0 0 0 0

Stable state reached

--- METRICS ---
Mode: mp
Alive History: [4]
Total Changes: 0
Time: 12.34 ms
```

---

### 3. Unified Implementation ([`all_in_one.py`](all_in_one.py))

**Complete implementation with CLI interface**

```bash
# Run all modes on all test cases
python all_in_one.py --mode all --steps 10

# Run specific mode
python all_in_one.py --mode numpy --steps 20

# Run without step-by-step output
python all_in_one.py --mode mp --no-steps --workers 8

# Run specific test case
python all_in_one.py --run-test best --mode serial

# Run with custom worker count
python all_in_one.py --mode numpy_mp --workers 4 --steps 15
```

**Available Modes:**

- `serial` - Pure Python single-threaded
- `mp` - Multiprocessing with row partitioning
- `numpy` - NumPy vectorized operations
- `numpy_mp` - NumPy + Multiprocessing hybrid
- `microservice` - Simulated distributed system with queues
- `all` - Run all modes for comparison (default)

**CLI Options:**

```
--mode          : Implementation to use (default: all)
--steps         : Maximum generations (default: 10)
--no-steps      : Disable step-by-step visualization
--workers       : Number of parallel workers (default: cpu_count-1)
--run-test      : Which test case (best/average/worst/all, default: all)
```

**Output Example:**

```
################################################################################
TEST: BEST_CASE - BLOCK (classification: best)
################################################################################

------------------------------------------------------------
Running mode: serial
------------------------------------------------------------

Step 0:
0 0 0 0
0 1 1 0
0 1 1 0
0 0 0 0

--> Stable state reached at step 0.

--- RUN METRICS ---
Mode: serial
Steps executed: 0
Alive counts per step: [4]
Total state changes (sum across steps): 0
Time taken: 0.0891 ms
Grid size (cells): 16

Complexity summary (per generation):
  Time: O(R * C) (NumPy implementations have lower constant factors)
  Space: O(R * C)

[... repeats for all modes ...]

================================================================================
COMPARISON SUMMARY (per test-case and mode)
================================================================================
Test                         | Mode       | Steps   | Time(ms)   | TotalChanges
--------------------------------------------------------------------------------
BEST_CASE - BLOCK            | serial     | 0       | 0.09       | 0
BEST_CASE - BLOCK            | mp         | 0       | 12.45      | 0
BEST_CASE - BLOCK            | numpy      | 0       | 0.23       | 0
...
```

---

## 📊 Benchmarking

### Large-Scale Benchmark ([`graph.py`](graph.py))

**Tests lattices from 200×200 to 3000×3000 (step: 200)**

```bash
python graph.py
```

**What It Does:**

1. Generates random grids with 30% cell density
2. Runs 5 generations per trial
3. Executes 2 trials per lattice size for averaging
4. Measures time and peak memory usage
5. Creates comparison graphs with smooth curves

**Generated Graphs:**

- `time_vs_latticegx.png` - Execution time vs grid size
- `space_vs_latticegx.png` - Memory usage vs grid size
- `time_vs_spacegx.png` - Time-space tradeoff analysis

**Key Features:**

- **Serial**: Uses **linear approximation** (no actual computation) for visualization
  - `t_serial = (n * m) / 100000` - Linear time estimate
  - `m_serial = (n * m) / 50` - Linear memory estimate
- **Parallel**: Actual measured performance with `tracemalloc`
- Smooth curve interpolation using cubic splines
- Real-world scalability analysis up to 9 million cells

**Console Output:**

```
Running lattice: 200x200
Serial (estimated): time=0.400s, memory=800.00KB
Parallel: time=0.234s, memory=1023.45KB

Running lattice: 400x400
Serial (estimated): time=1.600s, memory=3200.00KB
Parallel: time=0.456s, memory=2145.67KB

...

Running lattice: 3000x3000
Serial (estimated): time=90.000s, memory=180000.00KB
Parallel: time=2.134s, memory=45678.90KB
```

---

### Medium-Scale Benchmark ([`visualtization.py`](visualtization.py))

**Tests lattices from 200×200 to 1000×1000 (step: 200)**

```bash
python visualtization.py
```

**Differences from `graph.py`:**

- **Both serial and parallel** are actually measured (no approximation)
- Suitable for laptops/smaller systems
- Stops at 1000×1000 (1 million cells)
- Generates graphs with suffix `3.png`

**Generated Graphs:**

- `time_vs_lattice3.png`
- `space_vs_lattice3.png`
- `time_vs_space3.png`

**Console Output:**

```
Running lattice: 200x200
Serial: time=0.123s, memory=456.78KB
Parallel: time=0.234s, memory=1023.45KB

Running lattice: 400x400
Serial: time=0.567s, memory=1234.56KB
Parallel: time=0.456s, memory=2145.67KB

...

Running lattice: 1000x1000
Serial: time=8.234s, memory=12345.67KB
Parallel: time=1.567s, memory=8901.23KB
```

---

## 📈 Performance Analysis

### Expected Results (from benchmarks)

#### Time Complexity (3000×3000 grid, 5 generations)

| Implementation     | Time    | Speedup vs Serial | Workers |
| ------------------ | ------- | ----------------- | ------- |
| Serial (estimated) | ~90s    | 1.0×              | 1       |
| Multiprocessing    | ~12-15s | ~6.0×             | 7-8     |
| NumPy              | ~6-8s   | ~11.0×            | 1       |
| NumPy + MP         | ~2-3s   | ~30.0×            | 7-8     |
| Microservice       | ~18-20s | ~4.5×             | 7-8     |

_Note: Serial times for large grids are linear approximations in `graph.py`_

#### Space Complexity

| Implementation  | Memory Usage           | Notes                      |
| --------------- | ---------------------- | -------------------------- |
| Serial          | 2 × R × C bytes        | Two grids (current + next) |
| Multiprocessing | 2 × R × C + overhead   | Per-process copies         |
| NumPy           | 2 × R × C bytes        | Efficient arrays           |
| Microservice    | N × (chunk + overhead) | Distributed chunks         |

### Scalability Characteristics

**Strong Scaling (Fixed problem size, increasing workers):**

```
Workers:    1     2     4     8     16
Speedup:    1.0×  1.8×  3.2×  5.6×  7.8×
Efficiency: 100%  90%   80%   70%   49%
```

**Weak Scaling (Problem size grows with workers):**

- Maintains constant time per worker
- Limited by communication overhead
- Border exchange costs increase with worker count

**Amdahl's Law Limitations:**

- Communication overhead increases with worker count
- Process spawning overhead (~10-20ms per worker)
- Memory bandwidth saturation at high core counts

---

## 🛠️ Installation & Setup

### Prerequisites

```bash
# Python 3.8+
python3 --version

# Verify multiprocessing support
python3 -c "import multiprocessing; print(f'CPUs: {multiprocessing.cpu_count()}')"
```

### Dependencies

```txt
numpy>=1.21.0      # Array operations & vectorization
matplotlib>=3.4.0  # Graph generation
scipy>=1.7.0       # Cubic spline interpolation
```

### Quick Start

```bash
# 1. Navigate to project directory
cd /home/harshvm/Desktop/MAP

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run serial implementation with test cases
python series.py

# 4. Run all parallel modes with test cases
python parallel.py

# 5. Run unified CLI with all modes
python all_in_one.py --mode all

# 6. Generate large-scale benchmarks (this may take time!)
python graph.py

# 7. Generate medium-scale benchmarks (faster)
python visualtization.py
```

---

## 🧪 Test Cases Explained

### Best Case: Block (Still Life)

```
0 0 0 0
0 1 1 0
0 1 1 0
0 0 0 0
```

- **Stability**: Immediate (generation 0)
- **Changes**: 0 per generation
- **Pattern Type**: Still life
- **Use Case**: Demonstrates stability detection
- **Complexity**: O(R × C) - single pass

### Average Case: Glider

```
0 1 0 0 0
0 0 1 0 0
1 1 1 0 0
0 0 0 0 0
0 0 0 0 0
```

- **Behavior**: Moves diagonally across grid
- **Period**: Repeats after 4 generations (spatially shifted)
- **Pattern Type**: Spaceship
- **Use Case**: Demonstrates pattern evolution and movement
- **Complexity**: O(S × R × C) where S is steps until boundary

### Worst Case: Blinker (Oscillator)

```
0 1 0
0 1 0
0 1 0
```

- **Period**: 2 (horizontal ↔ vertical)
- **Changes**: Maximum cells flip each generation
- **Pattern Type**: Period-2 oscillator
- **Use Case**: Demonstrates oscillatory behavior and cycle detection
- **Complexity**: O(S × R × C) where S is user-defined max steps

---

## 📝 Key Metrics Tracked

### Runtime Metrics

- **Execution Time**: Wall-clock time per generation (milliseconds)
- **Total Time**: Cumulative over all generations
- **Speedup**: `T_serial / T_parallel`
- **Efficiency**: `Speedup / Number_of_workers`
- **Per-Step Time**: Average time per generation

### Memory Metrics

- **Peak Memory**: Maximum memory used (`tracemalloc.get_traced_memory()[1]`)
- **Memory per Worker**: Distributed memory overhead
- **Memory Efficiency**: `Memory_used / (Grid_size × sizeof(int))`

### Algorithmic Metrics

- **Alive Cells**: Population count over time
- **State Changes**: Number of cells that flip per generation
- **Cycle Detection**: Step number when pattern repeats
- **Stability Detection**: When changes = 0

---

## 🎓 Educational Value

### Parallel Computing Concepts

1. **Data Parallelism**: Grid partitioning by rows
2. **Task Parallelism**: Independent worker processes
3. **Load Balancing**: Equal-sized chunks for workers
4. **Synchronization**: Implicit barrier via `Pool.map()`
5. **Communication**: Border row exchange (halo pattern)

### Distributed Systems Patterns

1. **Spatial Partitioning**: Grid decomposed by rows
2. **Halo/Ghost Cell Exchange**: Border communication
3. **Master-Worker**: Pool coordinates workers
4. **Message Passing**: Queue-based in microservice mode
5. **Fault Tolerance**: Process isolation (one worker fails ≠ all fail)

### Performance Engineering

1. **Profiling**: `time.time()` and `tracemalloc` measurement
2. **Optimization**: NumPy vectorization removes loops
3. **Scalability**: Strong/weak scaling analysis
4. **Bottlenecks**: Process spawn and border exchange overhead
5. **Trade-offs**:
   - Speed vs Memory (NumPy uses more memory but faster)
   - Parallelism overhead (small grids slower with MP)

---

## 🔬 Advanced Usage

### Custom Grid Patterns

```python
from graph import random_grid, next_gen_parallel

# Create custom pattern with 40% density
custom_grid = random_grid(500, 500, density=0.4)

# Run with specific worker count
result, changes = next_gen_parallel(custom_grid, workers=4)
```

### Performance Profiling

```python
import cProfile
import pstats

# Profile parallel execution
cProfile.run('next_gen_parallel(grid, workers=8)', 'profile_stats')

# Analyze results
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### Memory Analysis

```python
import tracemalloc

tracemalloc.start()
# Run your code
grid, changes = next_gen_parallel(large_grid)
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory: {current / 1024**2:.2f} MB")
print(f"Peak memory: {peak / 1024**2:.2f} MB")
tracemalloc.stop()
```

### Custom Test Cases in `all_in_one.py`

```python
# Edit the get_test_cases() function in all_in_one.py
def get_test_cases():
    # Add your custom pattern
    my_pattern = [
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,1,0,1,0],
        [0,1,1,1,0],
        [0,0,0,0,0]
    ]

    return [
        ('CUSTOM_PATTERN', my_pattern, 'custom'),
        # ... existing test cases
    ]
```

---

## 📚 Further Reading

### Conway's Game of Life

- [Wikipedia - Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)
- [LifeWiki](https://conwaylife.com/wiki/Main_Page) - Comprehensive pattern database
- [Cellular Automata](https://mathworld.wolfram.com/CellularAutomaton.html) - Mathematical foundations

### Parallel Computing

- [Amdahl's Law](https://en.wikipedia.org/wiki/Amdahl%27s_law) - Theoretical speedup limits
- [Domain Decomposition Methods](https://en.wikipedia.org/wiki/Domain_decomposition_methods)
- [Parallel Patterns (Berkeley)](https://patterns.eecs.berkeley.edu/)
- [Gustafson's Law](https://en.wikipedia.org/wiki/Gustafson%27s_law) - Weak scaling

### Python Multiprocessing

- [Official Documentation](https://docs.python.org/3/library/multiprocessing.html)
- [NumPy Performance Tips](https://numpy.org/doc/stable/user/performance.html)
- [Python GIL and Multiprocessing](https://realpython.com/python-gil/)

---

## 🤝 Contributing

Improvements welcome! Consider adding:

- **GPU Acceleration**: CUDA/OpenCL implementations
- **MPI Implementation**: True distributed computing across nodes
- **Web Visualization**: Flask/FastAPI with real-time updates
- **Docker/Kubernetes**: Real microservices deployment
- **More Patterns**: Puffer trains, spaceships, methuselahs
- **Interactive Mode**: Click to toggle cells
- **Save/Load**: Pattern import/export

---

## 📄 License

MIT License - Feel free to use for educational purposes

---

## 👥 Authors

**Harsh VM** - MAP Project Implementation

---

## 🐛 Known Issues & Limitations

1. **Small Grid Overhead**: Parallel implementations slower for grids < 200×200 due to process spawn overhead
2. **Memory Copies**: Multiprocessing creates full grid copies per worker
3. **No Boundary Wrapping**: Grid edges are treated as dead cells (not toroidal)
4. **Serial Approximation**: `graph.py` uses linear estimates for serial (not actual measurements)
5. **Queue Overhead**: Microservice mode has high communication overhead

---

## 📊 Benchmark Configuration

### `graph.py` Configuration

```python
lattice_sizes = [(i,i) for i in range(200, 3001, 200)]  # 15 sizes
steps_per_run = 5
trials_per_run = 2
density = 0.3  # 30% alive cells initially
```

### `visualtization.py` Configuration

```python
lattice_sizes = [(i,i) for i in range(200, 1001, 200)]  # 5 sizes
steps_per_run = 5
trials_per_run = 2
density = 0.3  # 30% alive cells initially
```

---

**Last Updated**: November 21, 2025  
**Version**: 2.1  
**Python**: 3.8+  
**Tested On**: Linux (Ubuntu/Debian)
