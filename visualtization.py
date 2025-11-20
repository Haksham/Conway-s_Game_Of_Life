import time
import tracemalloc
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt
import multiprocessing as mp
import math
from copy import deepcopy
from scipy.interpolate import make_interp_spline

# ----------------------------
# Serial Version of Game of Life
# ----------------------------
def next_gen_serial(grid):
    rows, cols = len(grid), len(grid[0])
    new_grid = [[0]*cols for _ in range(rows)]
    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    changes = 0
    for i in range(rows):
        for j in range(cols):
            n = sum(grid[i+dx][j+dy] for dx,dy in dirs
                    if 0 <= i+dx < rows and 0 <= j+dy < cols)
            p = grid[i][j]
            new = 1 if (p==1 and n in (2,3)) or (p==0 and n==3) else 0
            new_grid[i][j] = new
            if new != p: changes += 1
    return new_grid, changes

# ----------------------------
# Parallel Version of Game of Life
# ----------------------------
def _mp_worker(args):
    chunk, top, bottom = args
    rows, cols = len(chunk), len(chunk[0])
    new_chunk = [[0]*cols for _ in range(rows)]
    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    changes = 0
    for i in range(rows):
        for j in range(cols):
            n = 0
            for dx, dy in dirs:
                ni, nj = i + dx, j + dy
                if ni < 0:
                    if top and 0<=nj<cols: n += top[nj]
                elif ni >= rows:
                    if bottom and 0<=nj<cols: n += bottom[nj]
                else:
                    if 0<=nj<cols: n += chunk[ni][nj]
            p = chunk[i][j]
            new = 1 if (p==1 and n in (2,3)) or (p==0 and n==3) else 0
            new_chunk[i][j] = new
            if new != p: changes += 1
    return new_chunk, changes

def next_gen_parallel(grid, workers=None):
    rows = len(grid)
    workers = workers or max(1, mp.cpu_count()-1)
    chunk_size = math.ceil(rows/workers)
    args = []
    for k in range(0, rows, chunk_size):
        chunk = grid[k:k+chunk_size]
        top = grid[k-1] if k>0 else None
        bottom = grid[k+chunk_size] if k+chunk_size<rows else None
        args.append((chunk, top, bottom))
    with mp.Pool(len(args)) as pool:
        results = pool.map(_mp_worker, args)
    new_grid = []
    total_changes = 0
    for c, ch in results:
        new_grid.extend(c)
        total_changes += ch
    return new_grid, total_changes

# ----------------------------
# Utility Functions
# ----------------------------
def random_grid(n, m, density=0.3):
    return [[1 if np.random.rand() < density else 0 for _ in range(m)] for _ in range(n)]

def measure_time_space(func, grid, steps=5, trials=2):
    times, mems = [], []
    for _ in range(trials):
        grid_copy = deepcopy(grid)
        tracemalloc.start()
        t0 = time.time()
        for _ in range(steps):
            grid_copy, changes = func(grid_copy)
            if changes == 0: break
        elapsed = time.time() - t0
        mem_used = tracemalloc.get_traced_memory()[1]/1024  # KB
        tracemalloc.stop()
        times.append(elapsed)
        mems.append(mem_used)
    return np.mean(times), np.mean(mems)

# ----------------------------
# Main Benchmark
# ----------------------------
lattice_sizes = [(i,i) for i in range(200, 1001, 200)]  # 200x200 → 1000x1000
steps_per_run = 5
trials_per_run = 2

serial_times, parallel_times = [], []
serial_mem, parallel_mem = [], []

for n, m in lattice_sizes:
    print(f"\nRunning lattice: {n}x{m}")
    grid = random_grid(n,m)

    # Serial execution for lattices <= 1000x1000
    t_serial, m_serial = measure_time_space(next_gen_serial, grid, steps=steps_per_run, trials=trials_per_run)
    serial_times.append(t_serial)
    serial_mem.append(m_serial)
    print(f"Serial: time={t_serial:.3f}s, memory={m_serial:.2f}KB")

    # Parallel execution
    t_parallel, m_parallel = measure_time_space(next_gen_parallel, grid, steps=steps_per_run, trials=trials_per_run)
    parallel_times.append(t_parallel)
    parallel_mem.append(m_parallel)
    print(f"Parallel: time={t_parallel:.3f}s, memory={m_parallel:.2f}KB")

# ----------------------------
# Graphs
# ----------------------------
x = np.array([n*m for n,m in lattice_sizes])
y_serial_time = np.array(serial_times)
y_serial_mem = np.array(serial_mem)
y_parallel_time = np.array(parallel_times)
y_parallel_mem = np.array(parallel_mem)
x_new = np.linspace(x.min(), x.max(), 300)

# Smooth curves
spl_parallel_time = make_interp_spline(x, y_parallel_time, k=3)
y_parallel_time_smooth = spl_parallel_time(x_new)
spl_parallel_mem = make_interp_spline(x, y_parallel_mem, k=3)
y_parallel_mem_smooth = spl_parallel_mem(x_new)

# Time vs Lattice
plt.figure(figsize=(10,5))
plt.plot(x_new, y_parallel_time_smooth, label="Parallel Time", color='red')
plt.scatter(x, y_parallel_time, color='red', s=10)
plt.plot(x, y_serial_time, label="Serial Time", color='blue', marker='o')
plt.xlabel("Lattice Cells (n*m)")
plt.ylabel("Time (s)")
plt.title("Time Complexity vs Lattice Size")
plt.legend()
plt.grid(True)
plt.savefig("time_vs_lattice3.png")
plt.close()

# Memory vs Lattice
plt.figure(figsize=(10,5))
plt.plot(x_new, y_parallel_mem_smooth, label="Parallel Memory", color='orange')
plt.scatter(x, y_parallel_mem, color='orange', s=10)
plt.plot(x, y_serial_mem, label="Serial Memory", color='green', marker='o')
plt.xlabel("Lattice Cells (n*m)")
plt.ylabel("Memory Usage (KB)")
plt.title("Space Complexity vs Lattice Size")
plt.legend()
plt.grid(True)
plt.savefig("space_vs_lattice3.png")
plt.close()

# Time vs Memory
plt.figure(figsize=(10,5))
plt.plot(y_parallel_mem, y_parallel_time, label="Parallel", color='red', marker='o')
plt.plot(y_serial_mem, y_serial_time, label="Serial", color='blue', marker='o')
plt.xlabel("Memory Usage (KB)")
plt.ylabel("Execution Time (s)")
plt.title("Time Complexity vs Space Complexity")
plt.legend()
plt.grid(True)
plt.savefig("time_vs_space3.png")
plt.close()
