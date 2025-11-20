# Option C: Clean & Minimal Parallel Game of Life with Embedded Test Cases
# Contains ONLY:
#   - Multiprocessing
#   - NumPy Vectorized
#   - Simulated Microservice Mode
# Test cases included: Best, Average, Worst

import multiprocessing as mp
import math
import time

try:
    import numpy as np
    HAS_NUMPY = True
except:
    HAS_NUMPY = False

###############################
# COMMON UTILITIES
###############################
def print_grid(grid):
    for r in grid:
        print(" ".join(str(int(x)) for x in r))


def count_alive(grid):
    if HAS_NUMPY and isinstance(grid, np.ndarray):
        return int(grid.sum())
    return sum(sum(r) for r in grid)


def grid_to_key(grid):
    if HAS_NUMPY and isinstance(grid, np.ndarray):
        return tuple(map(tuple, grid.tolist()))
    return tuple(tuple(r) for r in grid)

###############################
# MULTIPROCESSING ENGINE
###############################
def _mp_worker(args):
    chunk, top, bottom = args
    rows = len(chunk)
    cols = len(chunk[0])
    new_chunk = [[0]*cols for _ in range(rows)]
    changes = 0

    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    for i in range(rows):
        for j in range(cols):
            n = 0
            for dx,dy in dirs:
                nx, ny = i+dx, j+dy
                if nx < 0:
                    if top is not None and 0 <= ny < cols: n += top[ny]
                elif nx >= rows:
                    if bottom is not None and 0 <= ny < cols: n += bottom[ny]
                else:
                    if 0 <= ny < cols: n += chunk[nx][ny]

            p = chunk[i][j]
            new = 1 if (p == 1 and n in (2,3)) or (p == 0 and n == 3) else 0
            new_chunk[i][j] = new
            if new != p: changes += 1

    return new_chunk, changes


def mp_next_gen(grid, workers=None):
    rows = len(grid)
    workers = workers or max(1, mp.cpu_count()-1)

    size = math.ceil(rows/workers)
    args = []
    for k in range(0, rows, size):
        chunk = grid[k:k+size]
        top = grid[k-1] if k > 0 else None
        bottom = grid[k+size] if k+size < rows else None
        args.append((chunk, top, bottom))

    with mp.Pool(len(args)) as pool:
        results = pool.map(_mp_worker, args)

    new_grid = []
    total = 0
    for c, ch in results:
        new_grid.extend(c)
        total += ch

    return new_grid, total

###############################
# NUMPY VECTORIZED ENGINE
###############################
def numpy_next_gen(grid_np):
    padded = np.pad(grid_np, 1)
    neigh = (
        padded[:-2,:-2] + padded[:-2,1:-1] + padded[:-2,2:] +
        padded[1:-1,:-2] + padded[1:-1,2:] +
        padded[2:,:-2] + padded[2:,1:-1] + padded[2:,2:]
    )
    c = padded[1:-1,1:-1]
    new = (((c==1)&((neigh==2)|(neigh==3))) | ((c==0)&(neigh==3))).astype(np.uint8)
    changes = int((new != c).sum())
    return new, changes

###############################
# SIMULATED MICROSERVICE ENGINE
###############################
def _service_loop(in_q, out_q):
    while True:
        job = in_q.get()
        if job is None or job[0] == "exit": break
        _, chunk, top, bottom, jid = job
        ng, ch = _mp_worker((chunk, top, bottom))
        out_q.put((jid, ng, ch))


def microservice_next_gen(grid, workers=None):
    rows = len(grid)
    workers = workers or max(1, mp.cpu_count()-1)
    size = math.ceil(rows/workers)

    mgr = mp.Manager()
    in_queues = [mgr.Queue() for _ in range(workers)]
    out_q = mgr.Queue()

    procs = [mp.Process(target=_service_loop, args=(iq,out_q)) for iq in in_queues]
    for p in procs: p.start()

    jobs = []
    jid = 0
    for k in range(0, rows, size):
        chunk = grid[k:k+size]
        top = grid[k-1] if k>0 else None
        bottom = grid[k+size] if k+size < rows else None
        in_queues[jid % workers].put(("work", chunk, top, bottom, jid))
        jobs.append(jid)
        jid += 1

    results = {}
    total = 0
    while len(results) < len(jobs):
        jid, ng, ch = out_q.get()
        results[jid] = (ng, ch)
        total += ch

    for iq in in_queues: iq.put(("exit",))
    for p in procs: p.join()

    new_grid = []
    for j in range(len(jobs)):
        new_grid.extend(results[j][0])

    return new_grid, total

###############################
# TEST CASES
###############################
def get_test_cases():
    best = [[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]]  # Stable block
    average = [[0,1,0,0,0],[0,0,1,0,0],[1,1,1,0,0],[0,0,0,0,0],[0,0,0,0,0]]  # Glider
    worst = [[0,1,0],[0,1,0],[0,1,0]]  # Oscillator (blinker)
    return [("BEST_CASE", best), ("AVERAGE_CASE", average), ("WORST_CASE", worst)]

###############################
# RUNNER
###############################
def run(initial, mode, steps=10, print_steps=True, workers=None):
    if mode == "numpy":
        if not HAS_NUMPY: raise RuntimeError("NumPy missing")
        grid_np = np.array(initial, dtype=np.uint8)
        grid = grid_np.tolist()
    else:
        grid = [row[:] for row in initial]
        grid_np = None

    seen = {}
    alive_hist = []
    total_ch = 0

    t0 = time.time()

    for step in range(steps+1):
        if print_steps:
            print(f"\nStep {step}")
            print_grid(grid)

        alive_hist.append(count_alive(grid))
        key = grid_to_key(grid)
        if key in seen:
            print(f"Cycle detected at step {step}")
            break
        seen[key] = True

        if mode == "mp": ng, ch = mp_next_gen(grid, workers)
        elif mode == "numpy": ng_np, ch = numpy_next_gen(grid_np); grid_np=ng_np; ng=ng_np.tolist()
        elif mode == "microservice": ng, ch = microservice_next_gen(grid, workers)
        else: raise ValueError("Invalid mode")

        total_ch += ch
        if ch == 0:
            print("Stable state reached")
            break
        grid = ng

    ms = (time.time() - t0)*1000

    print("\n--- METRICS ---")
    print("Mode:", mode)
    print("Alive History:", alive_hist)
    print("Total Changes:", total_ch)
    print(f"Time: {ms:.2f} ms")

###############################
# MAIN EXECUTION
###############################
if __name__ == "__main__":
    cases = get_test_cases()
    modes = ["mp", "numpy", "microservice"]
    for name, grid in cases:
        print("\n==============================")
        print(f"TEST CASE: {name}")
        print("==============================")
        for mode in modes:
            print(f"\n--- Running mode: {mode} ---")
            run(grid, mode, steps=10, print_steps=True)