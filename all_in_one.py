#!/usr/bin/env python3
"""
game_of_life_multi_modes.py

Single-file: Serial, Multiprocessing, NumPy, NumPy+Multiprocessing, Simulated Microservice
- Includes three test cases (Best / Average / Worst)
- Step-by-step printing (configurable)
- Timing and metrics
"""

import argparse
import time
import math
from copy import deepcopy
from functools import partial

# Multiprocessing imports (standard library)
import multiprocessing as mp

# Optional: NumPy
try:
    import numpy as np
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False


# -------------------------
# Utilities / Helpers
# -------------------------
def print_grid(grid):
    for row in grid:
        print(" ".join(str(int(x)) for x in row))


def grid_to_key(grid):
    # Accepts list-of-lists or numpy array
    if HAS_NUMPY and isinstance(grid, np.ndarray):
        return tuple(map(tuple, grid.tolist()))
    return tuple(tuple(row) for row in grid)


def count_alive(grid):
    if HAS_NUMPY and isinstance(grid, np.ndarray):
        return int(grid.sum())
    return sum(sum(row) for row in grid)


# -------------------------
# 1) Serial Implementation (pure python loops)
# -------------------------
def serial_next_generation(grid):
    rows = len(grid)
    cols = len(grid[0])
    new_grid = [[0] * cols for _ in range(rows)]
    changes = 0

    directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    for i in range(rows):
        for j in range(cols):
            neighbors = 0
            for dx, dy in directions:
                nx, ny = i + dx, j + dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    neighbors += grid[nx][ny]
            prev = grid[i][j]
            new = 1 if (prev == 1 and (neighbors == 2 or neighbors == 3)) or (prev == 0 and neighbors == 3) else 0
            new_grid[i][j] = new
            if new != prev:
                changes += 1

    return new_grid, changes


# -------------------------
# 2) Multiprocessing Implementation (split by row-chunks)
# -------------------------
def _mp_worker_compute(args):
    """
    Worker for multiprocessing splitting by rows.
    Receives (chunk_rows, top_row, bottom_row) where chunk_rows are the rows this worker is responsible for
    but top_row and bottom_row are the immediate neighbor rows (or None) for boundary checks.
    """
    chunk_rows, top_row, bottom_row = args
    rows = len(chunk_rows)
    cols = len(chunk_rows[0])
    new_chunk = [[0] * cols for _ in range(rows)]
    changes = 0

    directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    # We'll refer to "global" indices conceptually: top_row corresponds to row index -1 of chunk, bottom_row -> rows
    for i in range(rows):
        for j in range(cols):
            neighbors = 0
            for dx, dy in directions:
                nx = i + dx
                ny = j + dy
                if nx < 0:
                    # look into top_row if exists
                    if top_row is not None:
                        if 0 <= ny < cols:
                            neighbors += top_row[ny]
                elif nx >= rows:
                    # look into bottom_row if exists
                    if bottom_row is not None:
                        if 0 <= ny < cols:
                            neighbors += bottom_row[ny]
                else:
                    if 0 <= ny < cols:
                        neighbors += chunk_rows[nx][ny]
            prev = chunk_rows[i][j]
            new = 1 if (prev == 1 and (neighbors == 2 or neighbors == 3)) or (prev == 0 and neighbors == 3) else 0
            new_chunk[i][j] = new
            if new != prev:
                changes += 1

    return new_chunk, changes


def mp_next_generation(grid, num_workers=None):
    rows = len(grid)
    cols = len(grid[0])
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    # determine chunk sizes
    chunk_size = math.ceil(rows / num_workers)
    chunks_args = []
    for k in range(0, rows, chunk_size):
        chunk_rows = grid[k: min(k + chunk_size, rows)]
        # top_row is the previous row in grid if exists
        top_row = grid[k - 1] if k - 1 >= 0 else None
        # bottom_row is the next row after this chunk if exists
        bottom_row = grid[min(k + chunk_size, rows)] if (k + chunk_size) < rows else None
        chunks_args.append((chunk_rows, top_row, bottom_row))

    # Pool map
    with mp.Pool(processes=len(chunks_args)) as pool:
        results = pool.map(_mp_worker_compute, chunks_args)

    # assemble new grid
    new_grid = []
    total_changes = 0
    for new_chunk, changes in results:
        new_grid.extend(new_chunk)
        total_changes += changes

    return new_grid, total_changes


# -------------------------
# 3) NumPy vectorized Implementation (no Python loops)
#    Uses pad + slicing to compute neighbor sums (no wrap)
# -------------------------
def numpy_next_generation(grid_np):
    """
    grid_np: numpy 2D array of 0/1
    returns new_grid_np, changes_count
    """
    # pad with zeros on all sides so neighbors off-edge are zero
    padded = np.pad(grid_np, pad_width=1, mode='constant', constant_values=0)
    # sum of 8 neighbors by slicing (no wrap)
    neighbors = (
        padded[0:-2, 0:-2] + padded[0:-2, 1:-1] + padded[0:-2, 2:] +
        padded[1:-1, 0:-2] +                 0 + padded[1:-1, 2:] +
        padded[2:  , 0:-2] + padded[2:  , 1:-1] + padded[2:  , 2:]
    )
    # current central
    central = padded[1:-1, 1:-1]
    new = ((central == 1) & ((neighbors == 2) | (neighbors == 3))) | ((central == 0) & (neighbors == 3))
    new = new.astype(np.uint8)
    changes = int((new != central).sum())
    return new, changes


# -------------------------
# 4) NumPy + Multiprocessing (split by row-chunks; each worker uses NumPy)
# -------------------------
def _numpy_mp_worker(args):
    """
    args: (chunk_np, top_row_np (1d) or None, bottom_row_np or None)
    chunk_np is a 2D numpy array (rows_in_chunk x cols)
    """
    chunk_np, top_row_np, bottom_row_np = args
    # create padded area:
    # stack rows: [top_row (1 x cols) if exists] + chunk + [bottom_row if exists]
    arrays = []
    if top_row_np is not None:
        arrays.append(top_row_np.reshape(1, -1))
    arrays.append(chunk_np)
    if bottom_row_np is not None:
        arrays.append(bottom_row_np.reshape(1, -1))
    stacked = np.vstack(arrays)
    # now use same neighbor logic as numpy_next_generation but operate on stacked and then extract central slice
    padded = np.pad(stacked, pad_width=((1,1),(1,1)), mode='constant', constant_values=0)
    neighbors = (
        padded[0:-2, 0:-2] + padded[0:-2, 1:-1] + padded[0:-2, 2:] +
        padded[1:-1, 0:-2] +                 0 + padded[1:-1, 2:] +
        padded[2:  , 0:-2] + padded[2:  , 1:-1] + padded[2:  , 2:]
    )
    central = padded[1:-1, 1:-1]
    new_stacked = ((central == 1) & ((neighbors == 2) | (neighbors == 3))) | ((central == 0) & (neighbors == 3))
    new_stacked = new_stacked.astype(np.uint8)

    # extract only the rows corresponding to the chunk (i.e., excluding top_row and bottom_row if they were present)
    start = 1 if top_row_np is not None else 0
    end = start + chunk_np.shape[0]
    new_chunk = new_stacked[start:end, :]
    changes = int((new_chunk != chunk_np).sum())
    return new_chunk, changes


def numpy_mp_next_generation(grid_np, num_workers=None):
    rows, cols = grid_np.shape
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    chunk_size = math.ceil(rows / num_workers)
    args = []
    for k in range(0, rows, chunk_size):
        start = k
        end = min(k + chunk_size, rows)
        chunk_np = grid_np[start:end, :]
        top_row = grid_np[start - 1, :] if start - 1 >= 0 else None
        bottom_row = grid_np[end, :] if end < rows else None
        args.append((chunk_np, top_row, bottom_row))

    # multiprocessing pool with numpy inside workers
    with mp.Pool(processes=len(args)) as pool:
        results = pool.map(_numpy_mp_worker, args)

    new_grid_parts = []
    total_changes = 0
    for new_chunk, changes in results:
        new_grid_parts.append(new_chunk)
        total_changes += changes

    new_grid = np.vstack(new_grid_parts)
    return new_grid, total_changes


# -------------------------
# 5) Simulated "Microservice" mode
#    (spawns worker processes that communicate via queues)
# -------------------------
def _sim_worker_loop(in_q, out_q):
    """
    Worker process that receives jobs via in_q and sends back results via out_q.
    Jobs: ('compute', chunk_rows, top_row, bottom_row, job_id) -> returns ('result', new_chunk, changes, job_id)
    Exit: ('exit',)
    """
    while True:
        job = in_q.get()
        if job is None:
            break
        tag = job[0]
        if tag == 'compute':
            _, chunk_rows, top_row, bottom_row, job_id = job
            # reuse mp worker compute (pure python)
            new_chunk, changes = _mp_worker_compute((chunk_rows, top_row, bottom_row))
            out_q.put(('result', new_chunk, changes, job_id))
        elif tag == 'exit':
            break
        else:
            # ignore unknown
            pass


def simulated_microservice_next_generation(grid, num_workers=None):
    rows = len(grid)
    cols = len(grid[0])
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    chunk_size = math.ceil(rows / num_workers)

    manager = mp.Manager()
    in_queues = []
    out_queue = manager.Queue()

    workers = []
    for w in range(min(num_workers, math.ceil(rows / chunk_size))):
        iq = manager.Queue()
        in_queues.append(iq)
        p = mp.Process(target=_sim_worker_loop, args=(iq, out_queue))
        p.start()
        workers.append(p)

    # distribute jobs round-robin
    jobs = []
    job_id = 0
    for k in range(0, rows, chunk_size):
        chunk_rows = grid[k: min(k + chunk_size, rows)]
        top_row = grid[k - 1] if k - 1 >= 0 else None
        bottom_row = grid[min(k + chunk_size, rows)] if (k + chunk_size) < rows else None
        target_q = in_queues[job_id % len(in_queues)]
        target_q.put(('compute', chunk_rows, top_row, bottom_row, job_id))
        jobs.append(job_id)
        job_id += 1

    # collect results
    results_collected = 0
    parts = {}
    total_changes = 0
    while results_collected < len(jobs):
        res = out_queue.get()
        if res[0] == 'result':
            _, new_chunk, changes, jid = res
            parts[jid] = (new_chunk, changes)
            total_changes += changes
            results_collected += 1

    # send exit signals
    for iq in in_queues:
        iq.put(('exit',))
    for p in workers:
        p.join()

    # assemble in job id order
    new_grid = []
    for jid in range(len(jobs)):
        new_chunk, _ = parts[jid]
        new_grid.extend(new_chunk)

    return new_grid, total_changes


# -------------------------
# Runner harness: step-by-step with cycle detection and metrics
# -------------------------
def run_mode(initial_grid, mode='serial', max_steps=10, print_steps=True, num_workers=None):
    """
    mode in {'serial', 'mp', 'numpy', 'numpy_mp', 'microservice'}
    initial_grid: list-of-lists
    """
    # convert to numpy if needed
    if mode in ('numpy', 'numpy_mp'):
        if not HAS_NUMPY:
            raise RuntimeError("NumPy not available; install numpy to use numpy modes.")
        grid_np = np.array(initial_grid, dtype=np.uint8)
    else:
        grid_np = None

    grid = deepcopy(initial_grid)
    seen = {}
    step = 0
    total_changes = 0
    alive_counts = []
    t0 = time.time()

    while step <= max_steps:
        if print_steps:
            print(f"\nStep {step}:")
            if mode in ('numpy', 'numpy_mp'):
                print_grid(grid_np.tolist())
            else:
                print_grid(grid)

        alive = count_alive(grid_np if grid_np is not None else grid)
        alive_counts.append(alive)

        key = grid_to_key(grid_np if grid_np is not None else grid)
        if key in seen:
            cycle_length = step - seen[key]
            if print_steps:
                print(f"\n--> Cycle detected at step {step} (length {cycle_length}).")
            break
        seen[key] = step

        # compute next based on mode
        if mode == 'serial':
            next_grid, changes = serial_next_generation(grid)
            grid = next_grid
        elif mode == 'mp':
            next_grid, changes = mp_next_generation(grid, num_workers=num_workers)
            grid = next_grid
        elif mode == 'numpy':
            next_grid_np, changes = numpy_next_generation(grid_np)
            grid_np = next_grid_np
            grid = grid_np.tolist()
        elif mode == 'numpy_mp':
            next_grid_np, changes = numpy_mp_next_generation(grid_np, num_workers=num_workers)
            grid_np = next_grid_np
            grid = grid_np.tolist()
        elif mode == 'microservice':
            next_grid, changes = simulated_microservice_next_generation(grid, num_workers=num_workers)
            grid = next_grid
        else:
            raise ValueError("Unknown mode")

        total_changes += changes

        # stability check: if next equals current (no changes)
        if changes == 0:
            if print_steps:
                print(f"\n--> Stable state reached at step {step}.")
            break

        step += 1

    t1 = time.time()
    elapsed_ms = (t1 - t0) * 1000.0

    metrics = {
        'mode': mode,
        'steps_executed': step,
        'alive_counts': alive_counts,
        'total_changes': total_changes,
        'time_ms': elapsed_ms,
        'grid_cells': len(initial_grid) * len(initial_grid[0])
    }

    # print metrics summary for this run
    print("\n--- RUN METRICS ---")
    print(f"Mode: {mode}")
    print(f"Steps executed: {metrics['steps_executed']}")
    print(f"Alive counts per step: {metrics['alive_counts']}")
    print(f"Total state changes (sum across steps): {metrics['total_changes']}")
    print(f"Time taken: {metrics['time_ms']:.4f} ms")
    print(f"Grid size (cells): {metrics['grid_cells']}")

    # complexity notes
    print("\nComplexity summary (per generation):")
    print("  Time: O(R * C) (NumPy implementations have lower constant factors)")
    print("  Space: O(R * C)")

    return metrics


# -------------------------
# Test cases (Best / Average / Worst)
# -------------------------
def get_test_cases():
    """
    Returns a dict of (name, grid, classification)
    Best: block (stabilize immediately)
    Average: glider (moves)
    Worst: blinker (oscillator / keeps flipping)
    """
    block = [
        [0,0,0,0],
        [0,1,1,0],
        [0,1,1,0],
        [0,0,0,0]
    ]

    blinker = [
        [0,1,0],
        [0,1,0],
        [0,1,0]
    ]

    glider = [
        [0,1,0,0,0],
        [0,0,1,0,0],
        [1,1,1,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0]
    ]

    return [
        ('BEST_CASE - BLOCK', block, 'best'),
        ('WORST_CASE - BLINKER', blinker, 'worst'),
        ('AVERAGE_CASE - GLIDER', glider, 'average'),
    ]


# -------------------------
# Main: CLI and orchestrator
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Game of Life: serial/mp/numpy/numpy_mp/microservice modes (single-file)")
    parser.add_argument('--mode', choices=['serial', 'mp', 'numpy', 'numpy_mp', 'microservice', 'all'], default='all',
                        help="Which implementation to run (default: all in comparison)")
    parser.add_argument('--steps', type=int, default=10, help="Max steps per test (default 10)")
    parser.add_argument('--no-steps', action='store_true', help="Do not print step-by-step grids (only metrics)")
    parser.add_argument('--workers', type=int, default=None, help="Number of worker processes for mp modes (default: cpu_count-1)")
    parser.add_argument('--run-test', choices=['best', 'average', 'worst', 'all'], default='all', help="Which test(s) to run")
    args = parser.parse_args()

    tests = get_test_cases()
    # filter tests per --run-test
    if args.run_test != 'all':
        tests = [t for t in tests if t[2] == args.run_test]

    modes_to_run = []
    if args.mode == 'all':
        # default order: serial, mp, numpy (if available), numpy_mp (if available), microservice
        modes_to_run = ['serial', 'mp']
        if HAS_NUMPY:
            modes_to_run += ['numpy', 'numpy_mp']
        modes_to_run += ['microservice']
    else:
        if args.mode in ('numpy', 'numpy_mp') and not HAS_NUMPY:
            raise RuntimeError("NumPy not available; install numpy or choose another mode.")
        modes_to_run = [args.mode]

    # overall comparison table
    comparison = []

    for test_name, grid, klass in tests:
        print("\n" + "#" * 80)
        print(f"TEST: {test_name} (classification: {klass})")
        print("#" * 80)
        for mode in modes_to_run:
            print("\n" + "-" * 60)
            print(f"Running mode: {mode}")
            print("-" * 60)
            try:
                metrics = run_mode(grid, mode=mode, max_steps=args.steps, print_steps=(not args.no_steps), num_workers=args.workers)
                metrics.update({'test_name': test_name})
                comparison.append(metrics)
            except Exception as e:
                print(f"Error running mode {mode}: {e}")

    # Print comparison summary
    print("\n\n" + "=" * 80)
    print("COMPARISON SUMMARY (per test-case and mode)")
    print("=" * 80)
    # header
    fmt = "{:28} | {:10} | {:7} | {:10} | {:12}"
    print(fmt.format("Test", "Mode", "Steps", "Time(ms)", "TotalChanges"))
    print("-" * 80)
    for m in comparison:
        print(fmt.format(m['test_name'][:28], m['mode'], str(m['steps_executed']), f"{m['time_ms']:.2f}", str(m['total_changes'])))
    print("\nNote: NumPy modes generally have much smaller constants and will be faster for large grids.\n")

if __name__ == "__main__":
    main()
