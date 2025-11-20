# ======================================================================
# Conway's Game of Life - SERIAL VERSION (One File)
# Includes:
# - Step-by-step output
# - Timing
# - Metrics (alive cells, state changes, steps)
# - Best / Average / Worst Case classification
# - Complexity printing
# ======================================================================

import time
from copy import deepcopy


# ----------------------------------------------------------
# CORE LOGIC
# ----------------------------------------------------------
def count_neighbors(grid, x, y):
    rows = len(grid)
    cols = len(grid[0])
    directions = [
        (-1,-1), (-1,0), (-1,1),
        (0,-1),         (0,1),
        (1,-1), (1,0),  (1,1)
    ]
    count = 0
    for dx, dy in directions:
        nx, ny = x+dx, y+dy
        if 0 <= nx < rows and 0 <= ny < cols:
            count += grid[nx][ny]
    return count


def next_generation(grid):
    rows = len(grid)
    cols = len(grid[0])
    new_grid = [[0]*cols for _ in range(rows)]
    changes = 0

    for i in range(rows):
        for j in range(cols):
            neighbors = count_neighbors(grid, i, j)
            prev = grid[i][j]
            new = 0

            if prev == 1:
                if neighbors == 2 or neighbors == 3:
                    new = 1
            else:
                if neighbors == 3:
                    new = 1

            new_grid[i][j] = new
            if new != prev:
                changes += 1

    return new_grid, changes


def print_grid(grid):
    for row in grid:
        print(" ".join(str(x) for x in row))


def grid_to_key(grid):
    return tuple(tuple(row) for row in grid)


# ----------------------------------------------------------
# RUNNER FUNCTION (TIMING + METRICS + STEP-BY-STEP)
# ----------------------------------------------------------
def run_and_print_steps(initial_grid, max_steps, title, case_type):

    print("\n" + "=" * 70)
    print(f"{title} ({case_type}) — max {max_steps} steps")
    print("=" * 70)

    grid = deepcopy(initial_grid)
    seen = {}
    step = 0
    total_changes = 0
    alive_counts = []

    start_time = time.time()

    while step <= max_steps:
        print(f"\nStep {step}:")
        print_grid(grid)

        alive = sum(sum(row) for row in grid)
        alive_counts.append(alive)

        key = grid_to_key(grid)
        if key in seen:
            cycle_length = step - seen[key]
            print(f"\n--> Cycle detected at step {step} (cycle length = {cycle_length}).")
            break

        seen[key] = step

        next_grid, changes = next_generation(grid)
        total_changes += changes

        if grid_to_key(next_grid) == key:
            print("\n--> Stable state reached (no changes).")
            break

        grid = next_grid
        step += 1

    end_time = time.time()
    elapsed = (end_time - start_time) * 1000  # ms

    # ----------------------------------------------------------
    # Printing Metrics
    # ----------------------------------------------------------
    print("\n----- METRICS -----")
    print(f"Total Steps Executed : {step}")
    print(f"Alive Cells Per Step : {alive_counts}")
    print(f"Total State Changes  : {total_changes}")
    print(f"Time Taken           : {elapsed:.4f} ms")

    # Complexity Information
    print("\n----- COMPLEXITY -----")
    rows = len(initial_grid)
    cols = len(initial_grid[0])
    print(f"Grid Size: {rows} x {cols} = {rows*cols} cells")

    print("\nTime Complexity:")
    print("  Each cell checks 8 neighbors → O(1)")
    print("  For R × C grid → O(R × C) per generation")
    print("  For S steps → O(S × R × C) total")

    print("\nSpace Complexity:")
    print("  Two grids stored → O(R × C)")

    print("\nBest/Average/Worst Case Meaning:")
    print("  BEST CASE: Stabilizes immediately ⇒ S = 1   ⇒ O(R × C)")
    print("  AVERAGE CASE: Moves/changes for many steps  ⇒ O(S × R × C)")
    print("  WORST CASE: Oscillators that run longest    ⇒ O(S × R × C) with max S")

    print("\n")


# ======================================================================
# TEST CASES
# Mapped to:
# - Best Case
# - Average Case
# - Worst Case
# ======================================================================
if __name__ == "__main__":

    # -------------------------
    # BEST CASE — Block (Still Life)
    # stabilizes in 1 generation
    # -------------------------
    block = [
        [0,0,0,0],
        [0,1,1,0],
        [0,1,1,0],
        [0,0,0,0]
    ]
    run_and_print_steps(block, max_steps=5, title="TEST 1: STILL LIFE (BLOCK)", case_type="BEST CASE")


    # -------------------------
    # WORST CASE — Blinker
    # oscillates forever (period 2)
    # -------------------------
    blinker = [
        [0,1,0],
        [0,1,0],
        [0,1,0]
    ]
    run_and_print_steps(blinker, max_steps=8, title="TEST 2: BLINKER (OSCILLATOR)", case_type="WORST CASE")


    # -------------------------
    # AVERAGE CASE — Glider
    # moves diagonally, continues changing for many steps
    # -------------------------
    glider = [
        [0,1,0,0,0],
        [0,0,1,0,0],
        [1,1,1,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0]
    ]
    run_and_print_steps(glider, max_steps=10, title="TEST 3: GLIDER", case_type="AVERAGE CASE")
