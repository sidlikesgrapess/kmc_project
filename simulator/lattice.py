import numpy as np
from simulator.grain_tracking import get_grains

N = 100

NEIGHBORS_EVEN = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
NEIGHBORS_ODD  = [(-1,  0), (-1, 1), (0, -1), (0, 1), (1,  0), (1, 1)]

def get_neighbors(r, c, n=N):
    offsets = NEIGHBORS_EVEN if r % 2 == 0 else NEIGHBORS_ODD
    return [(r+dr, c+dc) for dr, dc in offsets
            if 0 <= r+dr < n and 0 <= c+dc < n]

def create_lattice(n=N):
    grid = np.zeros((n, n), dtype=np.int8)
    return grid

def coverage(grid):
    return grid.sum() / grid.size

def grain_boundary_density(grid, n=N):
    grain = get_grains(grid, get_neighbors, n)
    boundary = 0
    total    = 0
    rows, cols = np.where(grid == 1)
    for r, c in zip(rows, cols):
        for nr, nc in get_neighbors(r, c, n):
            if grid[nr, nc] == 1:
                total += 1
                if grain[r, c] != grain[nr, nc]:
                    boundary += 1
    return boundary / total if total > 0 else 0.0