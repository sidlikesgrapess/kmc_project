import numpy as np

N = 100

NEIGHBORS_EVEN = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
NEIGHBORS_ODD  = [(-1,  0), (-1, 1), (0, -1), (0, 1), (1,  0), (1, 1)]


def get_neighbors(r, c, n=N):
    offsets = NEIGHBORS_EVEN if r % 2 == 0 else NEIGHBORS_ODD
    return [(r + dr, c + dc) for dr, dc in offsets
            if 0 <= r + dr < n and 0 <= c + dc < n]


def build_neighbor_table(n=N):
    """Precompute flat neighbor indices. Shape (n*n, 6), -1 for out-of-bounds."""
    table = np.full((n * n, 6), -1, dtype=np.int32)
    for r in range(n):
        for c in range(n):
            offsets = NEIGHBORS_EVEN if r % 2 == 0 else NEIGHBORS_ODD
            for k, (dr, dc) in enumerate(offsets):
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n:
                    table[r * n + c, k] = nr * n + nc
    return table


def create_lattice(n=N):
    """Flat int8 array; 0 = empty, 1 = occupied."""
    return np.zeros(n * n, dtype=np.int8)


def coverage(grid):
    print(f"Current coverage: {grid.sum()} / {grid.size} = {grid.sum() / grid.size:.4f}")
    return grid.sum() / grid.size