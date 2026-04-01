import numpy as np
from collections import deque

def get_grains(grid, neighbors_fn, n=100):
    """
    BFS flood fill on final grid to assign grain IDs.
    Returns grain array (n,n), -1 for empty sites.
    """
    grain = np.full((n, n), -1, dtype=np.int32)
    gid = 0
    for r in range(n):
        for c in range(n):
            if grid[r, c] == 1 and grain[r, c] == -1:
                # new grain found, BFS from here
                queue = deque([(r, c)])
                grain[r, c] = gid
                while queue:
                    cr, cc = queue.popleft()
                    for nr, nc in neighbors_fn(cr, cc, n):
                        if grid[nr, nc] == 1 and grain[nr, nc] == -1:
                            grain[nr, nc] = gid
                            queue.append((nr, nc))
                gid += 1
    return grain

def add_site(r, c, grid):
    grid[r, c] = 1