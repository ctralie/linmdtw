import pytest
import numpy as np
import linmdtw

def get_random_warppath(N):
    path = np.zeros((N, 2), dtype=int)
    for i in range(1, N):
        r = np.random.rand()
        if r < 1/3:
            path[i, :] = path[i-1, :] + np.array([1, 1])
        elif r < 2/3:
            path[i, :] = path[i-1, :] + np.array([1, 0])
        else:
            path[i, :] = path[i-1, :] + np.array([0, 1])
    return path

class TestAlignmentTools:
    def test_refine(self):
        np.random.seed(0)
        P1 = get_random_warppath(1000)
        P2 = linmdtw.refine_warping_path(P1)
        assert(np.sum(np.abs(P1[-1, :] - P2[-1, :])) < 4)
