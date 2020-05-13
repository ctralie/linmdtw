import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import sparse
import time
from AlignmentTools import *
from DTW import *
import dynseqalign

def fill_block(A, p, radius, val):
    """
    Fill a square block with values
    Parameters
    ----------
    A: ndarray(M, N) or sparse(M, N)
        The array to fill
    p: list of [i, j]
        The coordinates of the center of the box
    radius: int
        Half the width of the box
    val: float
        Value to fill in
    """
    i1 = max(p[0]-radius, 0)
    i2 = min(p[0]+radius, A.shape[0]-1)
    j1 = max(p[1]-radius, 0)
    j2 = min(p[1]+radius, A.shape[1]-1)
    A[i1:i2+1, j1:j2+1] = val

def fastdtw(X, Y, radius, debug=False, level = 0, do_plot=False):
    """
    An implementation of [1]
    [1] FastDTW: Toward Accurate Dynamic Time Warping in Linear Time and Space
                Stan Salvador and Philip Chan
    Parameters
    ----------
    X: ndarray(M, d)
        A d-dimensional Euclidean point cloud with M points
    Y: ndarray(N, d)
        A d-dimensional Euclidean point cloud with N points
    radius: int
        Radius of the l-infinity box that determines sparsity structure
        at each level
    debug: boolean
        Whether to keep track of debugging information
    level: int
        An int for keeping track of the level of recursion
    """
    minTSsize = radius + 2
    M = X.shape[0]
    N = Y.shape[0]
    X = np.ascontiguousarray(X)
    Y = np.ascontiguousarray(Y)
    if M < radius or N < radius:
        return DTW_Backtrace(X, Y)
    # Recursive step
    path = fastdtw(X[0::2, :], Y[0::2, :], radius, debug, level+1, do_plot)
    path = np.array(path)
    path *= 2
    tic = time.time()
    S = sparse.lil_matrix((M, N))
    P = sparse.lil_matrix((M, N), dtype=int)
    Occ = sparse.lil_matrix((M, N))

    ## Step 1: Figure out the indices of the occupied cells
    for p in path:
        fill_block(Occ, p, radius, 1)
    I, J = Occ.nonzero()
    # Sort cells in raster order
    idx = np.argsort(J)
    I = I[idx]
    J = J[idx]
    idx = np.argsort(I, kind='stable')
    I = I[idx]
    J = J[idx]

    ## Step 2: Find indices of left, up, and diag neighbors.  
    # All neighbors must be within bounds *and* within sparse structure
    # Make idx M+1 x N+1 so -1 will wrap around to 0
    # Make 1-indexed so all valid entries have indices > 0
    idx = sparse.coo_matrix((np.arange(I.size)+1, (I, J)), shape=(M+1, N+1)).tocsr()
    # Left neighbors
    left = np.array(idx[I, J-1], dtype=np.int32).flatten()
    left[left <= 0] = -1
    left -= 1
    # Up neighbors
    up = np.array(idx[I-1, J], dtype=np.int32).flatten()
    up[up <= 0] = -1
    up -= 1
    # Diag neighbors
    diag = np.array(idx[I-1, J-1], dtype=np.int32).flatten()
    diag[diag <= 0] = -1
    diag -= 1

    ## Step 3: Pass information to cython for dynamic programming steps
    S = np.zeros(I.size, dtype=np.float32) # Dyn prog matrix
    P = np.zeros(I.size, dtype=np.int32) # Path pointer matrix
    dynseqalign.FastDTW_DynProg_Step(X, Y, I, J, left, up, diag, S, P)
    P = sparse.coo_matrix((P, (I, J)), shape=(M, N)).tocsr()
    if debug or do_plot:
        S = sparse.coo_matrix((S, (I, J)), shape=(M, N)).tocsr()
    
    # Step 4: Do backtracing
    i = M-1
    j = N-1
    path = [[i, j]]
    step = [[0, -1], [-1, 0], [-1, -1]] # LEFT, UP, DIAG
    while not(path[-1][0] == 0 and path[-1][1] == 0):
        s = step[P[i, j]]
        i += s[0]
        j += s[1]
        path.append([i, j])
    path.reverse()
    
    if do_plot:
        plt.figure(figsize=(8, 8))
        plt.imshow(S.toarray())
        path = np.array(path)
        plt.scatter(path[:, 1], path[:, 0], c='C1')
        plt.title("Level {}".format(level))
        plt.savefig("%i.png"%level, bbox_inches='tight')

    if debug:
        return {'path':path, 'S':S, 'P':P}
    else:
        return path

def test_fastdtw():
    from fastdtw import fastdtw as fastdtw2
    from scipy.spatial.distance import euclidean
    from Tests import get_figure8s

    X, Y = get_figure8s(800, 600)
    tic = time.time()
    res1 = DTW_Backtrace(X, Y, debug=True)
    print("Elapsed time ordinary", time.time()-tic)
    path1 = np.array(res1['path'])

    tic = time.time()
    res2 = fastdtw(X, Y, 20, debug=True)
    print("Elapsed time my fastdtw", time.time()-tic)

    tic = time.time()
    fastdtw2(X, Y, radius = 20, dist=euclidean)
    print("Elapsed time other fastdtw", time.time()-tic)

    path2 = np.array(res2['path'])
    plt.imshow(res2['S'].toarray())
    plt.scatter(path1[:, 1], path1[:, 0])
    plt.scatter(path2[:, 1], path2[:, 0], marker='x')
    plt.show()

if __name__ == '__main__':
    test_fastdtw()