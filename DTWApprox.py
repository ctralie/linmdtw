import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import sparse
from AlignmentTools import *
from DTW import *

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
    minTSsize = radius + 2
    M = X.shape[0]
    N = Y.shape[0]
    if M < radius or N < radius:
        return {'path':DTW_Backtrace(np.ascontiguousarray(X), np.ascontiguousarray(Y))}
    # Recursive step
    path = fastdtw(X[0::2, :], Y[0::2], radius, debug, level+1, do_plot)['path']
    path = np.array(path)
    path *= 2
    tic = time.time()
    S = sparse.lil_matrix((M, N))
    P = sparse.lil_matrix((M, N), dtype=int)
    Occ = sparse.lil_matrix((M, N))
    # Step 1: Figure out the indices of the occupied cells
    for p in path:
        fill_block(Occ, p, radius, 1)
    I, J = Occ.nonzero()
    idx = np.argsort(J)
    I = I[idx]
    J = J[idx]
    idx = np.argsort(I, kind='stable')
    I = I[idx]
    J = J[idx]
    if level == 0:
        print("Elapsed time building sparse structure", time.time()-tic)

    # Step 2: Compute distances at those indices
    d = np.sqrt(np.sum((X[I, :] - Y[J, :])**2, 1))
    D = sparse.coo_matrix((d, (I, J)), shape=(M, N))
    D = D.tocsr()
    
    # Step 3: Do dynamic programming in the chosen band
    tic = time.time()
    S[0, 0] = np.sqrt(np.sum((X[0, :] - Y[0, :])**2))
    for i, j in zip(I, J):
        if i == 0 and j == 0:
            continue
        vals = [np.inf, np.inf, np.inf]
        # Left
        if j > 0 and Occ[i, j-1] == 1:
            vals[0] = S[i, j-1]
        # Up
        if i > 0 and Occ[i-1, j] == 1:
            vals[1] = S[i-1, j]
        # Diag
        if i > 0 and j > 0 and Occ[i-1, j-1] == 1:
            vals[2] = S[i-1, j-1]
        idx = np.argmin(vals)
        P[i, j] = idx
        S[i, j] = vals[idx] + D[i, j]
    print("Elapsed time dynamic programming level", level, ":", time.time()-tic)

    if do_plot:
        plt.figure(figsize=(8, 8))
        plt.imshow(S.toarray())
        plt.scatter(path[:, 1], path[:, 0], c='C1')
        plt.title("Level {}".format(level))
        plt.savefig("%i.png"%level, bbox_inches='tight')

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
    ret = {'path':path}
    if debug:
        ret['S'] = S
        ret['P'] = P
    return ret

def test_fastdtw():
    from fastdtw import fastdtw as fastdtw2
    from scipy.spatial.distance import euclidean
    import time

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