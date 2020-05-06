import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from AlignmentTools import get_diag_indices

def DTW(X, Y, debug=False):
    """
    Compute dynamic time warping between two time-ordered
    point clouds in Euclidean space, using cython on the 
    backend
    Parameters
    ----------
    X: ndarray(M, d)
        A d-dimensional Euclidean point cloud with M points
    Y: ndarray(N, d)
        A d-dimensional Euclidean point cloud with N points
    debug: boolean
        Whether to keep track of debugging information
    """
    import dynseqalign
    return dynseqalign.DTW(X, Y, int(debug))

def DTW_Backtrace(X, Y, debug=False):
    """
    Compute dynamic time warping between two time-ordered
    point clouds in Euclidean space, using cython on the 
    backend
    Parameters
    ----------
    X: ndarray(M, d)
        A d-dimensional Euclidean point cloud with M points
    Y: ndarray(N, d)
        A d-dimensional Euclidean point cloud with N points
    debug: boolean
        Whether to keep track of debugging information
    """
    res = DTW(X, Y, debug)
    res['P'] = np.asarray(res['P'])
    if debug:
        for key in ['U', 'L', 'UL', 'S']:
            res[key] = np.asarray(res[key])
    i = X.shape[0]-1
    j = Y.shape[0]-1
    path = [[i, j]]
    step = [[0, -1], [-1, 0], [-1, -1]] # LEFT, UP, DIAG
    while not(path[-1][0] == 0 and path[-1][1] == 0):
        s = step[res['P'][i, j]]
        i += s[0]
        j += s[1]
        path.append([i, j])
    path.reverse()
    res['path'] = path
    return res
    

def DTWDiag(X, Y, k_save = -1, k_stop = -1, box = None, reverse=False, debug=False):
    """
    Parameters
    ----------
    X: ndarray(M, d)
        An M-dimensional Euclidean point cloud
    Y: ndarray(N, d)
        An N-dimensional Euclidean point cloud
    k_save: int
        Index of the diagonal d2 at which to save d0, d1, and d2
    k_stop: int
        Index of the diagonal d2 at which to stop computation
    box: list
        A list of [startx, endx, starty, endy]
    reverse: boolean
        Whether we're going in reverse
    debug: boolean
        Whether to save the accumulated cost matrix
    
    Returns
    -------
    {
        'cost': float
            The optimal cost of the alignment (if computation didn't stop prematurely),
        'U'/'L'/'UL': ndarray(M, N)
            The choice matrices (if debugging),
        'd0'/'d1'/'d2':ndarray(min(M, N))
            The saved rows if a save index was chosen,
        'csm0'/'csm1'/'csm2':ndarray(min(M, N))
            The saved cross-similarity distances if a save index was chosen
    }
    """
    import dynseqalign
    if not box:
        box = [0, X.shape[0]-1, 0, Y.shape[0]-1]
    M = box[1] - box[0] + 1
    N = box[3] - box[2] + 1

    # Debugging info
    U = np.zeros((1, 1), dtype=np.float32)
    L = np.zeros_like(U)
    UL = np.zeros_like(U)
    S = np.zeros_like(U)
    if debug:
        U = np.zeros((M, N), dtype=np.float32)
        L = np.zeros_like(U)
        UL = np.zeros_like(U)
        S = np.zeros_like(U)
    
    # Diagonals
    diagLen = min(M, N)
    d0 = np.zeros(diagLen, dtype=np.float32)
    d1 = np.zeros_like(d0)
    d2 = np.zeros_like(d0)
    # Distances between points along diagonals
    csm0 = np.zeros_like(d0)
    csm1 = np.zeros_like(d1)
    csm2 = np.zeros_like(d2)

    # Loop through diagonals
    res = {}
    for k in range(M+N-1):
        dynseqalign.DTW_Diag_Step(d0, d1, d2, csm0, csm1, M, N, diagLen, k, int(debug), U, L, UL, S)
        i, j = get_diag_indices(X.shape[0], Y.shape[0], k, box, reverse)
        csm2 = np.sqrt(np.sum((X[i, :] - Y[j, :])**2, 1))
        if k == k_save:
            res['d0'] = d0.copy()
            res['csm0'] = csm0.copy()
            res['d1'] = d1.copy()
            res['csm1'] = csm1.copy()
            res['d2'] = d2.copy()
            res['csm2'] = csm2.copy()
        if k == k_stop:
            break
        # Shift diagonals
        d0 = d1.copy()
        csm0 = csm1.copy()
        d1 = d2.copy()
        csm1 = csm2.copy()
    res['cost'] = d2[0] + csm2[0]
    if debug:
        res['U'] = U
        res['L'] = L
        res['UL'] = UL
        res['S'] = S
    return res

def DTWDiag_Backtrace(X, Y, cost, min_dim = 50, max_dim = 500, DTWDiag_fn = DTWDiag, pathGT = None):
    """
    Parameters
    ----------
    X: ndarray(N1, d)
        An N1-dimensional Euclidean point cloud
    Y: ndarray(N2, d)
        An N2-dimensional Euclidean point cloud
    cost: float
        Known cost to match X to Y between start and end
    min_dim: int
        If one of the dimensions of the rectangular region
        to the left or to the right is less than this number,
        then switch to brute force
    DTWDiag_fn: function handle
        A function handle to the function used to compute diagonal-based
        DTW, so that the GPU version can be easily swapped in
    path_GT: ndarray(N, 2)
        A ground-truth warping path to check against if debugging
    """
    M = X.shape[0]
    N = Y.shape[0]
    K = M + N - 1
    
    rtol = 1e-5
    atol = 1e-8
    # Do the forward computation
    k_save = int(np.ceil(K/2.0))
    res1 = DTWDiag_fn(X, Y, k_save=k_save, k_stop=k_save)

    # Do the backward computation
    k_save_rev = k_save
    if K%2 == 0:
        k_save_rev += 1
    X2 = np.zeros_like(X)
    X2[:, :] = np.flipud(X)
    Y2 = np.zeros_like(Y)
    Y2[:, :] = np.flipud(Y)
    res2 = DTWDiag_fn(X2, Y2, k_save=k_save_rev, k_stop=k_save_rev)
    res2['d0'], res2['d2'] = res2['d2'], res2['d0']
    res2['csm0'], res2['csm2'] = res2['csm2'], res2['csm0']
    for d in ['d0', 'd1', 'd2', 'csm0', 'csm1', 'csm2']:
        res2[d] = res2[d][::-1]
    
    