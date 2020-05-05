import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

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

def get_diag_indices(M, N, k):
    """
    Compute the indices on a diagonal into indices on an accumulated
    distance matrix
    Parameters
    ----------
    M: int
        Number of rows
    N: int
        Number of columns
    k: int
        Index of the diagonal
    Returns
    -------
    i: ndarray(dim)
        Row indices
    j: ndarray(dim)
        Column indices
    """
    starti = k
    startj = 0
    if k > M-1:
        starti = M-1
        startj = k - (M-1)
    i = np.arange(starti, -1, -1)
    j = startj + np.arange(i.size)
    dim = np.sum(j < N) # Length of this diagonal
    i = i[0:dim]
    j = j[0:dim]
    return i, j
    

def DTWDiag(X, Y, k_save = -1, k_stop = -1, debug=False):
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
    debug: boolean
        Whether to save the accumulated cost matrix
    
    Returns
    -------
    {
        'cost': float
            The optimal cost of the alignment (if computation didn't stop prematurely),
        'S': ndarray(M, N)
            The accumulated cost matrix (if debugging),
        'd0':ndarray(min(M, N)), 'd1':ndarray(min(M, N)), 'd2':ndarray(min(M, N))
            The saved rows if a save index was chosen
    }
    """
    M = X.shape[0]
    N = Y.shape[0]
    # Initialize first two diagonals and helper variables
    d0 = np.array([np.sqrt(np.sum((X[0, :] - Y[0, :])**2))], dtype=np.float32)
    d1 = np.array([np.sqrt(np.sum((X[1, :] - Y[0, :])**2)), np.sqrt(np.sum((X[0, :] - Y[1, :])**2))], dtype=np.float32) + d0[0]
    
    # Store the result of each r2 in memory
    S = np.array([])
    if debug:
        S = np.zeros((M, N), dtype=np.float32)
        S[0, 0] = d0[0]
        S[1, 0] = d1[0]
        S[0, 1] = d1[1]
    
    # Loop through diagonals
    res = {}
    for k in range(2, M+N-1):
        i, j = get_diag_indices(M, N, k)
        dim = i.size
        d2 = np.inf*np.ones(dim, dtype=np.float32)
        
        left_cost = np.inf*np.ones(dim, dtype=np.float32)
        up_cost = np.inf*np.ones(dim, dtype=np.float32)
        diag_cost = np.inf*np.ones(dim, dtype=np.float32)
        
        # Pull out appropriate distances
        ds = np.sqrt(np.sum((X[i, :] - Y[j, :])**2, 1))
        if j[0] == 0:
            left_cost[1::] = d1[0:dim-1]
            # l > 0, i > 0
            idx = np.arange(dim)[(np.arange(dim) > 0)*(i > 0)] 
            diag_cost[idx] = d0[idx-1]
            # i > 0
            idx = np.arange(dim)[i > 0]
            up_cost[idx] = d1[idx]
        elif i[0] == X.shape[0]-1 and j[0] == 1:
            left_cost = d1[0:dim]
            # i > 0
            idx = np.arange(dim)[i > 0]
            diag_cost[idx] = d0[idx]
            up_cost[idx] = d1[idx+1]
        elif i[0] == X.shape[0]-1 and j[0] > 1:
            left_cost = d1[0:dim]
            idx = np.arange(dim)[i > 0]
            diag_cost[idx] = d0[idx+1]
            up_cost[idx] = d1[idx+1]
        
        
        d2[0:dim] = np.minimum(np.minimum(left_cost, diag_cost), up_cost) + ds
        if debug:
            S[i, j] = d2[0:dim]
        if k == k_save:
            res['d0'] = np.array(d0)
            res['d1'] = np.array(d1)
            res['d2'] = np.array(d2)
        if k == k_stop:
            break
        # Shift diagonals
        d0 = np.array(d1)
        d1 = np.array(d2)
    res['cost'] = d2[0]
    res['S'] = S
    return res

def DTWDiag_Backtrace(X, Y, cost, min_dim = 5, DTWDiag_fn = DTWDiag):
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
    """
    M = X.shape[0]
    N = Y.shape[0]
    K = M + N - 1
    
    # Do the forward computation
    k_save = int(np.ceil(K/2.0))
    res1 = DTWDiag_fn(X, Y, k_save=k_save, k_stop=k_save)

    # Do the backward computation
    k_save_rev = k_save
    if K%2 == 0:
        k_save_rev += 1
    res2 = DTWDiag_fn(np.flipud(X), np.flipud(Y), k_save=k_save_rev, k_stop=k_save_rev)
    res2['d0'], res2['d2'] = res2['d2'], res2['d0']
    for d in ['d0', 'd1', 'd2']:
        res2[d] = res2[d][::-1]
    
    # Look for optimal cost over all 3 diagonals
    center_path = []
    center_costs = []
    diagsums = []
    for ki, d in enumerate(['d0', 'd1', 'd2']):
        k = k_save - 2 + ki
        i, j = get_diag_indices(M, N, k)
        ds = np.sqrt(np.sum((X[i, :] - Y[j, :])**2, 1))
        diagsum = res1[d]+res2[d]-ds
        l = np.argmin(diagsum)
        if np.allclose(diagsum[l], cost):
            diagsums.append(diagsum[l])
            center_path.append([i[l], j[l]])
            center_costs.append([res1[d][l], res2[d][l]])
    idx = np.argmin(np.array(diagsums))
    center_costs = [center_costs[idx]]
    center_path = [center_path[idx]]
    
    # Recursively compute left paths
    L = center_path[0]
    XL = X[0:L[0]+1, :]
    YL = Y[0:L[1]+1, :]
    left_path = []
    if L[0] < min_dim or L[1] < min_dim:
        left_path = DTW_Backtrace(XL, YL)['path']
    else:
        left_path = DTWDiag_Backtrace(XL, YL, center_costs[0][0], min_dim, DTWDiag_fn)
    path = left_path[0:-1] + center_path
    
    # Recursively compute right paths
    R = center_path[-1]
    XR = X[R[0]::, :]
    YR = Y[R[1]::, :]
    right_path = []
    if XR.shape[0] < min_dim or YR.shape[0] < min_dim:
        right_path = DTW_Backtrace(XR, YR)['path']
    else:
        right_path = DTWDiag_Backtrace(XR, YR, center_costs[-1][1], min_dim, DTWDiag_fn)
    right_path = [[i + R[0], j + R[1]] for [i, j] in right_path]
    path = path + right_path[1::]

    return path