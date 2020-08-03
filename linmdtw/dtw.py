import numpy as np
from .alignmenttools import get_diag_len, get_diag_indices, update_alignment_metadata
import warnings

def check_euclidean_inputs(X, Y):
    """
    Check the input of two time series in Euclidean spaces, which
    are to be warped to each other.  They must satisfy:
    1. They are in the same dimension space
    2. They are 32-bit
    3. They are in C-contiguous order
    
    If #2 or #3 are not satisfied, automatically fix them and
    warn the user.
    Furthermore, warn the user if X or Y has more columns than rows,
    since the convention is that points are along rows and dimensions
    are along columns
    
    Parameters
    ----------
    X: ndarray(M, d)
        The first time series
    Y: ndarray(N, d)    
        The second time series
    
    Returns
    -------
    X: ndarray(M, d)
        The first time series, possibly copied in memory to be 32-bit, C-contiguous
    Y: ndarray(N, d)    
        The second time series, possibly copied in memory to be 32-bit, C-contiguous
    """
    if X.shape[1] != Y.shape[1]:
        raise ValueError("The input time series are not in the same dimension space")
    if X.shape[0] < X.shape[1]:
        warnings.warn("X {} has more columns than rows; did you mean to transpose?".format(X.shape))
    if Y.shape[0] < Y.shape[1]:
        warnings.warn("Y {} has more columns than rows; did you mean to transpose?".format(Y.shape))
    if not X.dtype == np.float32:
        warnings.warn("X is not 32-bit, so creating 32-bit version")
        X = np.array(X, dtype=np.float32)
    if not X.flags['C_CONTIGUOUS']:
        warnings.warn("X is not C-contiguous; creating a copy that is C-contiguous")
        X = X.copy(order='C')
    if not Y.dtype == np.float32:
        warnings.warn("Y is not 32-bit, so creating 32-bit version")
        Y = np.array(Y, dtype=np.float32)
    if not Y.flags['C_CONTIGUOUS']:
        warnings.warn("Y is not C-contiguous; creating a copy that is C-contiguous")
        Y = Y.copy(order='C')
    return X, Y

def dtw_brute(X, Y, debug=False):
    """
    Compute brute force dynamic time warping between two 
    time-ordered point clouds in Euclidean space, using 
    cython on the backend

    Parameters
    ----------
    X: ndarray(M, d)
        A d-dimensional Euclidean point cloud with M points
    Y: ndarray(N, d)
        A d-dimensional Euclidean point cloud with N points
    debug: boolean
        Whether to keep track of debugging information
    
    Returns
    -------
    {
        'cost': float
            The optimal cost of the alignment (if computation didn't stop prematurely),
        'U'/'L'/'UL': ndarray(M, N)
            The choice matrices (if debugging),
        'S': ndarray(M, N)
            The accumulated cost matrix (if debugging)
    }
    """
    from dynseqalign import DTW
    X, Y = check_euclidean_inputs(X, Y)
    return DTW(X, Y, int(debug))

def dtw_brute_backtrace(X, Y, debug=False):
    """
    Compute dynamic time warping between two time-ordered
    point clouds in Euclidean space, using cython on the 
    backend.  Then, trace back through the matrix of backpointers
    to extract an alignment path

    Parameters
    ----------
    X: ndarray(M, d)
        A d-dimensional Euclidean point cloud with M points
    Y: ndarray(N, d)
        A d-dimensional Euclidean point cloud with N points
    debug: boolean
        Whether to keep track of debugging information
    Returns
    -------
    path (If not debugging): ndarray(K, 2)
        The warping path
    
    If debugging
    {
        'cost': float
            The optimal cost of the alignment (if computation didn't stop prematurely),
        'U'/'L'/'UL': ndarray(M, N)
            The choice matrices (if debugging),
        'S': ndarray(M, N)
            The accumulated cost matrix (if debugging)
        'path': ndarray(K, 2)
            The warping path
    }
    """
    res = dtw_brute(X, Y, debug)
    res['P'] = np.asarray(res['P'])
    if debug: # pragma: no cover
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
    path = np.array(path, dtype=int)
    if debug: # pragma: no cover
        res['path'] = path
        return res
    return path
    

def dtw_diag(X, Y, k_save = -1, k_stop = -1, box = None, reverse=False, debug=False, metadata=None):
    """
    A CPU version of linear memory diagonal DTW

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
    metadata: dictionary
        A dictionary for storing information about the computation
    
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
    from dynseqalign import DTW_Diag_Step
    if not box:
        box = [0, X.shape[0]-1, 0, Y.shape[0]-1]
    M = box[1] - box[0] + 1
    N = box[3] - box[2] + 1
    box = np.array(box, dtype=np.int32)

    # Debugging info
    U = np.zeros((1, 1), dtype=np.float32)
    L = np.zeros_like(U)
    UL = np.zeros_like(U)
    S = np.zeros_like(U)
    if debug: # pragma: no cover
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
    csm0len = diagLen
    csm1len = diagLen
    csm2len = diagLen

    # Loop through diagonals
    res = {}
    for k in range(M+N-1):
        DTW_Diag_Step(d0, d1, d2, csm0, csm1, csm2, X, Y, diagLen, box, int(reverse), k, int(debug), U, L, UL, S)
        csm2len = get_diag_len(box, k)
        if metadata:
            update_alignment_metadata(metadata, csm2len)
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
        temp = d0
        d0 = d1
        d1 = d2
        d2 = temp
        temp = csm0
        csm0 = csm1
        csm1 = csm2
        csm2 = temp
        temp = csm0len
        csm0len = csm1len
        csm1len = csm2len
        csm2len = temp
    res['cost'] = d2[0] + csm2[0]
    if debug: # pragma: no cover
        res['U'] = U
        res['L'] = L
        res['UL'] = UL
        res['S'] = S
    return res

def linmdtw(X, Y, box=None, min_dim=500, do_gpu=True, metadata=None):
    """
    Linear memory exact, parallelizable DTW

    Parameters
    ----------
    X: ndarray(N1, d)
        An N1-dimensional Euclidean point cloud
    Y: ndarray(N2, d)
        An N2-dimensional Euclidean point cloud
    min_dim: int
        If one of the dimensions of the rectangular region
        to the left or to the right is less than this number,
        then switch to brute force CPU
    do_gpu: boolean
        If true, use the GPU diagonal DTW function as a subroutine.
        Otherwise, use the CPU version.  Both are linear memory, but 
        the GPU will go faster for larger synchronization problems
    metadata: dictionary
        A dictionary for storing information about the computation
    
    Returns
    -------
    path: ndarray(K, 2)
        The optimal warping path
    """
    X, Y = check_euclidean_inputs(X, Y)
    dtw_diag_fn = dtw_diag
    if do_gpu:
        from .dtwgpu import DTW_GPU_Initialized, init_gpu, dtw_diag_gpu
        if not DTW_GPU_Initialized:
            init_gpu()
        from .dtwgpu import DTW_GPU_Failed
        if DTW_GPU_Failed:
            warnings.warn("Falling back to CPU")
            do_gpu = False
        else:
            dtw_diag_fn = dtw_diag_gpu
    if not box:
        box = [0, X.shape[0]-1, 0, Y.shape[0]-1]
    M = box[1]-box[0]+1
    N = box[3]-box[2]+1

    # Stopping condition, revert to CPU
    if M < min_dim or N < min_dim:
        if metadata:
            metadata['totalCells'] += M*N
        path = dtw_brute_backtrace(X[box[0]:box[1]+1, :], Y[box[2]:box[3]+1, :])
        for p in path:
            p[0] += box[0]
            p[1] += box[2]
        return path
    
    # Otherwise, proceed with recursion
    K = M + N - 1
    # Do the forward computation
    k_save = int(np.ceil(K/2.0))
    res1 = dtw_diag_fn(X, Y, k_save=k_save, k_stop=k_save, box=box, metadata=metadata)

    # Do the backward computation
    k_save_rev = k_save
    if K%2 == 0:
        k_save_rev += 1
    res2 = dtw_diag_fn(X, Y, k_save=k_save_rev, k_stop=k_save_rev, box=box, reverse=True, metadata=metadata)
    res2['d0'], res2['d2'] = res2['d2'], res2['d0']
    res2['csm0'], res2['csm2'] = res2['csm2'], res2['csm0']
    # Chop off extra diagonal elements
    for res in res1, res2:
        for i in range(3):
            res['d%i'%i] = res['d%i'%i][0:res['csm%i'%i].size]
    # Line up the reverse diagonals
    for d in ['d0', 'd1', 'd2', 'csm0', 'csm1', 'csm2']:
        res2[d] = res2[d][::-1]
    
    # Find the min cost over the three diagonals and split on that element
    min_cost = np.inf
    min_idxs = []
    for k in range(3):
        dleft = res1['d%i'%k]
        dright = res2['d%i'%k]
        csmright = res2['csm%i'%k]
        diagsum = dleft + dright + csmright
        idx = np.argmin(diagsum)
        if diagsum[idx] < min_cost:
            min_cost = diagsum[idx]
            i, j = get_diag_indices(X.shape[0], Y.shape[0], k_save-2+k, box)
            min_idxs = [i[idx], j[idx]]

    # Recursively compute left paths
    left_path = []
    box_left = [box[0], min_idxs[0], box[2], min_idxs[1]]
    left_path = linmdtw(X, Y, box_left, min_dim, do_gpu, metadata)

    # Recursively compute right paths
    right_path = []
    box_right = [min_idxs[0], box[1], min_idxs[1], box[3]]
    right_path = linmdtw(X, Y, box_right, min_dim, do_gpu, metadata)
    
    return np.concatenate((left_path, right_path[1::, :]), axis=0)

    