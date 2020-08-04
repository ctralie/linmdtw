import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import sparse
import time
import dynseqalign
from .dtw import dtw_brute_backtrace, linmdtw, check_euclidean_inputs

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
    [1] FastDTW: Toward Accurate Dynamic Time Warping in Linear Time and Space. Stan Salvador and Philip Chan
    
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
    do_plot: boolean
        Whether to plot the warping path at each level and save to image files
    
    Returns
    -------
    path: ndarray(K, 2)
        The  warping path
    """
    X, Y = check_euclidean_inputs(X, Y)
    minTSsize = radius + 2
    M = X.shape[0]
    N = Y.shape[0]
    X = np.ascontiguousarray(X)
    Y = np.ascontiguousarray(Y)
    if M < radius or N < radius:
        return dtw_brute_backtrace(X, Y)
    # Recursive step
    path = fastdtw(X[0::2, :], Y[0::2, :], radius, debug, level+1, do_plot)
    if type(path) is dict:
        path = path['path']
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
    if debug or do_plot: # pragma: no cover
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
    path = np.array(path, dtype=int)
    
    if do_plot: # pragma: no cover
        plt.figure(figsize=(8, 8))
        plt.imshow(S.toarray())
        path = np.array(path)
        plt.scatter(path[:, 1], path[:, 0], c='C1')
        plt.title("Level {}".format(level))
        plt.savefig("%i.png"%level, bbox_inches='tight')

    if debug: # pragma: no cover
        return {'path':path, 'S':S, 'P':P}
    else:
        return path

def get_box_area(a1, a2):
    """
    Get the area of a box specified by two anchors
    
    Parameters
    ----------
    a1: list(2)
        Row/column of first anchor
    a2: list(2)
        Row/column of second anchor
    
    Returns
    -------
    Area of box determined by these two anchors
    """
    m = a2[0]-a1[0]+1
    n = a2[1]-a1[1]+1
    return m*n

def mrmsdtw(X, Y, tau, debug=False, refine=True):
    """
    An implementation of the approximate, memory-restricted
    multiscale DTW technique from [2]
    [2] "Memory-Restricted Multiscale Dynamic Time Warping"
    Thomas Praetzlich, Jonathan Driedger and Meinard Mueller
    
    Parameters
    ----------
    X: ndarray(M, d)
        A d-dimensional Euclidean point cloud with M points
    Y: ndarray(N, d)
        A d-dimensional Euclidean point cloud with N points
    tau: int
        The max amount of cells to be in memory at any given
        time
    debug: boolean
        Whether to keep track of debugging information
    refine: boolean
        Whether to do refinement with the "white anchors"
    
    Returns
    -------
    path: ndarray(K, 2)
        The warping path
    """
    X, Y = check_euclidean_inputs(X, Y)
    M = X.shape[0]
    N = Y.shape[0]
    if M*N < tau:
        # If the matrix is already within the memory bounds, simply
        # return DTW
        return dtw_brute_backtrace(X, Y)

    ## Step 1: Perform DTW at the coarse level
    # Figure out the subsampling factor for the
    # coarse alignment based on memory requirements
    d = int(np.ceil(np.sqrt(M*N/tau)))
    X2 = np.ascontiguousarray(X[0::d, :])
    Y2 = np.ascontiguousarray(Y[0::d, :])
    anchors = dtw_brute_backtrace(X2, Y2)
    anchors = (np.array(anchors)*d).tolist()
    if anchors[-1][0] < M-1 or anchors[-1][1] < N-1:
        anchors.append([M-1, N-1])
    
    ## Step 2: Subdivide anchors if necessary to keep
    # within memory bounds
    idx = 0
    while idx < len(anchors)-1:
        a1 = anchors[idx]
        a2 = anchors[idx+1]
        if get_box_area(a1, a2) > tau:
            # Subdivide cell
            i = int((a1[0]+a2[0])/2)
            j = int((a1[1]+a2[1])/2)
            anchors = anchors[0:idx+1] + [[i, j]] + anchors[idx+1::]
        else:
            # Move on
            idx += 1
    
    ## Step 3: Do alignments in each block
    path = np.array([], dtype=int)
    # Keep track of the "black anchor" indices in the path
    banchors_idx = [0]
    for i in range(len(anchors)-1):
        a1 = anchors[i]
        a2 = anchors[i+1]
        box = [a1[0], a2[0], a1[1], a2[1]]
        pathi = linmdtw(X, Y, box=box)
        if path.size == 0:
            path = pathi
        else:
            path = np.concatenate((path, pathi[0:-1, :]), axis=0)
        banchors_idx.append(len(path)-1)
    # Add last endpoints
    path = np.concatenate((path, np.array([[M-1, N-1]], dtype=int)), axis=0)
    if not refine:
        return path
    
    ## Step 4: Come up with the set of "white anchors"
    # First choose them to be at the center of each block
    wanchors_idx = []
    for idx in range(len(banchors_idx)-1):
        wanchors_idx.append([int(0.5*(banchors_idx[idx]+banchors_idx[idx+1]))]*2)
    # Split anchor positions if the blocks are too big
    for i in range(len(wanchors_idx)-1):
        a1 = path[wanchors_idx[i][-1]]
        a2 = path[wanchors_idx[i+1][0]]
        while get_box_area(a1, a2) > tau:
            # Move the anchors towards each other
            wanchors_idx[i][-1] += 1
            wanchors_idx[i+1][0] -= 1
            a1 = path[wanchors_idx[i][-1]]
            a2 = path[wanchors_idx[i+1][0]]
    ## Step 5: Do DTW between white anchors and splice path together
    pathret = path[0:wanchors_idx[0][0]+1, :]
    for i in range(len(wanchors_idx)-1):
        a1 = path[wanchors_idx[i][-1]]
        a2 = path[wanchors_idx[i+1][0]]
        box = [a1[0], a2[0], a1[1], a2[1]]
        pathi = linmdtw(X, Y, box=box)
        pathret = np.concatenate((pathret, pathi[0:-1, :]), axis=0)
        # If there's a gap in between this box and 
        # the next one, use the path from before
        i1 = wanchors_idx[i+1][0]
        i2 = wanchors_idx[i+1][1]
        if i1 != i2:
            pathret = np.concatenate((pathret, path[i1:i2, :]), axis=0)
    i1 = wanchors_idx[-1][-1]
    pathret = np.concatenate((pathret, path[i1::, :]), axis=0)
    return pathret