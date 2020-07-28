"""
Provides an interface to CUDA for running a parallel version
of the diagonal DTW algorithm
"""

import numpy as np
from .alignmenttools import get_diag_len, get_diag_indices, update_alignment_metadata
import warnings

DTW_Step_ = None
DTW_GPU_Initialized = False
DTW_GPU_Failed = False

def init_gpu():
    s = """
    __global__ void DTW_Diag_Step(float* d0, float* d1, float* d2, float* csm0, float* csm1, float* csm2, float* X, float* Y, int dim, int diagLen, int* box, int reverse, int i, int debug, float* U, float* L, float* UL, float* S) {
        //Other local variables
        int i1, i2, j1, j2; // Endpoints of the diagonal
        int thisi, thisj; // Current indices on the diagonal
        // Optimal score and particular score for up/right/left
        float score, left, up, diag; 
        int idx = threadIdx.x + blockIdx.x*blockDim.x;
        int xi, yj;

        //Process each diagonal
        score = -1;
        if (idx < diagLen) {
            // Figure out indices in X and Y on diagonal
            int M = box[1] - box[0] + 1;
            int N = box[3] - box[2] + 1;
            i1 = i;
            j1 = 0;
            if (i >= M) {
                i1 = M-1;
                j1 = i - (M-1);
            }
            j2 = i;
            i2 = 0;
            if (j2 >= N) {
                j2 = N-1;
                i2 = i - (N-1);
            }
            thisi = i1 - idx;
            thisj = j1 + idx;
            
            if (thisi >= i2 && thisj <= j2) {
                xi = thisi;
                yj = thisj;
                if (reverse == 1) {
                    xi = M-1-xi;
                    yj = N-1-yj;
                }
                xi += box[0];
                yj += box[2];
                // Step 1: Update csm2
                csm2[idx] = 0.0;
                for (int d = 0; d < dim; d++) {
                    float diff = X[xi*dim+d] - Y[yj*dim+d];
                    csm2[idx] += diff*diff;
                }
                csm2[idx] = sqrt(csm2[idx]);
                

                // Step 2: Figure out the optimal cost
                if (thisi == 0 && thisj == 0) {
                    score = 0;
                    if (debug == -1) {
                        S[0] = 0;
                        U[0] = -1;
                        L[0] = -1;
                        UL[0] = -1;
                    }
                }
                else {
                    left = -1;
                    up = -1;
                    diag = -1;
                    if (j1 == 0) {
                        if (idx > 0) {
                            left = d1[idx-1] + csm1[idx-1];
                        }
                        if (idx > 0 && thisi > 0) {
                            diag = d0[idx-1] + csm0[idx-1];
                        }
                        if (thisi > 0) {
                            up = d1[idx] + csm1[idx];
                        }
                    }
                    else if (i1 == M-1 && j1 == 1) {
                        left = d1[idx] + csm1[idx];
                        if (thisi > 0) {
                            diag = d0[idx] + csm0[idx];
                            up = d1[idx+1] + csm1[idx+1];
                        }
                    }
                    else if (i1 == M-1 && j1 > 1) {
                        left = d1[idx] + csm1[idx];
                        if (thisi > 0) {
                            diag = d0[idx+1] + csm0[idx+1];
                            up = d1[idx+1] + csm1[idx+1];
                        }
                    }
                    if (left > -1) {
                        score = left;
                    }
                    if (up > -1 && (up < score || score == -1)) {
                        score = up;
                    }
                    if (diag > -1 && (diag < score || score == -1)) {
                        score = diag;
                    }
                    if (debug == 1) {
                        U[thisi*N + thisj] = up;
                        L[thisi*N + thisj] = left;
                        UL[thisi*N + thisj] = diag;
                        S[thisi*N + thisj] = score;
                    }
                }
            }
            d2[idx] = score;
        }
    }
    """
    global DTW_GPU_Initialized
    if not DTW_GPU_Initialized:
        try:
            import pycuda.autoinit
            from pycuda.compiler import SourceModule
            mod = SourceModule(s)
            global DTW_Step_
            DTW_Step_ = mod.get_function("DTW_Diag_Step")
            DTW_GPU_Initialized = True
        except Exception as e:
            global DTW_GPU_Failed
            DTW_GPU_Failed = True
            warnings.warn("Unable to compile GPU kernel {}".format(e))

def dtw_diag_gpu(X, Y, k_save = -1, k_stop = -1, box = None, reverse=False, debug=False, metadata=None):
    """
    Compute dynamic time warping between two time-ordered
    point clouds in Euclidean space, using CUDA on the back end
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
    metadata: dictionary
        A dictionary for storing information about the computation
    """
    assert(X.shape[1] == Y.shape[1])
    global DTW_GPU_Initialized
    if not DTW_GPU_Initialized:
        init_gpu()
    if DTW_GPU_Failed:
        return
    import pycuda.gpuarray as gpuarray
    if not metadata:
        metadata = {}
    if not 'XGPU' in metadata:
        metadata['XGPU'] = gpuarray.to_gpu(np.array(X, dtype=np.float32))
    XGPU = metadata['XGPU']
    if not 'YGPU' in metadata:
        metadata['YGPU'] = gpuarray.to_gpu(np.array(Y, dtype=np.float32))
    YGPU = metadata['YGPU']
    if not box:
        box = [0, X.shape[0]-1, 0, Y.shape[0]-1]
    M = box[1] - box[0] + 1
    N = box[3] - box[2] + 1
    box_gpu = gpuarray.to_gpu(np.array(box, dtype=np.int32))
    reverse = np.array(reverse, dtype=np.int32)

    diagLen = np.array(min(M, N), dtype = np.int32)
    threadsPerBlock = min(diagLen, 512)
    gridSize = int(np.ceil(diagLen/float(threadsPerBlock)))
    threadsPerBlock = np.array(threadsPerBlock, dtype=np.int32)

    d0 = gpuarray.to_gpu(np.zeros(diagLen, dtype=np.float32))
    d1 = gpuarray.to_gpu(np.zeros(diagLen, dtype=np.float32))
    d2 = gpuarray.to_gpu(np.zeros(diagLen, dtype=np.float32))
    csm0 = gpuarray.zeros_like(d0)
    csm1 = gpuarray.zeros_like(d0)
    csm2 = gpuarray.zeros_like(d0)
    csm0len = diagLen
    csm1len = diagLen
    csm2len = diagLen

    if debug:
        U = gpuarray.to_gpu(np.zeros((M, N), dtype=np.float32))
        L = gpuarray.to_gpu(np.zeros((M, N), dtype=np.float32))
        UL = gpuarray.to_gpu(np.zeros((M, N), dtype=np.float32))
        S = gpuarray.to_gpu(np.zeros((M, N), dtype=np.float32))
    else:
        U = gpuarray.to_gpu(np.zeros(1, dtype=np.float32))
        L = gpuarray.to_gpu(np.zeros(1, dtype=np.float32))
        UL = gpuarray.to_gpu(np.zeros(1, dtype=np.float32))
        S = gpuarray.to_gpu(np.zeros(1, dtype=np.float32))

    res = {}
    for k in range(M+N-1):
        DTW_Step_(d0, d1, d2, csm0, csm1, csm2, XGPU, YGPU, np.array(X.shape[1], dtype=np.int32), diagLen, box_gpu, np.array(reverse, dtype=np.int32), np.array(k, dtype=np.int32), np.array(int(debug), dtype=np.int32), U, L, UL, S, block=(int(threadsPerBlock), 1, 1), grid=(gridSize, 1))
        csm2len = get_diag_len(box, k)
        update_alignment_metadata(metadata, csm2len)
        if k == k_save:
            res['d0'] = d0.get()
            res['csm0'] = csm0.get()[0:csm0len]
            res['d1'] = d1.get()
            res['csm1'] = csm1.get()[0:csm1len]
            res['d2'] = d2.get()
            res['csm2'] = csm2.get()[0:csm2len]
        if k == k_stop:
            break
        if k < M+N-2:
            # Rotate buffers
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
    res['cost'] = d2.get()[0] + csm2.get()[0]
    if debug:
        res['U'] = U.get()
        res['L'] = L.get()
        res['UL'] = UL.get()
        res['S'] = S.get()
    return res