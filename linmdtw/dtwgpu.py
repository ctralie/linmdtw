"""
Provides an interface to CUDA for running a parallel version
of the diagonal DTW algorithm
"""
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as sio
import pkg_resources
import sys
from AlignmentTools import *

from pycuda.compiler import SourceModule

DTW_Step_ = None

def getResourceString(filename):
    #If calling from within this directory
    fin = open(filename)
    s = fin.read()
    fin.close()
    return s

def initParallelAlgorithms():
    s = getResourceString("DTWGPU.cu")
    mod = SourceModule(s)
    global DTW_Step_
    DTW_Step_ = mod.get_function("DTW_Diag_Step")

def DTWDiag_GPU(X, Y, k_save = -1, k_stop = -1, box = None, reverse=False, debug=False, metadata=None):
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
    if not metadata:
        metadata = {}
    if not 'XGPU' in metadata:
        metadata['XGPU'] = gpuarray.to_gpu(np.array(X, dtype=np.float32))
    XGPU = metadata['XGPU']
    if not 'YGPU' in metadata:
        metadata['YGPU'] = gpuarray.to_gpu(np.array(Y, dtype=np.float32))
    YGPU = metadata['YGPU']
    dim = np.array(X.shape[1], dtype=np.int32)
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