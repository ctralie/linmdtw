"""
Provides an interface to CUDA for running the parallel IBDTW and 
partial IBDTW algorithms
"""
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.cumath
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as sio
import pkg_resources
import sys

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
    DTW_Step_ = mod.get_function("DTW_Step")

def roundUpPow2(x):
    return np.array(int(2**np.ceil(np.log2(float(x)))), dtype=np.int32)

def doDTWGPU(X, Y, debug=False):
    assert(X.shape[1] == Y.shape[1])
    dim = np.array(X.shape[1], dtype=np.int32)
    M = X.shape[0]
    N = Y.shape[0]

    diagLen = np.array(min(M, N), dtype = np.int32)
    diagLenPow2 = roundUpPow2(diagLen)
    threadsPerBlock = min(diagLen, 512)
    gridSize = int(np.ceil(diagLen/float(threadsPerBlock)))
    print("gridSize = ", gridSize)
    threadsPerBlock = np.array(threadsPerBlock, dtype=np.int32)
    M = np.array(M, dtype=np.int32)
    N = np.array(N, dtype=np.int32)
    X = gpuarray.to_gpu(np.array(X, dtype=np.float32))
    Y = gpuarray.to_gpu(np.array(Y, dtype=np.float32))

    d0 = gpuarray.to_gpu(-1*np.ones(diagLen, dtype=np.float32))
    d1 = gpuarray.to_gpu(-1*np.ones(diagLen, dtype=np.float32))
    d2 = gpuarray.to_gpu(-1*np.ones(diagLen, dtype=np.float32))
    if debug:
        U = gpuarray.to_gpu(np.zeros((M, N), dtype=np.float32))
        L = gpuarray.to_gpu(np.zeros((M, N), dtype=np.float32))
        UL = gpuarray.to_gpu(np.zeros((M, N), dtype=np.float32))
    else:
        U = gpuarray.to_gpu(np.zeros(1, dtype=np.float32))
        L = gpuarray.to_gpu(np.zeros(1, dtype=np.float32))
        UL = gpuarray.to_gpu(np.zeros(1, dtype=np.float32))

    for i in range(M+N-1):
        i = np.array(i, dtype=np.int32)
        DTW_Step_(X, Y, dim, d0, d1, d2, M, N, diagLen, i, np.array(int(debug), dtype=np.int32), U, L, UL, block=(int(threadsPerBlock), 1, 1), grid=(gridSize, 1))
        if i < M+N-2:
            # Rotate buffers
            temp = d0
            d0 = d1
            d1 = d2
            d2 = temp
    ret = {'cost':d2.get()[0]}
    if debug:
        ret['U'] = U.get()
        ret['L'] = L.get()
        ret['UL'] = UL.get()
    return ret

def testTiming():
    from DTW import DTW, getCSM, DTWPurePython
    initParallelAlgorithms()

    M = 4000
    t = 2*np.pi*np.linspace(0, 1, M)**2
    X = np.zeros((M, 2))
    X[:, 0] = np.cos(t)
    X[:, 1] = np.sin(2*t)
    N = 4000
    t = 2*np.pi*np.linspace(0, 1, N)
    Y = np.zeros((N, 2))
    Y[:, 0] = 1.1*np.cos(t)
    Y[:, 1] = 1.1*np.sin(2*t)
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    tic = time.time()
    res1 = DTW(X, Y, True)
    cost1 = res1['cost']

    time1 = time.time() - tic
    print("cost1 = ", cost1, ", time1 = ", time1)
    tic = time.time()
    
    res2 = doDTWGPU(X, Y, True)
    cost2 = res2['cost']
    time2 = time.time() - tic
    print("cost2 = ", cost2, ", time2 = ", time2)

    for i, key in enumerate(['U', 'UL', 'L']):
        plt.subplot(2, 3, i+1)
        plt.imshow(res1[key])
        plt.colorbar()
        plt.title("%s CPU"%key)
        plt.subplot(2, 3, i+4)
        plt.imshow(res2[key])
        plt.title("%s GPU"%key)
        plt.colorbar()
    plt.show()


if __name__ == '__main__':
    testTiming()