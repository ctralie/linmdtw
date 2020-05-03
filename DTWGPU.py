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

DTW_ = None

def getResourceString(filename):
    #If calling from within this directory
    fin = open(filename)
    s = fin.read()
    fin.close()
    return s

def initParallelAlgorithms():
    s = getResourceString("DTWGPU.cu")
    mod = SourceModule(s)
    global DTW_
    DTW_ = mod.get_function("DTW")

def roundUpPow2(x):
    return np.array(int(2**np.ceil(np.log2(float(x)))), dtype=np.int32)

def doDTWGPU(CSM):
    #Minimum dimension of array can be at max size 1024
    #for this scheme to fit in memory
    M = CSM.shape[0]
    N = CSM.shape[1]

    diagLen = np.array(min(M, N), dtype = np.int32)
    diagLenPow2 = roundUpPow2(diagLen)
    threadsPerBlock = min(diagLen, 512)
    gridSize = int(np.ceil(diagLen/float(threadsPerBlock)))
    print("gridSize = ", gridSize)
    threadsPerBlock = np.array(threadsPerBlock, dtype=np.int32)
    res = gpuarray.to_gpu(np.array([0.0], dtype=np.float32))
    M = np.array(M, dtype=np.int32)
    N = np.array(N, dtype=np.int32)
    CSM = gpuarray.to_gpu(CSM)

    x = gpuarray.to_gpu(np.zeros(diagLen*3, dtype=np.float32))

    DTW_(CSM, x, M, N, diagLen, threadsPerBlock, res, block=(int(threadsPerBlock), 1, 1), grid=(gridSize, 1))
    ret = res.get()[0]
    return ret

def testTiming():
    from DTW import DTW, getCSM
    initParallelAlgorithms()

    M = 2000
    t = 2*np.pi*np.linspace(0, 1, M)**2
    X = np.zeros((M, 2))
    X[:, 0] = np.cos(t)
    X[:, 1] = np.sin(2*t)
    N = 1201
    t = 2*np.pi*np.linspace(0, 1, N)
    Y = np.zeros((N, 2))
    Y[:, 0] = 1.1*np.cos(t)
    Y[:, 1] = 1.1*np.sin(2*t)
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    tic = time.time()
    cost1 = DTW(X, Y)['cost']
    time1 = time.time() - tic
    #cost1 = res1['S'][-1, -1]
    print("cost1 = ", cost1, ", time1 = ", time1)

   
    tic = time.time()
    D = getCSM(X, Y)
    cost2 = doDTWGPU(D)
    time2 = time.time() - tic
    print("cost2 = ", cost2, ", time2 = ", time2)


if __name__ == '__main__':
    testTiming()