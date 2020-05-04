# distutils: language = c++

import numpy
cimport numpy
cimport dynseqalign
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def DTW(numpy.ndarray[float,ndim=2,mode="c"] X not None, numpy.ndarray[float,ndim=2,mode="c"] Y not None, int debug):
    assert(X.shape[1] == Y.shape[1])
    cdef int M = X.shape[0]
    cdef int N = Y.shape[0]
    cdef int d = X.shape[1]
    cdef int[:, :] P = numpy.zeros((M, N), dtype='int32')
    cdef float[:, :,] U 
    cdef float[:, :,] L
    cdef float[:, :,] UL
    if debug == 1:
        U = numpy.zeros((M, N), dtype='float32')
        L = numpy.zeros((M, N), dtype='float32')
        UL = numpy.zeros((M, N), dtype='float32')
    else:
        U = numpy.zeros((1, 1), dtype='float32')
        L = numpy.zeros((1, 1), dtype='float32')
        UL = numpy.zeros((1, 1), dtype='float32')
    cost = dynseqalign.c_dtw(&X[0,0], &Y[0,0], &P[0,0], M, N, d, debug, &U[0, 0], &L[0, 0], &UL[0, 0])
    ret = {'cost':cost, 'P':P}
    if debug == 1:
        ret['U'] = U
        ret['L'] = L
        ret['UL'] = UL
    return ret