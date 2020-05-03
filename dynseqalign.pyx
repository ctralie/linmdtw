# distutils: language = c++

import numpy
cimport numpy
cimport dynseqalign
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def DTW(numpy.ndarray[float,ndim=2,mode="c"] X not None, numpy.ndarray[float,ndim=2,mode="c"] Y not None):
    assert(X.shape[1] == Y.shape[1])
    cdef int M = X.shape[0]
    cdef int N = Y.shape[0]
    cdef int d = X.shape[1]
    cdef int[:, :] P = numpy.zeros((M, N), dtype='int32')
    res = dynseqalign.c_dtw(&X[0,0], &Y[0,0], &P[0,0], M, N, d)
    return res, P