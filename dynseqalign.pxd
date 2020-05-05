cdef extern from "DTW.cpp":
    float c_dtw(float* X, float* Y, int* P, int M, int N, int d, int debug, float* U, float* L, float* UL)

cdef extern from "DTW.cpp":
    void c_diag_step(float* d0, float* d1, float* d2, float* csm0, float* csm1, int M, int N, int diagLen, int i, int debug, float* U, float* L, float* UL)