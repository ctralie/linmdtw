cdef extern from "DTW.cpp":
	float c_dtw(float* X, float* Y, int* P, int M, int N, int d, int debug, float* U, float* L, float* UL)