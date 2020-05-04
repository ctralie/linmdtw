#include <math.h>
#include <float.h>

#define LEFT 0
#define UP 1
#define DIAG 2

float c_dtw(float* X, float* Y, int* P, int M, int N, int d, int debug, float* U, float* L, float* UL) {
    float* S = (float*)malloc(M*N*sizeof(float));
    float dist;
    if (debug == 1) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                U[i*N + j] = -1;
                L[i*N + j] = -1;
                UL[i*N + j] = -1;
            }
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            // Step 1: Compute the Euclidean distance
            dist = 0.0;
            for (int k = 0; k < d; k++) {
                double diff = (X[i*d + k] - Y[j*d + k]);
                dist += diff*diff;
            }
            dist = sqrt(dist);

            // Step 2: Do dynamic proramming step
            float left = FLT_MAX, up = FLT_MAX, diag = FLT_MAX;
            if (i > 0) {
                up = S[(i-1)*N + j];
                if (debug == 1) {
                    U[i*N + j] = up; 
                }
            }
            if (j > 0) {
                left = S[i*N + (j-1)];
                if (debug == 1) {
                    L[i*N + j] = left;
                }
            }
            if (i > 0 && j > 0) {
                diag = S[(i-1)*N + (j-1)];
                if (debug == 1) {
                    UL[i*N + j] = diag;
                }
            }
            if (i == 0 && j == 0) {
                diag = 0;
            }
            float min = left;
            P[i*N + j] = LEFT;
            if (up < min) {
                min = up;
                P[i*N + j] = UP;
            }
            if (diag < min) {
                min = diag;
                P[i*N + j] = DIAG;
            }
            S[i*N + j] = min + dist;
        }
    }
    dist = S[M*N-1];
    free(S);
    return dist;
}