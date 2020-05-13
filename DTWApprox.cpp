#include <math.h>
#include <float.h>

#define LEFT 0
#define UP 1
#define DIAG 2

void c_fastdtw_dynstep(float* X, float* Y, int d, int* I, int* J, int* left, int* up, int* diag, float* S, int* P, int N) {
    float dist;
    for (int idx = 0; idx < N; idx++) {
        // Step 1: Compute the Euclidean distance
        dist = 0.0;
        for (int k = 0; k < d; k++) {
            double diff = (X[I[idx]*d + k] - Y[J[idx]*d + k]);
            dist += diff*diff;
        }
        dist = sqrt(dist);

        // Step 2: Do dynamic progamming step
        float score = -1;
        if (idx == 0) {
            score = 0;
        }
        else {
            // Left
            float leftScore = -1;
            if (left[idx] >= 0) {
                leftScore = S[left[idx]];
            }
            // Up
            float upScore = -1;
            if (up[idx] >= 0) {
                upScore = S[up[idx]];
            }
            // Diag
            float diagScore = -1;
            if (diag[idx] >= 0) {
                diagScore = S[diag[idx]];
            }

            if (leftScore > -1) {
                score = leftScore;
                P[idx] = LEFT;
            }
            if (upScore > -1 && (upScore < score || score == -1)) {
                score = upScore;
                P[idx] = UP;
            }
            if (diagScore > -1 && (diagScore <= score || score == -1)) {
                score = diagScore;
                P[idx] = DIAG;
            }
        }
        S[idx] = score + dist;

    }
}
