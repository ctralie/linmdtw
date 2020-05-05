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

void c_diag_step(float* d0, float* d1, float* d2, float* csm0, float* csm1, int M, int N, int diagLen, int i, int debug, float* U, float* L, float* UL) {
    int upoff = 0;

    //Other local variables
    int i1, i2, j1, j2; // Endpoints of the diagonal
    int thisi, thisj; // Current indices on the diagonal
    float score, lastscore; // Optimal score and particular score for up/right/left

    //Process each diagonal
    score = -1;
    for (int idx = 0; idx < diagLen; idx++) {
        //Figure out the bounds of this diagonal
        i1 = i;
        j1 = 0;
        upoff = -1;
        if (i1 >= M) {
            i1 = M-1;
            j1 = i - (M-1);
            upoff = 0;
        }
        j2 = i;
        i2 = 0;
        if (j2 >= N) {
            j2 = N-1;
            i2 = i - (N-1);
        }
        //Update each batch
        thisi = i1 - idx;
        thisj = j1 + idx;
        if (thisi >= i2 && thisj <= j2) {
            //Figure out the optimal cost

            //Above
            if (idx + upoff + 1 < N + M - 1 && thisi > 0) {
                lastscore = d1[idx + upoff + 1];
                if (lastscore > -1) {
                    score = lastscore + csm1[idx + upoff + 1];
                }
                if (debug == 1) {
                    U[thisi*N + thisj] = lastscore;
                }
            }

            //Left
            if (idx + upoff >= 0 && thisj > 0) {
                lastscore = d1[idx + upoff];
                if (lastscore > -1) {
                    lastscore += csm1[idx + upoff];
                    if (score == -1 || lastscore < score) {
                        score = lastscore;
                    }
                }
                if (debug == 1) {
                    L[thisi*N + thisj] = lastscore;
                }
            }

            //Diagonal
            if (i1 == M-1 && j1 > 1) {
                upoff = 1;
            }
            if (idx + upoff >= 0 && thisi > 0) {
                lastscore = d0[idx + upoff];
                if (lastscore > -1) {
                    lastscore += csm0[idx + upoff];
                    if (score == -1 || lastscore < score) {
                        score = lastscore;
                    }
                }
                if (debug == 1) {
                    UL[thisi*N + thisj] = lastscore;
                }
            }
            
            if (score == -1) {
                score = 0;
            }
        }
        d2[idx] = score;
    }
}
