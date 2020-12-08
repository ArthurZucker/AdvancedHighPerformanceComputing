#ifndef BATCH_MERGE_h
#define BATCH_MERGE_h

__global__ void mergeSmallBatch_k(int **__restrict__ A,int **__restrict__ B,int **__restrict__ M,int *sA, int *sB, int sM);

#endif