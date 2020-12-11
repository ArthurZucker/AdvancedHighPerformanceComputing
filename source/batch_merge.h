#ifndef BATCH_MERGE_h
#define BATCH_MERGE_h

// __global__ void mergeSmallBatch_k(int **__restrict__ A,int **__restrict__ B,int **M,int *sA, int *sB, int sM);
__global__ void mergeSmallBatch_k(int *__restrict__ all_M,int *M,int *sA, int *sB,int d);

#endif