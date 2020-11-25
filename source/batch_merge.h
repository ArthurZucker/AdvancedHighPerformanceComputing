#ifndef BATCH_MERGE_h
#define BATCH_MERGE_h

__global__ void mergeSmallBatch_k(int *A,int *B, int *M,int a, int b, int m);

#endif