#ifndef BATCH_MERGE_h
#define BATCH_MERGE_h

__global__ void mergeSmallBatch_k(int *__restrict__ all_M,int *M,int *all_sA, int *all_sB,int d);

#endif