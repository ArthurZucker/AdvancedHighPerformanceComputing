#ifndef MERGE_h
#define MERGE_h
extern texture <int> texture_referenceA ;
extern texture <int> texture_referenceB ;
void merged_path_seq                 (int *A,int *B, int *M,int a, int b);
__global__ void mergedSmall_k_texture(int *A,int a, int b, int m);
__global__ void mergedSmall_k        (int *A,int *B, int *M,int a, int b, int m);
__global__ void merged_Big_k  (int *A,int *B, int *M,int a, int b, int m);
__global__ void pathBig_k        (int *A,int *B, int *M,int a, int b, int m);
#endif