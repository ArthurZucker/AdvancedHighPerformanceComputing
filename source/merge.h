#ifndef MERGE_h
#define MERGE_h
extern texture <int> texture_referenceA ;
extern texture <int> texture_referenceB ;
extern __device__ bool *__restrict__ path;

__device__ void merged_path_seq_device(const int *__restrict__ A,const int *__restrict__ B, int *__restrict__ M,const int a, const int b);
__global__ void mergedSmall_k_texture(int * __restrict__ A,const int a, int b, const int m);
__global__ void mergedSmall_k_ldg    (const int *__restrict__ A,const int *__restrict__ B, int *M,const int a, const int b, const int m);
__global__ void mergedSmall_k        (const int *__restrict__ A,const int *__restrict__ B, int *__restrict__ M,const int a, const int b, const int m);
__global__ void   pathBig_k_naive          (const int *__restrict__ A,const int *__restrict__ B,int *__restrict__ path ,const int a, const int b, const int m);
__global__ void merged_Big_k_naive         (const int *__restrict__ A,const int *__restrict__ B, int *__restrict__ M,int *__restrict__ path,const int m);
void merged_path_seq                 (const int *__restrict__ A,const int *__restrict__ B, int *__restrict__ M,const int a, const int b);
__global__ void pathBig_k_naive_ldg        (const int *__restrict__ A,const int *__restrict__ B,int *__restrict__ path,const int sA,const int sB,const int sM);
__device__ void merged_k_ldg(const int *__restrict__ A,const int *__restrict__ B,int *__restrict__ M,int sA, int sB, int sM);
__global__ void mergeSmall_k_shared  (const int *__restrict__ A,const int *__restrict__ B, int *M,const int a, const int b, const int m);
__global__ void pathBig_k (const int *__restrict__ A,const int *__restrict__ B,int *__restrict__ path,const int sA,const int sB,const int sM);
__global__ void    merged_Big_k(const int *__restrict__ A,const int *__restrict__ B,int *__restrict__ M, const int *__restrict__ path, const int m);
void sort_array( int  *hD, int *hsD,const int sizeM,const int threads_per_blocks);
#endif


