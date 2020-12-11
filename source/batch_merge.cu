#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <assert.h>
#include <time.h>
#include "batch_merge.h"

// function mergeSmallBatch_k takes a big array all_M containing (Ai and Bi) like this : all_M = (A1|B1|...|AN|BN)
// it returns M the array of all_M with arrays Ai and Bi merge and sort
// all_sA contains all size of different A (all_sA[0]= size(A0))
// all_sB contains all size of different B (all_sB[0]= size(B0))
// we stocked sizes of Ai and Bi because |Ai|!=|Bi|
// d is the number of element that there is in the array Mi, i.e all_sA[i]+all_sB[i] = d (|Ai|+|Bi|=d)
// size of M is d*N 
__global__ void mergeSmallBatch_k(int *__restrict__ all_M,int *M,int *all_sA, int *all_sB,int d){
    
    int tidx = threadIdx.x%d; // to know which element of the below-array (Ai) it treats
    int Qt = (threadIdx.x-tidx)/d;
    int gbx = Qt + blockIdx.x*(blockDim.x/d);// which array it treats

    // take the good size 
    int sA = all_sA[gbx]; 
    int sB = all_sB[gbx];
    // take the good arrays A and B
    int *A = &all_M[gbx*d];
    int *B = &all_M[gbx*d+sA];
   
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(tidx<d){
        int2 K;
        int2 P;
        if(tidx>sA){
            K = {tidx-sA,sA};
            P = {sA,tidx-sA};
        }
        else{
            K = {0,tidx};
            P = {tidx,0};
        }
        while(1){
            int offset = int(abs(K.y-P.y)/2);
            int2 Q = {K.x+offset,K.y-offset};

            if(Q.y >= 0 && Q.x <= sB && (Q.y == sA || Q.x == 0 || A[Q.y] > B[Q.x-1])){
                if(Q.x==sB || Q.y==0 || A[Q.y-1]<=B[Q.x]){
                   if(Q.y < sA && (Q.x == sB || A[Q.y]<=B[Q.x])){
                        M[i] = A[Q.y];
                   }
                   else{
                        M[i] = B[Q.x];
                   }
                   break;
                }
                else{
                   K = {Q.x+1,Q.y-1};
                }
            }
            else{
                P = {Q.x-1,Q.y+1};
            }
        }
    }
}

// mergeSmallBatch using ldg 
__global__ void mergeSmallBatch_k_ldg(int *__restrict__ all_M,int *M,int *all_sA, int *all_sB,int d){
    int tidx = threadIdx.x%d;
    int Qt = (threadIdx.x-tidx)/d;
    int gbx = Qt + blockIdx.x*(blockDim.x/d);

    int sA = all_sA[gbx];
    int sB = all_sB[gbx];
    int *A = &all_M[gbx*d];
    int *B = &all_M[gbx*d+sA];
   
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(tidx<d){
        int2 K;
        int2 P;
        if(tidx>sA){
            K = {tidx-sA,sA};
            P = {sA,tidx-sA};
        }
        else{
            K = {0,tidx};
            P = {tidx,0};
        }
        while(1){
            int offset = int(abs(K.y-P.y)/2);
            int2 Q = {K.x+offset,K.y-offset};
            // __ldg intrinsic and const __restrict__ garanties the compiler that it is read only
            // thus no aliasing is done
            if(Q.y >= 0 && Q.x <= sB && (Q.y == sA || Q.x == 0 || __ldg(&A[Q.y]) > __ldg(&B[Q.x-1]))){
                if(Q.x==sB || Q.y==0 || __ldg(&A[Q.y-1])<=__ldg(&B[Q.x])){
                   if(Q.y < sA && (Q.x == sB || __ldg(&A[Q.y])<=__ldg(&B[Q.x]))){
                        M[i] = __ldg(&A[Q.y]);
                   }
                   else{
                        M[i] = __ldg(&B[Q.x]);
                   }
                   break;
                }
                else{
                   K = {Q.x+1,Q.y-1};
                }
            }
            else{
                P = {Q.x-1,Q.y+1};
            }
        }
    }
}

// mergeSmallBatch using shared memory 
__global__ void mergeSmallBatch_k_shared(int *__restrict__ all_M,int *M,int *all_sA, int *all_sB,int d){
    int tidx = threadIdx.x%d;
    int Qt = (threadIdx.x-tidx)/d;
    int gbx = Qt + blockIdx.x*(blockDim.x/d);
    
    extern __shared__ int shared[];

    int sA = all_sA[gbx];
    int sB = all_sB[gbx];
    int *A = &all_M[gbx*d];
    int *B = &all_M[gbx*d+sA];
   
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(tidx<d){
        if (0<=tidx && tidx<sA)shared[tidx] = A[tidx];          
        else if (sA<=tidx && tidx<d)shared[tidx] = B[tidx-sA];
        __syncthreads();
        int2 K;
        int2 P;
        if(tidx>sA){
            K = {tidx-sA,sA};
            P = {sA,tidx-sA};
        }
        else{
            K = {0,tidx};
            P = {tidx,0};
        }
        while(1){
            int offset = int(abs(K.y-P.y)/2);
            int2 Q = {K.x+offset,K.y-offset};
            // to access A[i]: shared[i]
            // to access B[i]: shared[sA+i]
            if(Q.y >= 0 && Q.x <= sB && (Q.y == sA || Q.x == 0 || shared[Q.y] > shared[sA+Q.x-1])){
                if(Q.x==sB || Q.y==0 || shared[Q.y-1]<=shared[sA+Q.x]){
                   if(Q.y < sA && (Q.x == sB || shared[Q.y]<=shared[sA+Q.x])){
                        M[i] = shared[Q.y];
                   }
                   else{
                        M[i] = shared[sA+Q.x];
                   }
                   break;
                }
                else{
                   K = {Q.x+1,Q.y-1};
                }
            }
            else{
                P = {Q.x-1,Q.y+1};
            }
        }
    }
}

// function SortSmallBatch_k takes a big array all_M containing (Ai and Bi) like this : all_M = (A1|B1|...|AN|BN)
// it returns M the array of all_M with arrays Ai and Bi merge and sort
// si is the size of the arrays A and B (we fixed |A|=|B|)
__global__ void SortSmallBatch_k(int *__restrict__ all_M,int *M,int si,int d){
    int tidx = threadIdx.x%d;
    int Qt = (threadIdx.x-tidx)/d;
    int gbx = Qt + blockIdx.x*(blockDim.x/d);

    int sA = si;
    int sB = si;
    int *A = &all_M[gbx*d];
    int *B = &all_M[gbx*d+sA];
   
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(tidx<d){
        int2 K;
        int2 P;
        if(tidx>sA){
            K = {tidx-sA,sA};
            P = {sA,tidx-sA};
        }
        else{
            K = {0,tidx};
            P = {tidx,0};
        }
        while(1){
            int offset = int(abs(K.y-P.y)/2);
            int2 Q = {K.x+offset,K.y-offset};

            if(Q.y >= 0 && Q.x <= sB && (Q.y == sA || Q.x == 0 || A[Q.y] > B[Q.x-1])){
                if(Q.x==sB || Q.y==0 || A[Q.y-1]<=B[Q.x]){
                   if(Q.y < sA && (Q.x == sB || A[Q.y]<=B[Q.x])){
                        M[i] = A[Q.y];
                   }
                   else{
                        M[i] = B[Q.x];
                   }
                   break;
                }
                else{
                   K = {Q.x+1,Q.y-1};
                }
            }
            else{
                P = {Q.x-1,Q.y+1};
            }
        }
    }
}

// SortSmallBatch using ldg
__global__ void SortSmallBatch_k_ldg(int *__restrict__ all_M,int *M,int si,int d){
    int tidx = threadIdx.x%d;
    int Qt = (threadIdx.x-tidx)/d;
    int gbx = Qt + blockIdx.x*(blockDim.x/d);

    int sA = si;
    int sB = si;
    int *A = &all_M[gbx*d];
    int *B = &all_M[gbx*d+sA];
   
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(tidx<d){
        int2 K;
        int2 P;
        if(tidx>sA){
            K = {tidx-sA,sA};
            P = {sA,tidx-sA};
        }
        else{
            K = {0,tidx};
            P = {tidx,0};
        }
        while(1){
            int offset = int(abs(K.y-P.y)/2);
            int2 Q = {K.x+offset,K.y-offset};
            // __ldg intrinsic and const __restrict__ garanties the compiler that it is read only
            // thus no aliasing is done
            if(Q.y >= 0 && Q.x <= sB && (Q.y == sA || Q.x == 0 || __ldg(&A[Q.y]) > __ldg(&B[Q.x-1]))){
                if(Q.x==sB || Q.y==0 || __ldg(&A[Q.y-1])<=__ldg(&B[Q.x])){
                   if(Q.y < sA && (Q.x == sB || __ldg(&A[Q.y])<=__ldg(&B[Q.x]))){
                        M[i] = __ldg(&A[Q.y]);
                   }
                   else{
                        M[i] = __ldg(&B[Q.x]);
                   }
                   break;
                }
                else{
                   K = {Q.x+1,Q.y-1};
                }
            }
            else{
                P = {Q.x-1,Q.y+1};
            }
        }
    }
}