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

__global__ void mergeSmallBatch_k(int *__restrict__ all_M,int *M,int *all_sA, int *all_sB,int d){
    int tidx = threadIdx.x%d;
    int Qt = (threadIdx.x-tidx)/d;
    int gbx = Qt + blockIdx.x*(blockDim.x/d);
    // printf("threadIdx = %d; blockIdx = %d; tidx = %d, Qt = %d, gbx = %d, combined=%d\n",threadIdx.x,blockIdx.x,tidx,Qt,gbx,tidx+gbx*d);
    // for(int i = 0;i<6;i++){ 
    //     printf("all_size_A[%d]=%d, all_size_B[%d]=%d \n",i,sA[i],i,sB[i]);
    // }
    int blx = blockIdx.x;
    int sA = all_sA[gbx];
    int sB = all_sB[gbx];
    // printf("sB[%d]=%d\n",blx,sB[blx]);
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
                        // printf("index globale = %d, threadIdx = %d; blockIdx = %d; tidx = %d, Qt = %d, gbx = %d, __ldg(&A[%d]) = %d\n",i,threadIdx.x,blockIdx.x,tidx,Qt,gbx,Q.y,__ldg(&A[Q.y]));
                   }
                   else{
                        M[i] = __ldg(&B[Q.x]);
                        // printf("index globale = %d, threadIdx = %d; blockIdx = %d; tidx = %d, Qt = %d, gbx = %d, __ldg(&B[%d]) = %d\n",i,threadIdx.x,blockIdx.x,tidx,Qt,gbx,Q.x,__ldg(&B[Q.x]));
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
