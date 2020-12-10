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

__global__ void mergeSmallBatch_k(int **__restrict__ A,int **__restrict__ B,int **M,int *sA, int *sB, int sM){
    int tidx = threadIdx.x%sM;
    // printf("tidx = %d\n",tidx);
    int Qt = (threadIdx.x-tidx)/sM;
    // printf("Qt = %d\n",Qt);
    int gbx = Qt + blockIdx.x*(blockDim.x/sM);
    // printf("gbx = %d\n",gbx);
    if(tidx<sM){
        int2 K;
        int2 P;
        if(tidx>sA[tidx]){
            K = {tidx-sA[tidx],sA[tidx]};
            P = {sA[tidx],tidx-sA[tidx]};
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
            if(Q.y >= 0 && Q.x <= sB[tidx] && (Q.y == sA[tidx] || Q.x == 0 || __ldg(&A[tidx][Q.y]) > __ldg(&B[tidx][Q.x-1]))){
                if(Q.x==sB[tidx] || Q.y==0 || __ldg(&A[tidx][Q.y-1])<=__ldg(&B[tidx][Q.x])){
                   if(Q.y < sA[tidx] && (Q.x == sB[tidx] || __ldg(&A[tidx][Q.y])<=__ldg(&B[tidx][Q.x]))){
                        M[tidx][tidx] = __ldg(&A[tidx][Q.y]);
                   }
                   else{
                        M[tidx][tidx] = __ldg(&B[tidx][Q.x]);
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
    M[0][304]=2513;
    printf("%d",M[0][304]);
}
