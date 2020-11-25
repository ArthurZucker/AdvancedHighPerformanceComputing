/**************************************************************
This code is an implementation of the merging of two arrays
as describes in the subject
Both the sequential and parralele versions will be detailed in
order to asses the performances
***************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <assert.h>
#include <time.h>
#include "merge.h"
using namespace std;
void merged_path_seq(const int *__restrict__ A,const int *__restrict__ B, int *__restrict__ M,const int a, const int b){
	int m = a+b;
	int i = 0;
	int j = 0;
	while(i+j<m){
		if(i>=a){
			M[i+j]=B[j];
			j++;}
		else if(j>=b ||A[i]<B[j]){
			M[i+j]=A[i];
			i++;}
		else{
			M[i+j]=B[j];
			j++;}
	}
}

// used shared memory
__global__ void mergeSmall_k_shared(const int *__restrict__ A,const int *__restrict__ B, int *M,const int sA, const int sB, const int sM){
    extern __shared__ int shared[];
    //__shared__ int shared[1024];
    int i = threadIdx.x;
    if(i<sM){
        if (0<=i && i<sA)shared[i] = A[i];
        else if (sA<=i && i<sM)shared[i] = B[i-sA];
        __syncthreads();
        int2 K;
        int2 P;
        if(i>sA){
            K = {i-sA,sA};
            P = {sA,i-sA};
        }
        else{
            K = {0,i};
            P = {i,0};
        }
        while(1){
            int offset = int(abs(K.y-P.y)/2);
            int2 Q = {K.x+offset,K.y-offset};
            if(Q.y >= 0 && Q.x <= sB && (Q.y == sA || Q.x == 0 || shared[Q.y] > shared[sA+Q.x-1])){
                //AQy_1 = shared[Q.y-1];
                //BQx   = shared[sA+Q.x];
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
// used texture memory
__global__ void mergedSmall_k_texture(int *__restrict__ M,const int sA, const int sB, const int sM){
    int i = threadIdx.x;
    if(i<sM){
        int2 K;
        int2 P;
        if(i>sA){
            K = {i-sA,sA};
            P = {sA,i-sA};
        }
        else{
            K = {0,i};
            P = {i,0};
        }
        while(1){
            int offset = int(abs(K.y-P.y)/2);
            int2 Q = {K.x+offset,K.y-offset};
            if(Q.y >= 0 && Q.x <= sB && (Q.y == sA || Q.x == 0 ||
            tex1Dfetch( texture_referenceA, Q.y    ) >
            tex1Dfetch( texture_referenceB, Q.x-1  )))
            {
                if(Q.x==sB || Q.y==0 || tex1Dfetch( texture_referenceA, (Q.y-1))<=tex1Dfetch( texture_referenceB, Q.x)){
                   if(Q.y < sA && (Q.x == sB || tex1Dfetch( texture_referenceA, Q.y)<=tex1Dfetch( texture_referenceB, Q.x))){
                        M[i] = tex1Dfetch( texture_referenceA, Q.y);
                   }
                   else{
                        M[i] = tex1Dfetch( texture_referenceB, Q.x);
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
// used ldg
__global__ void mergedSmall_k_ldg(const int *__restrict__ A,const int *__restrict__ B,int *__restrict__ M,int sA, int sB, int sM){
    int i = threadIdx.x;
    if(i<sM){
        int2 K;
        int2 P;
        if(i>sA){
            K = {i-sA,sA};
            P = {sA,i-sA};
        }
        else{
            K = {0,i};
            P = {i,0};
        }
        while(1){
            int offset = int(abs(K.y-P.y)/2);
            int2 Q = {K.x+offset,K.y-offset};
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
// used ldg (but doesnt work)
__global__ void mergedSmall_k_ldg2(const int *__restrict__ A,const int *__restrict__ B, int *__restrict__ M,const int sA, const int sB, const int sM){
    int i = threadIdx.x;
    if(i<sM){
        int2 K;
        int2 P;
        if(i>sA){
            K = {i-sA,sA};
            P = {sA,i-sA};
        }
        else{
            K = {0,i};
            P = {i,0};
        }
        while(1){
            int offset = int(abs(K.y-P.y)/2);
            int2 Q = {K.x+offset,K.y-offset};
            // int2 could load Qy1 and Qy but does not work here... illegal acces
            int2 AQ = __ldg((int2 *) &A[Q.y-1]);
            int2 BQ = __ldg((int2 *) &B[Q.x-1]);
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
// zerocopy
__global__ void mergedSmall_k(const int *__restrict__ A,const int *__restrict__ B, int *__restrict__ M,const int sA, const int sB, const int sM){
    int i = threadIdx.x;
    if(i<sM){
        int2 K;
        int2 P;
        if(i>sA){
            K = {i-sA,sA};
            P = {sA,i-sA};
        }
        else{
            K = {0,i};
            P = {i,0};
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

__global__ void pathBig_k (const int *__restrict__ A,const int *__restrict__ B,int *__restrict__ path,const int sA,const int sB,const int sM){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<sM){
        int2 K;
        int2 P;
        if(i>sA){
            K = {i-sA,sA};
            P = {sA,i-sA};
        }
        else{
            K = {0,i};
            P = {i,0};
        }
        while(1){
            int offset = int(abs(K.y-P.y)/2);
            int2 Q = {K.x+offset,K.y-offset};
            if(Q.y >= 0 && Q.x <= sB && (Q.y == sA || Q.x == 0 || A[Q.y] > B[Q.x-1])){
                if(Q.x==sB || Q.y==0 || A[Q.y-1]<=B[Q.x]){
                   if(Q.y < sA && (Q.x == sB || A[Q.y]<=B[Q.x])){
                        path[i] = -Q.y; // 0 means I take A
                   }
                   else{
                        path[i] = Q.x; // 1 means I take B
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


__global__ void pathBig_k_ldg (const int *__restrict__ A,const int *__restrict__ B,int *__restrict__ path,const int sA,const int sB,const int sM){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<sM){
        int2 K;
        int2 P;
        if(i>sA){
            K = {i-sA,sA};
            P = {sA,i-sA};
        }
        else{
            K = {0,i};
            P = {i,0};
        }
        while(1){
            int offset = int(abs(K.y-P.y)/2);
            int2 Q = {K.x+offset,K.y-offset};
            if(Q.y >= 0 && Q.x <= sB && (Q.y == sA || Q.x == 0 || __ldg(&A[Q.y]) > __ldg(&B[Q.x-1]))){
                if(Q.x==sB || Q.y==0 || __ldg(&A[Q.y-1])<=__ldg(&B[Q.x])){
                   if(Q.y < sA && (Q.x == sB || __ldg(&A[Q.y])<=__ldg(&B[Q.x]))){
                        path[i] = -Q.y; // 0 means I take A
                   }
                   else{
                        path[i] = Q.x;  // 1 means I take B
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

__global__ void    merged_Big_k(const int *__restrict__ A,const int *__restrict__ B,int *__restrict__ M, int *__restrict__ path, const int m){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<m){
    int p = path[i];
     M[i] = p>0? B[p] : A[-p]; // if path[i] == 1 then M[i] = B[i]
    }

}

__global__ void    merged_Big_k_ldg(const int *__restrict__ A,const int *__restrict__ B,int *__restrict__ M, int *__restrict__ path, const int m){
    
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i<m){
    int p = __ldg(&path[i]);
    M[i] = p>0? __ldg(&B[p]) : __ldg(&A[-p]); // if path[i] == 1 then M[i] = B[i]
    }

}

