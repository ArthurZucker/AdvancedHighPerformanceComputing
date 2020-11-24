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
__global__ void mergedSmall_k_texture(int *__restrict__ M,const int sA, const int sB, const int sM){
    int i = threadIdx.x;
    //printf("");
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
            int AQy_1 = tex1Dfetch( texture_referenceA, (Q.y-1));
            int AQy   = tex1Dfetch( texture_referenceA, Q.y);
            int BQx_1 = tex1Dfetch( texture_referenceB, (Q.x-1));
            int BQx   = tex1Dfetch( texture_referenceB, Q.x);
            if(Q.y >= 0 && Q.x <= sB && (Q.y == sA || Q.x == 0 || AQy > BQx_1)){
                if(Q.x==sB || Q.y==0 || AQy_1<=BQx){
                   if(Q.y < sA && (Q.x == sB || AQy<=BQx)){
                        M[i] = AQy;
                   }
                   else{
                        M[i] = BQx;
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
            int AQy_1 = __ldg(&A[Q.y-1]);
            int AQy   = __ldg(&A[Q.y]);
            int BQx_1 = __ldg(&B[Q.x-1]);
            int BQx   = __ldg(&B[Q.x]);
            if(Q.y >= 0 && Q.x <= sB && (Q.y == sA || Q.x == 0 || AQy > BQx_1)){
                if(Q.x==sB || Q.y==0 || AQy_1<=BQx){
                   if(Q.y < sA && (Q.x == sB || AQy<=BQx)){
                        M[i] = AQy;
                   }
                   else{
                        M[i] = BQx;
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
            int AQy_1 = AQ.x;
            int AQy   = AQ.y;
            int BQx_1 = BQ.x;
            int BQx   = BQ.y;
            if(Q.y >= 0 && Q.x <= sB && (Q.y == sA || Q.x == 0 || AQy > BQx_1)){
                if(Q.x==sB || Q.y==0 || AQy_1<=BQx){
                   if(Q.y < sA && (Q.x == sB || AQy<=BQx)){
                        M[i] = AQy;
                   }
                   else{
                        M[i] = BQx;
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
            int AQy_1 = A[Q.y-1];
            int AQy   = A[Q.y];
            int BQx_1 = B[Q.x-1];
            int BQx   = B[Q.x];
            if(Q.y >= 0 && Q.x <= sB && (Q.y == sA || Q.x == 0 || AQy > BQx_1)){
                if(Q.x==sB || Q.y==0 || AQy_1<=BQx){
                   if(Q.y < sA && (Q.x == sB || AQy<=BQx)){
                        M[i] = AQy;
                   }
                   else{
                        M[i] = BQx;
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
    if(i<=sM){
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
            int AQy_1 = A[Q.y-1];
            int AQy   = A[Q.y];
            int BQx_1 = B[Q.x-1];
            int BQx   = B[Q.x];
            if(Q.y >= 0 && Q.x <= sB && (Q.y == sA || Q.x == 0 || AQy > BQx_1)){
                if(Q.x==sB || Q.y==0 || AQy_1<=BQx){
                   if(Q.y < sA && (Q.x == sB || AQy<=BQx)){
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
    if(i<=sM){
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
            int AQy_1 = __ldg(&A[Q.y-1]);
            int AQy   = __ldg(&A[Q.y]);
            int BQx_1 = __ldg(&B[Q.x-1]);
            int BQx   = __ldg(&B[Q.x]);
            if(Q.y >= 0 && Q.x <= sB && (Q.y == sA || Q.x == 0 || AQy > BQx_1)){
                if(Q.x==sB || Q.y==0 || AQy_1<=BQx){
                   if(Q.y < sA && (Q.x == sB || AQy<=BQx)){
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
    int p = path[i];
    M[i] = p>0? B[p] : A[-p]; // if path[i] == 1 then M[i] = B[i]
    
}

__global__ void    merged_Big_k_ldg(const int *__restrict__ A,const int *__restrict__ B,int *__restrict__ M, int *__restrict__ path, const int m){
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    int p = __ldg(&path[i]);
    M[i] = p>0? __ldg(&B[p]) : __ldg(&A[-p]); // if path[i] == 1 then M[i] = B[i]
    
}


