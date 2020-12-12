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
#define VERBOSE 0
using namespace std;
// modified at home 
__device__ void merged_path_seq_device(const int *__restrict__ A,const int *__restrict__ B, int *__restrict__ M,const int a, const int b){
	int m = a+b;
	int i = 0;
	int j = 0;
	while(i+j<m){
		if(i>=a){
			M[i+j]=B[j];
            j++;
        }
		else if(j>=b ||A[i]<B[j]){
			M[i+j]=A[i];
            i++;
        }
		else{
			M[i+j]=B[j];
            j++;
        }
	}
}

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
    // Threads from same block share this memory 
    // In this case there is only 1 block in the call and at most 1024 threads in the block
    // Here the shared memory is dynamically allocated 
    extern __shared__ int shared[]; // dynamic. Extern allows for the host to allocate the memory 
    //__shared__ int shared[1024];  // static
    int i = threadIdx.x;            // only one thread thus Block idX not relevant 
    if(i<sM){
        // Shared: [ s1, s2 , ...., sA , sA+1, ....., sM]
        //         [   A               ,       B        ]
        if (0<=i && i<sA)shared[i] = A[i];          // threads <|A| load A[i] in shared memory 
        else if (sA<=i && i<sM)shared[i] = B[i-sA]; // threads >|A| but <|M| load B[i] in shared memory 
        // offset required, to get i == sM, i-sA = sB last index of shared and B 
        __syncthreads(); // make sure that every thread will have the data
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
// zerocopy
__global__ void mergedSmall_k(const int *__restrict__ A,const int *__restrict__ B, int *__restrict__ M,const int sA, const int sB, const int sM){
    /**
    * A brief description. A more elaborate class description
    * @param A an array of ints to merge with @param B into @param M
    * @param sA,sB,sM respective sizes of the arrays
    * @see Test()
    * @return 
    */
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


//_______________________________________________________________________Question 2____________________________________________________________________________________________________

__device__ void merged_k_ldg(const int *__restrict__ A,const int *__restrict__ B,int *__restrict__ M,int sA, int sB, int sM){
    // wrapper for merge_big_k
    // uses the same algorythm as in previous questions
    int i  = threadIdx.x;
    if(i<sM){
        int2 K;
        int2 P;
        if(i>sA){
            K = {i-sA,sA}; // same sA
            P = {sA,i-sA};
        }
        else{
            K = {0,i};  // 0 replace with Q.y (entry point) 
            P = {i,0};
        }
        while(1){
            int offset = int(abs(K.y-P.y)/2);
            int2 Q = {K.x+offset,K.y-offset};
            // __ldg intrinsic and const __restrict__ garanties the compiler that it is read only
            // thus no aliasing is done 
            // modifier le borne n haut a gauche en bas a droite point d'entrée et de sortie 
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

__device__ void merged_k(const int *__restrict__ A,const int *__restrict__ B,int *__restrict__ M,const int sA, const int sB, const int sM){
    // wrapper for merge_big_k
    // uses the same algorythm as in previous questions
    int i  = threadIdx.x;
    int2 K;
    int2 P;
    if(i>sA){
        K = {i-sA,sA}; // same sA
        P = {sA,i-sA};
    }
    else{
        K = {0,i};  // 0 replace with Q.y (entry point) 
        P = {i,0};
     }
    while(1){
        int offset = int(abs(K.y-P.y)/2);
        int2 Q = {K.x+offset,K.y-offset};
        if(Q.y >= 0 && Q.x <= sB && (Q.y == sA || Q.x == 0 || A[Q.y] > B[Q.x-1])){
            if(Q.x==sB || Q.y==0 || A[Q.y-1]<= B[Q.x]){
                if(Q.y < sA && (Q.x == sB || A[Q.y]<= B[Q.x])){
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

__global__ void pathBig_k (const int *__restrict__ A,const int *__restrict__ B,int *__restrict__ path,const int sA,const int sB,const int sM){
    // using blocks of 34/64 and 128 is faster
    // Only one thread per block finds entry points.
    // It will write the entry points on the path using the following syntax : 
    //     path[2* block indew] |  path[2* block indew +1] 
    //         ai               |       bi                      
    
    if(threadIdx.x == 0){
        int i = blockDim.x*(blockIdx.x);
        int nb_blocks = gridDim.x;              // because last element is at path[2*(GridDim.x-1))] since 195 is last index of thread
        #if VERBOSE == 1
        printf("thread %4d taking care of diagonal %d\n",blockDim.x*blockIdx.x + threadIdx.x,i);
        #endif
        if(i==0){
            path[2*(nb_blocks)]   = sA;
            path[2*(nb_blocks)+1] = sB;
        }
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
                        #if VERBOSE == 1
                        printf("%4d/%d  Entry point found (%d,%d)\n",blockIdx.x,gridDim.x,Q.y,Q.x);
                        #endif
                        path[2*blockIdx.x]   = Q.y;  // writing ai to path
                        path[2*blockIdx.x+1] = Q.x;  // writing bi
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

__global__ void    merged_Big_k(const int *__restrict__ A,const int *__restrict__ B,int *__restrict__ M, const int *__restrict__ path, const int m){
    //           ai            bi          ai+1         bi+1
    //          ( entring point)          (exit point) for the subarrays
    // if load in shared, then each thread will load 1 element from A[path[i]] to A[path[i+2]] 
    // and  B[path[i+1]] to B[path[i+3]]
    int i = blockIdx.x;
    #if VERBOSE == 1
    printf("thread %6.d, has to work on \tA[%6d]->A[%6d]\tB[%6d]->B[%6d]\tM[%6.d]\n",blockDim.x*blockIdx.x + threadIdx.x,path[2*i],path[2*(i+1)],path[2*(i)+1],path[2*(i+1)+1],   blockDim.x*blockIdx.x  );   
    #endif
    // each thread block deals with the same subarrays A and  B
    // each block will have a number of thread equal to the number of entry points
    // working on diagonals
    if(blockDim.x*blockIdx.x+threadIdx.x < m) merged_k(&A[  path[2*i]  ],&B[  path[(2*i)+1]  ], &M[  blockDim.x*blockIdx.x ],    path[  2*(i+1)  ] - path[2*i]     ,    path[2*(i+1)+1] - path[2*i+1]    ,   path[2*(i+1)] - path[2*i]+ path[2*(i+1)+1] - path[2*i+1]     );
}


__global__ void    merged_Big_k_naive(const int *__restrict__ A,const int *__restrict__ B,int *__restrict__ M, int *__restrict__ path, const int m){
    // each thread will use sequentiel merge on his path
    // it is slow 
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    merged_path_seq_device(&A[path[2*i]],&B[path[2*i+1]], &M[i],path[2*i+2]-path[2*i], path[2*i+3]-path[2*i]);

}


void sort_array( int   *hD, int   *hsD,const int sizeM,const int tpb){
    int i;
    for( i=1;i<sizeM;i*=2){
        // iterate over the size of the sub arrays that are being sorted
        int *__restrict__ path;
        int nblocks = (2*i+tpb-1)/tpb ;
        cudaMalloc((void **)&path,2*(nblocks+1)*sizeof(int));
        for(int j=0;j<sizeM;j+=2*i){
            // iterate over the subarrays. 
            if(i>512){ // if the global size of array is > 1024
                
                pathBig_k   <<<nblocks,tpb>>>(&hD[j],&hD[j+i],path,i,i,2*i);
                merged_Big_k<<<nblocks,tpb>>>(&hD[j],&hD[j+i],&hsD[j],path,2*i);
            }
            else{
                // One block is enough to deal with every sub arrays
                mergedSmall_k<<<1,2*i>>>(&hD[j],&hD[j+i],&hsD[j],i,i,2*i);
            }
        }
        // change pointers, memory efficient
        int *ht = hD;   
        hD = hsD;
        hsD = ht;
        cudaFree(path);
    }
}