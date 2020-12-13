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


/**
 * @file merge.cu
 * @author Arthur Zucker & Clément Apavou  
 * @date 12 Dec 2020
 * @brief contains all merge kernels
 */


__device__ void merged_path_seq_device(const int *__restrict__ A,const int *__restrict__ B, int *__restrict__ M,const int a, const int b){
    /**
    * Sequential merge for a thread using the code from the article
    * @param A an array of ints to merge with @param B into @param M
    * @param a @param b respective sizes of the arrays
    * @see main()
    * @return Nothing, M is sorted in place
    */
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
    /**
    * Sequential merge using the code from the article
    * @param A an array of ints to merge with @param B into @param M
    * @param a @param b respective sizes of the arrays
    * @see main()
    * @return Nothing, M is sorted in place
    */
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

//_______________________________________________________________________Question 1____________________________________________________________________________________________________
// normal
__global__ void mergedSmall_k(const int *__restrict__ A,const int *__restrict__ B, int *__restrict__ M,const int sA, const int sB, const int sM){
    /**
    * Parallel merge of two sorted arrays of size global size <1024
    * @param A an array of ints to merge with @param B into @param M
    * @param sA, @param sB, @param sM respective sizes of the arrays
    * @return Nothing, M is sorted in place
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

// used shared memory
__global__ void mergeSmall_k_shared(const int *__restrict__ A,const int *__restrict__ B, int *M,const int sA, const int sB, const int sM){
    /**
    * Parallel merge of two sorted arrays of size global size <1024  using shared memeory 
    * @see mergedSmall_k()
    * Threads from same block share this memory 
    * In this case there is only 1 block in the call and at most 1024 threads in the block
    * Here the shared memory is dynamically allocated 
    */

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
    /**
    * Parallel merge of two sorted arrays  of size global size <1024 using texture memory
    * @see mergedSmall_k()
    * Threads from same block share this memory 
    * In this case there is only 1 block in the call and at most 1024 threads in the block
    * Here the shared memory is dynamically allocated 
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
    /**
    * Parallel merge of two sorted arrays  of size global size <1024 using explicit load 
    * @see mergedSmall_k()
    * Threads from same block share this memory 
    * In this case there is only 1 block in the call and at most 1024 threads in the block
    * Here the shared memory is dynamically allocated 
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


//_______________________________________________________________________Question 2____________________________________________________________________________________________________

__device__ void merged_k_ldg(const int *__restrict__ A,const int *__restrict__ B,int *__restrict__ M,int sA, int sB, int sM){
    
    /**
    * Parallel merge wrapper for merge_big_k using ldg load
    * @param A an array of ints to merge with @param B into @param M
    * @param sA, @param sB, @param sM respective sizes of the arrays
    * @see merge_big_k()
    * @return Nothing, M is sorted in place
    */

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
    /**
    * Parallel merge wrapper for merge_big_k 
    * @param A an array of ints to merge with @param B into @param M
    * @param sA, @param sB, @param sM respective sizes of the arrays
    * @see merge_big_k()
    * @return Nothing, M is sorted in place
    */ 
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
    
    /**
    * Parallel path finder. Find the frontier of 0 and 1 in the merging matrix
    * @param A an array of ints to merge with @param B into @param M
    * @param sA, @param sB, @param sM respective sizes of the arrays
    * @param path contains the indexes in A and B at the diagonal point 
    * corresponding to the point were A < B
    * @return Nothing, the path is initialized
    * @note using blocks of 34/64 and 128 is faster
    * Only one thread per block finds entry points.
    * It will write the entry points on the path using the following syntax : 
    *     path[2* block indew] |  path[2* block indew +1] 
    *        ai               |       bi              
    */
    
        
    
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

__global__ void merged_Big_k(const int *__restrict__ A,const int *__restrict__ B,int *__restrict__ M, const int *__restrict__ path, const int m){
    /**
    * Merge two arrays from a given path.
    * @param A an array of ints to merge with @param B into @param M
    * @param sA, @param sB, @param sM respective sizes of the arrays
    * @param path contains the indexes in A and B at the diagonal point 
    * corresponding to the point were A < B
    * @return Nothing, the path is initialized
    * @note Details on the values of path
    *  [ ai            bi   ]      [ ai+1         bi+1  ]    
    *  [  entring point     ]      [     exit point     ] for the subarrays  
    */
    
    int i = blockIdx.x;
    #if VERBOSE == 1
    printf("thread %6.d, has to work on \tA[%6d]->A[%6d]\tB[%6d]->B[%6d]\tM[%6.d]\n",blockDim.x*blockIdx.x + threadIdx.x,path[2*i],path[2*(i+1)],path[2*(i)+1],path[2*(i+1)+1],   blockDim.x*blockIdx.x  );   
    #endif
    // each thread block deals with the same subarrays A and  B
    // each block will have a number of thread equal to the number of entry points
    // working on diagonals
    if(blockDim.x*blockIdx.x+threadIdx.x < m) merged_k(&A[  path[2*i]  ],&B[  path[(2*i)+1]  ], &M[  blockDim.x*blockIdx.x ],    path[  2*(i+1)  ] - path[2*i]     ,    path[2*(i+1)+1] - path[2*i+1]    ,   path[2*(i+1)] - path[2*i]+ path[2*(i+1)+1] - path[2*i+1]     );
}

__global__ void merged_Big_k_naive(const int *__restrict__ A,const int *__restrict__ B,int *__restrict__ M, int *__restrict__ path, const int m){
     /**
    * Naive Merge two arrays from a given path. A Single thread merges an array
    * @param A an array of ints to merge with @param B into @param M
    * @param sA, @param sB, @param sM respective sizes of the arrays
    * @param path contains the indexes in A and B at the diagonal point 
    * corresponding to the point were A < B
    * @return Nothing, the path is initialized
    * @note Each thread block will merge each subarray using the sequential merge
    * this has to be called on the maximum number of threads
    */
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    merged_path_seq_device(&A[path[2*i]],&B[path[2*i+1]], &M[i],path[2*(i+1)]-path[2*i], path[2*(i+1)+1]-path[2*i+1]);

}


void sort_array( int   *hD, int   *hsD,const int sizeD,const int tpb){
     /**
    * Sorts any array M using pervious functions
    * @param tpb number of threads to use in pathBig and mergedBig
    * @param hD constains the initals unsorted array
    * @param hsD will contain the final sorted array
    * @return Nothing, M is sorted on place
    * @note The complexity should be log(sizeD), but it is not
    * this implementation is not efficient since it loops
    */
    int i;
    for( i=1;i<sizeD;i*=2){
        // iterate over the size of the sub arrays that are being sorted
        int *__restrict__ path;
        int nblocks = (2*i+tpb-1)/tpb ;
        cudaMalloc((void **)&path,2*(nblocks+1)*sizeof(int));
        for(int j=0;j<sizeD;j+=2*i){
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