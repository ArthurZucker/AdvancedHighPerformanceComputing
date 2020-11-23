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
#include "batch_merge.h"
#include "utils.h"
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))
using namespace std;
#define TEXTURE 1 //set to 0 to use normal memory, else it will use texture memory for A and B
texture <int> texture_referenceA ;
texture <int> texture_referenceB ;

/*
TO DO : 
 - implement using ldg  avec restricted__  et int4 qui contient 4 int, read only memory 
 - mergeBig_k
 - pathBig_k
*/


int main(int argc, char* argv[]) {
    //___________ Basic initialisation ___________
	srand((unsigned int)time(NULL));
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Max Grid size: %dx%d\n",  prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("Max Thread Dim: %d,%d,%d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max Thread per blocks: %d\n", prop.maxThreadsPerBlock);
	}
	cudaSetDevice(0);
    //____________________________________________
    
    //___________ Variable declaration ___________
    int sizeA;
    int sizeB;
    
    if (argc < 3) {sizeA = rand()%1024;sizeB = rand()%(1024-sizeA);} // If no arguments are provided, set random sizes
    else{sizeA=atoi(argv[1]);sizeB=atoi(argv[2]);}
    int sizeM = sizeA+sizeB;
    printf("|A| = %d, |B| = %d, |M| = %d\n",sizeA,sizeB,sizeM);
    
    int *hostA;
    int *hostB;
    int *hostM;
    
    int *seqM = (int *) malloc(sizeM*sizeof(int));
    int *A = (int *) malloc(sizeA*sizeof(int));
    int *B = (int *) malloc(sizeB*sizeof(int));
    A[0]=rand()%20;
    B[0]=rand()%20;
    for(int i=1;i<sizeA;i++){A[i]=A[i-1]+rand()%20+1;}
    for(int i=1;i<sizeB;i++){B[i]=B[i-1]+rand()%20+1;}
    
    //___________ TO DO: explain texture memory ___________
    #if TEXTURE == 1
    testCUDA(cudaMalloc((void **)&hostA,sizeA*sizeof(int)));
    testCUDA(cudaMalloc((void **)&hostB,sizeB*sizeof(int)));
    
    testCUDA(cudaMemcpy(hostA, A, sizeA*sizeof(int), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(hostB, B, sizeB*sizeof(int), cudaMemcpyHostToDevice));
    
    testCUDA (cudaBindTexture(0,texture_referenceA, hostA,sizeA*sizeof(int)));
    testCUDA (cudaBindTexture(0,texture_referenceB, hostB,sizeB*sizeof(int)));
    #else
    testCUDA(cudaHostAlloc(&hostA,sizeA*sizeof(int),cudaHostAllocWriteCombined));
    testCUDA(cudaHostAlloc(&hostB,sizeB*sizeof(int),cudaHostAllocWriteCombined));
    hostA[0]=rand()%20;
    hostB[0]=rand()%20;
    for(int i=1;i<sizeA;i++){hostA[i]=hostA[i-1]+rand()%20+1;}
    for(int i=1;i<sizeB;i++){hostB[i]=hostB[i-1]+rand()%20+1;}
    #endif
        // WC memory can be transferred across the PCI Express bus more quickly on some system configurations, but cannot be read 
        // efficiently by most CPUs. WC memory is a good option for buffers that will be written by the CPU and read by the device 
        // via mapped pinned memory or host->device transfers.
    testCUDA(cudaHostAlloc(&hostM,sizeM*sizeof(int),cudaHostAllocMapped)); // in order to do zero copy
    // alternative for M : testCUDA(cudaMalloc(&hostM,sizeM*sizeof(int)));
    
    //_______________ Sequential _________________
    printf("_______________ Sequential _________________\n");
    printf("Starting sequencial code: \n");
    clock_t begin = clock();
    merged_path_seq(A,B,seqM,sizeA,sizeB);
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("elapsed time : %f\n",time_spent);
    cout<<"Check sorted : "<<is_sorted(seqM,sizeM)<<endl;
    //____________________________________________
    
    //____________________________________________
    //___________ call kernels ___________________
    cudaEvent_t start, stop;
    testCUDA(cudaEventCreate(&start));
	testCUDA(cudaEventCreate(&stop));
    float TimeVar;
    printf("_______________ Parallel ___________________\n");
    #if TEXTURE == 1
    printf("Starting kernel with texture: \n");
    testCUDA(cudaEventRecord(start,0));
    mergedSmall_k_texture<<<1,1024>>>(hostM,sizeA,sizeB,sizeM);
    testCUDA(cudaEventRecord(stop,0));
    #else
    printf("Starting kernel witout texture: \n");
    testCUDA(cudaEventRecord(start,0));
    mergedSmall_k<<<1,1024>>>(hostA,hostB,hostM,sizeA,sizeB,sizeM);
    testCUDA(cudaEventRecord(stop,0));
    #endif
	testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
    printf("elapsed time : %f\n",TimeVar);
    cout<<"Check sorted : "<<is_sorted(hostM,sizeM)<<endl;
    // tex1Dfetch( texX, i)
    
    
    //____________________________________________
    //___________ Cleaning up ____________________
    #if TEXTURE == 1
    testCUDA(cudaUnbindTexture ( texture_referenceA ));
    testCUDA(cudaUnbindTexture ( texture_referenceB ));
    cudaFree(hostA);
    cudaFree(hostB);
    free(A);
    free(B);
    #else
    testCUDA(cudaFreeHost(hostA));
    testCUDA(cudaFreeHost(hostB));
    #endif
	testCUDA(cudaEventDestroy(start));
	testCUDA(cudaEventDestroy(stop));
    testCUDA(cudaFreeHost(hostM));
    
    //____________________________________________
	return 0;
}