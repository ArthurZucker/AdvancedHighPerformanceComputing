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
#define TEXTURE 0 //set to 0 to use normal memory, else it will use texture memory for A and B
texture <int> texture_referenceA ;
texture <int> texture_referenceB ;

/*
TO DO :
 - implement using ldg  avec restricted__  et int4 qui contient 4 int, read only memory
     const int* __restrict__  A
 - mergeBig_k
 - pathBig_k

*/
int main(int argc, char* argv[]) {
    cudaDeviceReset();
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
    testCUDA(cudaSetDeviceFlags(cudaDeviceMapHost));
    //____________________________________________

    //___________ Variable declaration ___________
    int sizeA,sizeB;
    if (argc < 3) {sizeA = rand()%1024;sizeB = rand()%(1024-sizeA);} // If no arguments are provided, set random sizes
    else{sizeA=atoi(argv[1]);sizeB=atoi(argv[2]);}
    int sizeM = sizeA+sizeB;
    printf("|A| = %d, |B| = %d, |M| = %d\n",sizeA,sizeB,sizeM);
    int *hostA,*thostA,*hostB,*thostB,*hostM;
    int *seqM = (int *) malloc(sizeM*sizeof(int));
    int *A = (int *) malloc(sizeA*sizeof(int));
    int *B = (int *) malloc(sizeB*sizeof(int));
    A[0]=rand()%20;
    B[0]=rand()%20;
    for(int i=1;i<sizeA;i++){A[i]=A[i-1]+rand()%20+1;}
    for(int i=1;i<sizeB;i++){B[i]=B[i-1]+rand()%20+1;}

    //___________ TO DO: explain texture memory ___________
    testCUDA(cudaMalloc((void **)&thostA,sizeA*sizeof(int)));
    testCUDA(cudaMalloc((void **)&thostB,sizeB*sizeof(int)));

    testCUDA(cudaMemcpy(thostA, A, sizeA*sizeof(int), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(thostB, B, sizeB*sizeof(int), cudaMemcpyHostToDevice));

    testCUDA (cudaBindTexture(0,texture_referenceA, thostA,sizeA*sizeof(int)));
    testCUDA (cudaBindTexture(0,texture_referenceB, thostB,sizeB*sizeof(int)));
    //____________________________________________
    // zero copy
    testCUDA(cudaHostAlloc(&hostA,sizeA*sizeof(int),cudaHostAllocWriteCombined));
    testCUDA(cudaHostAlloc(&hostB,sizeB*sizeof(int),cudaHostAllocWriteCombined));
    hostA[0]=rand()%20;
    hostB[0]=rand()%20;
    for(int i=1;i<sizeA;i++){hostA[i]=hostA[i-1]+rand()%20+1;}
    for(int i=1;i<sizeB;i++){hostB[i]=hostB[i-1]+rand()%20+1;}

    // WC memory can be transferred across the PCI Express bus more quickly on some system configurations, but cannot be read
    // efficiently by most CPUs. WC memory is a good option for buffers that will be written by the CPU and read by the device
    // via mapped pinned memory or host->device transfers.

    testCUDA(cudaHostAlloc(&hostM,sizeM*sizeof(int),cudaHostAllocMapped)); // in order to do zero copy
    /*testCUDA(cudaHostGetDevicePointer((void **)&pM, (void *) hostM,0));
    testCUDA(cudaHostGetDevicePointer((void **)&pA, (void *) hostA,0));
    testCUDA(cudaHostGetDevicePointer((void **)&pB, (void *) hostB,0));
    */
    //_______________ Sequential _________________
    printf("_______________ Sequential _________________\n");
    clock_t begin = clock();
    merged_path_seq(A,B,seqM,sizeA,sizeB);
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("elapsed time : %f ms\n",time_spent);
    cout<<"Check sorted : "<<is_sorted(seqM,sizeM)<<endl;
    //____________________________________________


    //___________ call kernels ___________________
    cudaEvent_t start, stop;
    testCUDA(cudaEventCreate(&start));
	  testCUDA(cudaEventCreate(&stop));
    float TimeVar=0;
    //____________________________________________

    //___________ Shared _________________________
    printf("________________ Shared ___________________\n");
    testCUDA(cudaEventRecord(start));
    mergeSmall_k_shared<<<1,sizeM,sizeM*sizeof(int)>>>(hostA,hostB,hostM,sizeA,sizeB,sizeM);
    //mergeSmall_k_shared<<<1,sizeM>>>(hostA,hostB,hostM,sizeA,sizeB,sizeM);
    testCUDA(cudaEventRecord(stop));
    cout<<"Check sorted : "<<is_sorted(hostM,sizeM)<<endl;
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
    printf("elapsed time : %f ms\n",TimeVar);

    //____________________________________________

    //___________ texture ________________________
    printf("________________ Texture ___________________\n");
    testCUDA(cudaEventRecord(start,0));
    mergedSmall_k_texture<<<1,1024>>>(hostM,sizeA,sizeB,sizeM);
    testCUDA(cudaEventRecord(stop,0));
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
    printf("elapsed time : %f ms\n",TimeVar);
    cout<<"Check sorted : "<<is_sorted(hostM,sizeM)<<endl;

    //____________________________________________

    //___________ zerocopy _______________________
    printf("_______________ zero copy ___________________\n");
    testCUDA(cudaEventRecord(start,0));
    mergedSmall_k<<<1,1024>>>(hostA,hostB,hostM,sizeA,sizeB,sizeM);
    testCUDA(cudaEventRecord(stop,0));
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
    printf("elapsed time : %f ms\n",TimeVar);
    cout<<"Check sorted : "<<is_sorted(hostM,sizeM)<<endl;

    //____________________________________________

    //___________ LDG ____________________________
    printf("_____________________ LDG ___________________\n");
    testCUDA(cudaEventRecord(start,0));
    mergedSmall_k_ldg<<<1,1024>>>(hostA,hostB,hostM,sizeA,sizeB,sizeM);
    testCUDA(cudaEventRecord(stop,0));
	  testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
    printf("elapsed time : %f ms\n",TimeVar);
    cout<<"Check sorted : "<<is_sorted(hostM,sizeM)<<endl;
    //____________________________________________

  //   //___________ MergeBig _______________________
  //   printf("__________________ Path big noraml __________________\n");
  //   testCUDA(cudaEventRecord(start,0));
  //   int *__restrict__ path;
  //   //testCUDA(cudaMalloc((void **)&path,sizeM*sizeof(bool)));
  //   testCUDA(cudaHostAlloc((void **)&path,sizeA*sizeof(int),cudaHostAllocMapped));
  //   //merged_Big_k<<<(sizeM+1023)/1024,1024>>>(hostA,hostB,path,sizeA,sizeB,sizeM);
  //   //pathBig_k<<<1,1024>>>(hostA,hostB,path,sizeA,sizeB,sizeM);
  //   //pathBig_k<<<(sizeM+1023)/1024,1024>>>(hostA,hostB,path,sizeA,sizeB,sizeM);
  //   testCUDA(cudaEventRecord(stop,0));
	// testCUDA(cudaEventSynchronize(stop));
  //   testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
  //   printf("elapsed time : %f ms\n",TimeVar);
  //   //____________________________________________
  //
  //   //___________ Path Big _______________________
  //   printf("__________________ Merg big normal _________________\n");
  //   testCUDA(cudaEventRecord(start,0));
  //   //Big_k<<<(sizeM+1023)/1024,1024>>>(hostA,hostB,path,sizeA,sizeB,sizeM);
  //   //merged_Big_k<<<1,1024>>>(hostA,hostB,hostM,path,sizeM);
  //   //merged_Big_k<<<(sizeM+1023)/1024,1024>>>(hostA,hostB,hostM,path,sizeM);
  //   testCUDA(cudaEventRecord(stop,0));
	// testCUDA(cudaEventSynchronize(stop));
  //   testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
  //   printf("elapsed time : %f ms\n",TimeVar);
  //   cout<<"Check sorted : "<<is_sorted(hostM,sizeM)<<endl;
  //   //____________________________________________
  //
  //
  //    //___________ MergeBig _______________________
  //   printf("__________________ Path big ldg __________________\n");
  //   testCUDA(cudaEventRecord(start,0));
  //   //pathBig_k_ldg<<<1,1024>>>(hostA,hostB,path,sizeA,sizeB,sizeM);
  //   //pathBig_k_ldg<<<(sizeM+1023)/1024,1024>>>(hostA,hostB,path,sizeA,sizeB,sizeM);
  //   testCUDA(cudaEventRecord(stop,0));
	// testCUDA(cudaEventSynchronize(stop));
  //   testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
  //   printf("elapsed time : %f ms\n",TimeVar);
  //   //____________________________________________
  //
  //   //___________ Path Big _______________________
  //   printf("__________________ Merg big ldg _________________\n");
  //   testCUDA(cudaEventRecord(start,0));
  //   //Big_k<<<(sizeM+1023)/1024,1024>>>(hostA,hostB,path,sizeA,sizeB,sizeM);
  //   //merged_Big_k_ldg<<<1,1024>>>(hostA,hostB,hostM,path,sizeM);
  //   //merged_Big_k_ldg<<<(sizeM+1023)/1024,1024>>>(hostA,hostB,hostM,path,sizeM);
  //   testCUDA(cudaEventRecord(stop,0));
	// testCUDA(cudaEventSynchronize(stop));
  //   testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
  //   printf("elapsed time : %f ms\n",TimeVar);
  //   cout<<"Check sorted : "<<is_sorted(hostM,sizeM)<<endl;
    //____________________________________________



    //___________ Cleaning up ____________________
    testCUDA(cudaUnbindTexture ( texture_referenceA ));
    testCUDA(cudaUnbindTexture ( texture_referenceB ));
    cudaFree(hostA);
    cudaFree(hostB);
    cudaFree(thostA);
    cudaFree(thostB);
    free(A);
    free(B);
    testCUDA(cudaFreeHost(hostA));
    testCUDA(cudaFreeHost(hostB));
	  testCUDA(cudaEventDestroy(start));
	  testCUDA(cudaEventDestroy(stop));
    testCUDA(cudaFreeHost(hostM));
    //____________________________________________

	return 0;
}
