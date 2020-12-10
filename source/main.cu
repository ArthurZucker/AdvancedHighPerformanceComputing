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
#define QUESTION 4
#define INFO 0
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
    int Tmax;
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        #if INFO == 1
		printf("Max Grid size: %dx%d\n",  prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("Max Thread Dim: %d,%d,%d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max Thread per blocks: %d\n", prop.maxThreadsPerBlock);
        printf("Max number of threads per multiprocessor : %d\n",prop.maxThreadsPerMultiProcessor);
        printf("Number of multiprocessors on device : %d\n",prop.multiProcessorCount);
        printf("Amount of Shared mem available for int : %d\n",prop.sharedMemPerMultiprocessor/sizeof(int));
        printf("Max running threads : %d\n",prop.maxThreadsPerMultiProcessor*prop.multiProcessorCount);
        #endif
        Tmax = prop.maxThreadsPerMultiProcessor*prop.multiProcessorCount;
	}
    //Tmax =1024;
	cudaSetDevice(0);
    testCUDA(cudaSetDeviceFlags(cudaDeviceMapHost));
    //____________________________________________

    //___________ Variable declaration ___________
    int sizeA,sizeB;
    if (argc < 2) {sizeA = rand()%1024;sizeB = rand()%(1024-sizeA);} // If no arguments are provided, set random sizes
    else if(argc == 2){sizeA=atoi(argv[1]);sizeB=atoi(argv[1]);}
    else{sizeA=atoi(argv[1]);sizeB=atoi(argv[2]);}
    int sizeM = sizeA+sizeB;
    printf("|A| = %d, |B| = %d, |M| = %d\n",sizeA,sizeB,sizeM);
    int *hostA,*thostA,*hostB,*thostB,*hostM,*hA,*hB,*hM;
    int *seqM = (int *) malloc(sizeM*sizeof(int));
    int *A = (int *) malloc(sizeA*sizeof(int));
    int *B = (int *) malloc(sizeB*sizeof(int));
    int *M = (int *) malloc(sizeM*sizeof(int));
    A[0]=rand()%20;
    B[0]=rand()%20;
    for(int i=1;i<sizeA;i++){A[i]=A[i-1]+rand()%20+1;}
    for(int i=1;i<sizeB;i++){B[i]=B[i-1]+rand()%20+1;}


    //___________ call kernels ___________________
    cudaEvent_t start, stop;
    testCUDA(cudaEventCreate(&start));
	testCUDA(cudaEventCreate(&stop));
    float TimeVar=0;



    #if QUESTION == 1
    //___________ TO DO: explain texture memory ___________
    testCUDA(cudaMalloc((void **)&thostA,sizeA*sizeof(int)));
    testCUDA(cudaMalloc((void **)&thostB,sizeB*sizeof(int)));

    testCUDA(cudaMemcpy(thostA, A, sizeA*sizeof(int), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(thostB, B, sizeB*sizeof(int), cudaMemcpyHostToDevice));

    testCUDA (cudaBindTexture(0,texture_referenceA, thostA,sizeA*sizeof(int)));
    testCUDA (cudaBindTexture(0,texture_referenceB, thostB,sizeB*sizeof(int)));
    //____________________________________________
    // zero copy
    testCUDA(cudaHostAlloc(&hostA,sizeA*sizeof(int),cudaHostAllocMapped)); //cudaHostAllocWriteCombined
    testCUDA(cudaHostAlloc(&hostB,sizeB*sizeof(int),cudaHostAllocMapped));
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
    printf("elapsed time : %f ms\n",time_spent*1000);
    cout<<"Check sorted : "<<is_sorted(seqM,sizeM)<<endl;
    //____________________________________________


    
    //____________________________________________

    //___________ Shared _________________________
    printf("________________ Shared ___________________\n");
    testCUDA(cudaEventRecord(start));
    mergeSmall_k_shared<<<1,sizeM,sizeM*sizeof(int)>>>(hostA,hostB,hostM,sizeA,sizeB,sizeM);
    //mergeSmall_k_shared<<<1,sizeM>>>(hostA,hostB,hostM,sizeA,sizeB,sizeM);
    testCUDA(cudaEventRecord(stop));
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
    printf("elapsed time : %f ms\n",TimeVar);
    cout<<"Check sorted : "<<is_sorted(hostM,sizeM)<<endl;

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
    for(int i=1;i<sizeA;i++){hostM[i]=0;}
    
    #endif
    #if QUESTION==2
    //___________ MergeBig _______________________
    printf("__________________ Path big normal __________________\n");
    int *__restrict__ path;
    int nb_threads = 5;
    int nb_blocks = (sizeM+nb_threads-1)/nb_threads;
    if(sizeM<1024) nb_blocks=1024;
    nb_blocks = 2;
    testCUDA(cudaMalloc((void **)&hA,sizeA*sizeof(int)));
    testCUDA(cudaMalloc((void **)&hB,sizeB*sizeof(int)));
    testCUDA(cudaMalloc((void **)&hM,sizeM*sizeof(int)));

    testCUDA(cudaMemcpy(hA, A, sizeA*sizeof(int), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(hB, B, sizeB*sizeof(int), cudaMemcpyHostToDevice));

    testCUDA(cudaMalloc((void **)&path,2*(nb_blocks+1)*sizeof(int)));
    testCUDA(cudaEventRecord(start,0));
    pathBig_k<<<nb_blocks,nb_threads>>>(hA,hB,path,sizeA,sizeB,sizeM);
    testCUDA(cudaEventRecord(stop,0));
	testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
    printf("elapsed time : %f ms\n",TimeVar);
    //____________________________________________
  
    //___________ Path Big _______________________
    printf("__________________ Merg big normal _________________\n");
    testCUDA(cudaEventRecord(start,0));
    merged_Big_k<<<nb_blocks,nb_threads>>>(hA,hB,hM,path,sizeM);
    testCUDA(cudaEventRecord(stop,0));
	testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
    printf("elapsed time : %f ms\n",TimeVar);
    testCUDA(cudaMemcpy(M, hM, sizeB*sizeof(int), cudaMemcpyDeviceToHost));
    cout<<"Check sorted : "<<is_sorted(M,sizeM)<<endl;
    //print_t(hostM,sizeM);
    //____________________________________________
    #endif
  
    //___________ MergeBig _______________________
    // printf("__________________ Path big sans shared + ldg __________________\n");
    // testCUDA(cudaEventRecord(start,0));
    // pathBig_k_naive_ldg<<<(sizeM+1023)/1024,1024>>>(thostA,thostB,path,sizeA,sizeB,sizeM);
    // testCUDA(cudaEventRecord(stop,0));
	// testCUDA(cudaEventSynchronize(stop));
    // testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
    // printf("elapsed time : %f ms\n",TimeVar);
    //____________________________________________
  
    //___________ Path Big _______________________
    // printf("__________________ Merg big sans shared + ldg _________________\n");
    // testCUDA(cudaEventRecord(start,0));
    // merged_Big_k_naive_ldg<<<(sizeM+1023)/1024,1024>>>(thostA,thostB,hostM,path,sizeM);
    // testCUDA(cudaEventRecord(stop,0));
	// testCUDA(cudaEventSynchronize(stop));
    // testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
    // printf("elapsed time : %f ms\n",TimeVar);
    // cout<<"Check sorted : "<<is_sorted(hostM,sizeM)<<endl;
    //____________________________________________
    //___________ Cleaning up ____________________
    #if QUESTION == 1
    testCUDA(cudaUnbindTexture ( texture_referenceA ));
    testCUDA(cudaUnbindTexture ( texture_referenceB ));
    cudaFree(thostA);
    cudaFree(thostB);
    testCUDA(cudaFreeHost(hostA));
    testCUDA(cudaFreeHost(hostB));
    testCUDA(cudaFreeHost(hostM));
    #endif
    free(A);
    free(B);
    free(M);
	testCUDA(cudaEventDestroy(start));
	testCUDA(cudaEventDestroy(stop));
    // ____________________________________________
    #if QUESTION==4
    //__________________________ Batch merge part __________________________
    // L’objectif est simplement de répartir les block de manière intelligente 
    // sur l’ensemble des calculs Ai + Bi = Mi .
    int N = 6; //si trop gros on pet pas allouer sur le gpu (je crois)
    int d = 3; //306
    // int sizeA,sizeB,sizeM;

    // Allocation globale 2D
    // int** all_A = (int**)malloc(N*sizeof(int*));
    // int** all_B = (int**)malloc(N*sizeof(int*));
    // int** all_M = (int**)malloc(N*sizeof(int*));
    // int* all_size_A = (int*)malloc(N*sizeof(int));
    // int* all_size_B = (int*)malloc(N*sizeof(int));
    // int* all_size_M = (int*)malloc(N*sizeof(int));

    // int** all_A;
    // int** all_B; 
    // int** all_M;
    int* all_A;
    int* all_B; 
    int* all_M;
    int* all_size_A;
    int* all_size_B;
    // int* all_size_M;

    // allocation for save size
    testCUDA(cudaHostAlloc(&all_size_A,N*sizeof(int),cudaHostAllocMapped));
    testCUDA(cudaHostAlloc(&all_size_B,N*sizeof(int),cudaHostAllocMapped));
    // testCUDA(cudaHostAlloc(&all_size_M,N*sizeof(int),cudaHostAllocMapped));
    
    // Allocation device 2D
    // testCUDA(cudaHostAlloc(&all_A,N*sizeof(int*),cudaHostAllocMapped));
    // testCUDA(cudaHostAlloc(&all_B,N*sizeof(int*),cudaHostAllocMapped));
    // testCUDA(cudaHostAlloc(&all_M,N*sizeof(int*),cudaHostAllocMapped));

    // Allocation device 1D
    int size_all_A=0;
    int size_all_B=0;
    for(int i = 0;i<N;i++){ 
        sizeA = rand()%d+1;
        sizeB = (d-sizeA);
        sizeM = sizeA+sizeB;
        printf("|A| = %d, |B| = %d, |M| = %d\n",sizeA,sizeB,sizeM);
        all_size_A[i] = sizeA;
        all_size_B[i] = sizeB;
        size_all_A += sizeA;
        size_all_B +=sizeB;
    }
    // for(int i = 0;i<N;i++){ 
    //     printf("all_size_A[%d]=%d, all_size_B[%d]=%d \n",i,all_size_A[i],i,all_size_B[i]);
    // }
    printf("size_all_A = %d, size_all_B = %d, size_all_A + size_all_B = %d, size_all_M = %d\n",size_all_A,size_all_B,size_all_A+size_all_B,N*d);
    testCUDA(cudaHostAlloc(&all_A,size_all_A*sizeof(int),cudaHostAllocMapped));
    testCUDA(cudaHostAlloc(&all_B,size_all_B*sizeof(int),cudaHostAllocMapped));
    testCUDA(cudaHostAlloc(&all_M,N*d*sizeof(int),cudaHostAllocMapped));    

    printf("_______ Initialisation___________\n");
    // début init 1D
    all_A[0]=rand()%20;
    for(int j = 1;j<all_size_A[0];j++){
        all_A[j]=all_A[j-1]+rand()%20+1;
    }
    all_B[0]=rand()%20;
    for(int j = 1;j<all_size_B[0];j++){
        all_B[j]=all_B[j-1]+rand()%20+1;
    }
    // fin init 1D
    int tmp_A=all_size_A[0];
    int tmp_B=all_size_B[0];
    for(int i = 1;i<N;i++){ // mettre N pour deux 2D et commencer à 0 
        // printf("i = %d\n",i);
        // sizeA = rand()%d;
        // sizeB = (d-sizeA);
        // sizeM = sizeA+sizeB;
        // printf("|A| = %d, |B| = %d, |M| = %d\n",sizeA,sizeB,sizeM);
        // all_size_A[i] = sizeA;
        // all_size_B[i] = sizeB;
        
        // all_size_M[i] = sizeM;

        // Allocation gobale 2D
        // all_A[i] = (int *) malloc(sizeA*sizeof(int));
        // all_B[i] = (int *) malloc(sizeB*sizeof(int));
        // all_M[i] = (int *) malloc(sizeM*sizeof(int));
        
        // Allocation device 2D
        // testCUDA(cudaHostAlloc(&all_A[i],sizeA*sizeof(int),cudaHostAllocMapped));
        // testCUDA(cudaHostAlloc(&all_B[i],sizeB*sizeof(int),cudaHostAllocMapped));
        // testCUDA(cudaHostAlloc(&all_M[i],sizeM*sizeof(int),cudaHostAllocMapped));

        // Initialisation 2D
        // all_A[i][0]=rand()%20;
        // all_B[i][0]=rand()%20;
        // for(int j=1;j<sizeA;j++){all_A[i][j]=all_A[i][j-1]+rand()%20+1;}
        // for(int j=1;j<sizeB;j++){all_B[i][j]=all_B[i][j-1]+rand()%20+1;}
        
        // Initialisation 1D
        all_A[tmp_A]=rand()%20;
        for(int j = tmp_A+1;j<tmp_A+all_size_A[i];j++){
            all_A[j]=all_A[j-1]+rand()%20+1;
        }
        tmp_A+= all_size_A[i];

        // for(int j = all_size_A[i-1];j<all_size_A[i];j++){
        //     printf("all_A[%d] = %d\n",j,all_A[j]);
        // }

        all_B[tmp_B]=rand()%20;
        for(int j = tmp_B+1;j<tmp_B+all_size_B[i];j++){
            all_B[j]=all_B[j-1]+rand()%20+1;
        }
        tmp_B+= all_size_B[i];
    }
    // for(int i=0;i<N;i++){
    //     printf("all_size_B[%d] = %d\n",i,all_size_B[i]);
    // }
    // for(int i = 0;i<size_all_B;i++){
    //     printf("all_B[%d] = %d\n",i,all_B[i]);
    // }

    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    
    printf("_______ Début de la fonction___________\n");
    int numBlocks = N; //big number
    int threadsPerBlock = d; // multiple de d
    testCUDA(cudaEventRecord(start));
    mergeSmallBatch_k<<<numBlocks,threadsPerBlock>>>(all_A,all_B,all_M,all_size_A,all_size_B,N*d,d);
    testCUDA(cudaEventRecord(stop));
	testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
    printf("elapsed time : %f ms\n",TimeVar);

    printf("_______ Check résultats___________\n");

    for(int i = 0;i<size_all_A;i++){
        printf("all_A[%d] = %d\n",i,all_A[i]);
    }
    for(int i = 0;i<size_all_B;i++){
        printf("all_B[%d] = %d\n",i,all_B[i]);
    }
    for(int i = 0;i<N*d;i++){
        printf("M[%d]=%d\n",i,all_M[i]);
    }

    // for(int i = 0;i<N;i++){
    //     //printf("%d\n",i);
    //     cout<<"Check sorted : "<<is_sorted(all_M[i],d)<<endl;
    // }

    // for(int i = 0;i<N;i++){
    //     //printf("%d\n",i);
    //     cout<<"Check sorted : "<<is_sorted(all_M[i],d)<<endl;
    // }
    
    
    //for(int i = 0;i<N;i++){free(all_A[i]);free(all_B[i]);free(all_M[i]);}
    // free(all_A);
    // free(all_B);
    // free(all_M);
    // free(all_size_A);
    // free(all_size_B);
    // free(all_size_M);
    // for(int i = 0;i<N;i++){testCUDA(cudaFreeHost(all_A[i]));testCUDA(cudaFreeHost(all_B[i]));testCUDA(cudaFreeHost(all_M[i]));}
    // testCUDA(cudaFreeHost(all_A));
    // testCUDA(cudaFreeHost(all_B));
    // testCUDA(cudaFreeHost(all_M));
    // testCUDA(cudaFreeHost(all_size_A));
    // testCUDA(cudaFreeHost(all_size_B));
    
    // testCUDA(cudaEventDestroy(start));
    // testCUDA(cudaEventDestroy(stop));
    #endif
	return 0;
}
