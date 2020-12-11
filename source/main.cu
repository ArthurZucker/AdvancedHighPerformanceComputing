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
    
    #if QUESTION==2 || QUESTION ==1
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
    #endif
    #if QUESTION==3
    int sizeM;
    if (argc < 2) {sizeM = rand()%1024;} 
    if (argc == 2) {sizeM=atoi(argv[1]);} // If no arguments are provided, set random sizes
    printf("|M| = %d\n",sizeM);
    #endif
    
    
    


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
    printf("__________________ Path big normal __________________\n");
    int *__restrict__ path;
    int nb_threads = 128;
    int nb_blocks = (sizeM+nb_threads-1)/nb_threads;
    //if(sizeM<1024) nb_blocks=1024;
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
    printf("__________________ Merg big normal _________________\n");
    testCUDA(cudaEventRecord(start,0));
    merged_Big_k<<<nb_blocks,nb_threads>>>(hA,hB,hM,path,sizeM);
    testCUDA(cudaEventRecord(stop,0));
	testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
    printf("elapsed time : %f ms\n",TimeVar);
    testCUDA(cudaMemcpy(M, hM, sizeM*sizeof(int), cudaMemcpyDeviceToHost));
    cout<<"Check sorted : "<<is_sorted(M,sizeM)<<endl;
    //print_t(hostM,sizeM);
    //____________________________________________
    #endif

    #if QUESTION==3
    int *__restrict__ hD;
    int *__restrict__ hsD;
    int *D  ;
    int *sD ;
    int padding = 0;

    if(sizeM != 0 && (sizeM & (sizeM-1)) == 0){
        printf("|M| is a power of 2\n");
        D  = (int *) malloc(sizeM*sizeof(int));
        sD = (int *) malloc(sizeM*sizeof(int));
        for(int i=0;i<sizeM;i++){D[i]=rand()%sizeM*5+1;}
    }
    else{
        printf("|M| was not a power of 2, it will be changed\n");
        int power = 1;
        while(power < sizeM) power*=2;
        printf("new |M| with padding : %d\n",power);
        D  = (int *) malloc(power*sizeof(int));
        sD = (int *) malloc(power*sizeof(int));
        for(int i=0;i<sizeM;i++){D[i]=rand()%sizeM*5+1;}
        for(int i = sizeM;i<power;i++){D[i] = ( int) -1 >> 1;}
        padding = power-sizeM;
        sizeM = power;
    }
    printf("Assigning M\n");
    
    //int nb_threads = 128; // changing it might be smart
    //int nb_blocks = (sizeM+nb_threads-1)/nb_threads;
    printf("__________________ sort M __________________\n");
    
    //if(sizeM<1024) nb_blocks=1024;
    testCUDA(cudaMalloc((void **)&hsD,sizeM*sizeof(int)));
    testCUDA(cudaMalloc((void **)&hD,sizeM*sizeof(int)));
    testCUDA(cudaMemcpy(hD, D, sizeM*sizeof(int), cudaMemcpyHostToDevice));
    testCUDA(cudaEventRecord(start,0));
    for(int i=1;i<sizeM;i*=2){
        for(int j=0;j<sizeM;j+=2*i){
            
            if(i>512){
                int *__restrict__ path;
                int nblocks = (2*i+1023)/1024 ;
                //exit(0);
                testCUDA(cudaMalloc((void **)&path,2*(nblocks+1)*sizeof(int)));
                pathBig_k   <<<nblocks,1024>>>(&hD[j],&hD[j+i],path,i,i,2*i);
                merged_Big_k<<<nblocks,1024>>>(&hD[j],&hD[j+i],&hsD[j],path,2*i);
            }
            else{
                mergedSmall_k_ldg<<<1,2*i>>>(&hD[j],&hD[j+i],&hsD[j],i,i,2*i);
                //cout<<"Check sorted : "<<is_sorted(&hsD[j],i)<<endl;
                
            }
        }
        int *ht = hD;   
        hD = hsD;
        hsD = ht;
    }
    int *ht = hD;   
    hD = hsD;
    hsD = ht;
    testCUDA(cudaMemcpy(sD, hsD, sizeM*sizeof(int), cudaMemcpyDeviceToHost));
    //print_t(&sD[padding],sizeM-padding);
    testCUDA(cudaEventRecord(stop,0));
    testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
    testCUDA(cudaMemcpy(sD, hsD, sizeM*sizeof(int), cudaMemcpyDeviceToHost));
    printf("elapsed time : %f ms\n",TimeVar);
    cout<<"Check sorted : "<<is_sorted(&sD[padding],sizeM-padding)<<endl;
    //____________________________________________
    clock_t begin = clock();
    qsort(D, sizeM, sizeof(int), cmpfunc);
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("elapsed time : %f ms\n",time_spent*1000);
    cout<<"Check sorted : "<<is_sorted(D,sizeM)<<endl;
    
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
    #if QUESTION == 2||QUESTION==1
    free(A);
    free(B);
    free(M);
    #endif 
	testCUDA(cudaEventDestroy(start));
	testCUDA(cudaEventDestroy(stop));
    // ____________________________________________
    #if QUESTION==4
    //__________________________ Batch merge part __________________________
    // L’objectif est simplement de répartir les block de manière intelligente 
    // sur l’ensemble des calculs Ai + Bi = Mi .
    int N = 10; //si trop gros on peut pas allouer sur le gpu (je crois)
    int d = 6; //306

    // _________________________________zero copy____________________________________ 
    printf("_______________________________zero copy____________________________________\n");
    int* host_all_M;
    int* host_all_STM;
    int* host_all_size_A;
    int* host_all_size_B;

    // allocation for save size
    testCUDA(cudaHostAlloc(&host_all_size_A,N*sizeof(int),cudaHostAllocMapped));
    testCUDA(cudaHostAlloc(&host_all_size_B,N*sizeof(int),cudaHostAllocMapped));

    // Allocation device 1D
    int size_all_A=0;
    int size_all_B=0;
    int sizeA;
    int sizeB;
    for(int i = 0;i<N;i++){ 
        sizeA = rand()%d+1;
        sizeB = (d-sizeA);
        // printf("|A| = %d, |B| = %d\n",sizeA,sizeB);
        host_all_size_A[i] = sizeA;
        host_all_size_B[i] = sizeB;
        size_all_A += sizeA;
        size_all_B +=sizeB;
    }

    // allocation for M and STM
    printf("size_all_A = %d, size_all_B = %d, size_all_A + size_all_B = %d, size_all_M = %d\n",size_all_A,size_all_B,size_all_A+size_all_B,N*d);
    testCUDA(cudaHostAlloc(&host_all_STM,N*d*sizeof(int),cudaHostAllocMapped));    
    testCUDA(cudaHostAlloc(&host_all_M,N*d*sizeof(int),cudaHostAllocMapped));  

    printf("_______ Initialisation___________\n");
    // début init 1D
    if(host_all_size_A[0]!=0){
        host_all_M[0]=rand()%20+1;
        // printf("M[0]=%d\n",host_all_M[0]);
        for(int j = 1;j<host_all_size_A[0];j++){
            host_all_M[j]=host_all_M[j-1]+rand()%20+1;
            // printf("M[%d]=%d\n",j,host_all_M[j]);
        }
    }
    if(host_all_size_B[0]!=0){
        host_all_M[host_all_size_A[0]]=rand()%20+1;
        // printf("M[%d]=%d\n",host_all_size_A[0],host_all_M[host_all_size_A[0]]);
        for(int j = host_all_size_A[0]+1;j<host_all_size_B[0]+host_all_size_A[0];j++){
            host_all_M[j]=host_all_M[j-1]+rand()%20+1;
            // printf("M[%d]=%d\n",j,host_all_M[j]);
        }
    }
    //fin init 1D
    int tmp_A=host_all_size_A[0];
    int tmp_B=host_all_size_B[0];
    for(int i = 1;i<N;i++){ 
        // Initialisation 1D 1 tableau
        if(host_all_size_A[i]!=0){
            host_all_M[tmp_A+tmp_B]=rand()%20+1;
            for(int j = tmp_A+tmp_B+1;j<tmp_A+tmp_B+host_all_size_A[i];j++){
                host_all_M[j]=host_all_M[j-1]+rand()%20+1;
            }
            tmp_A+= host_all_size_A[i];
    
        }
        if(host_all_size_B[i]!=0){
            host_all_M[tmp_A+tmp_B]=rand()%20+1;
            for(int j = tmp_A+tmp_B+1;j<tmp_A+tmp_B+host_all_size_B[i];j++){
                host_all_M[j]=host_all_M[j-1]+rand()%20+1;
            }
            tmp_B+= host_all_size_B[i];
        }
    }

    testCUDA(cudaEventCreate(&start));
    testCUDA(cudaEventCreate(&stop));
    
    printf("_______ Début de la fonction___________\n");
    int numBlocks = N; //big number
    int threadsPerBlock = d; // multiple de d
    testCUDA(cudaEventRecord(start));
    mergeSmallBatch_k<<<numBlocks,threadsPerBlock>>>(host_all_M,host_all_STM,host_all_size_A,host_all_size_B,d);
    testCUDA(cudaEventRecord(stop));
	testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
    printf("elapsed time : %f ms\n",TimeVar);

    printf("_______ Check résultats___________\n");
    int all_sorted=1;
    int sorted;
    for(int i = 0;i<N*d;i+=d){
        sorted = is_sorted(&host_all_STM[i],d);
        if(sorted ==0){
            cout<<"Check sorted : "<<sorted<<endl;
            all_sorted = 0;
        }
    }
    if(all_sorted==1){
        printf("All table are sorted !\n");
    }
    else{
        printf("There is a table not sorted !\n");
    }
    
    // _____________________________________Copy __________________________________________
    printf("_______________________________Copy____________________________________\n");

    int* all_M = (int *) malloc(N*d*sizeof(int));
    int* all_STM = (int *) malloc(N*d*sizeof(int));
    int* all_size_A = (int *) malloc(N*sizeof(int));
    int* all_size_B = (int *) malloc(N*sizeof(int));
    int* h_all_M;
    int* h_all_STM;
    int* h_all_size_A;
    int* h_all_size_B;

    // allocation for save size
    testCUDA(cudaMalloc((void **)&h_all_size_A,N*sizeof(int)));
    testCUDA(cudaMalloc((void **)&h_all_size_B,N*sizeof(int)));

    // Initialisation size
    size_all_A=0;
    size_all_B=0;
    for(int i = 0;i<N;i++){ 
        sizeA = rand()%d+1;
        sizeB = (d-sizeA);
        // printf("|A| = %d, |B| = %d\n",sizeA,sizeB);
        all_size_A[i] = sizeA;
        all_size_B[i] = sizeB;
        size_all_A += sizeA;
        size_all_B +=sizeB;
    }

    testCUDA(cudaMemcpy(h_all_size_A, all_size_A, N*sizeof(int), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(h_all_size_B, all_size_B, N*sizeof(int), cudaMemcpyHostToDevice));
    

    // allocation for M and STM
    testCUDA(cudaMalloc((void **)&h_all_M,N*d*sizeof(int)));
    testCUDA(cudaMalloc((void **)&h_all_STM,N*d*sizeof(int)));

    printf("_______ Initialisation___________\n");
    // début init 1D
    if(all_size_A[0]!=0){
        all_M[0]=rand()%20+1;
        // printf("M[0]=%d\n",all_M[0]);
        for(int j = 1;j<all_size_A[0];j++){
            all_M[j]=all_M[j-1]+rand()%20+1;
            // printf("M[%d]=%d\n",j,all_M[j]);
        }
    }
    if(all_size_B[0]!=0){
        all_M[all_size_A[0]]=rand()%20+1;
        // printf("M[%d]=%d\n",all_size_A[0],all_M[all_size_A[0]]);
        for(int j = all_size_A[0]+1;j<all_size_B[0]+all_size_A[0];j++){
            all_M[j]=all_M[j-1]+rand()%20+1;
            // printf("M[%d]=%d\n",j,all_M[j]);
        }
    }
    //fin init 1D
    tmp_A=all_size_A[0];
    tmp_B=all_size_B[0];
    for(int i = 1;i<N;i++){ 
        // Initialisation 1D 1 tableau
        if(all_size_A[i]!=0){
            all_M[tmp_A+tmp_B]=rand()%20+1;
            for(int j = tmp_A+tmp_B+1;j<tmp_A+tmp_B+all_size_A[i];j++){
                all_M[j]=all_M[j-1]+rand()%20+1;
            }
            tmp_A+= all_size_A[i];
    
        }
        if(all_size_B[i]!=0){
            all_M[tmp_A+tmp_B]=rand()%20+1;
            for(int j = tmp_A+tmp_B+1;j<tmp_A+tmp_B+all_size_B[i];j++){
                all_M[j]=all_M[j-1]+rand()%20+1;
            }
            tmp_B+= all_size_B[i];
        }
    }
    // for(int i = 1;i<N*d;i++){
    //     printf("M[%d]=%d\n",i,all_M[i]);
    // }
    testCUDA(cudaMemcpy(h_all_M, all_M, N*d*sizeof(int), cudaMemcpyHostToDevice));

    printf("_______ Début de la fonction___________\n");
    numBlocks = N; //big number
    threadsPerBlock = d; // multiple de d
    testCUDA(cudaEventRecord(start));
    mergeSmallBatch_k<<<numBlocks,threadsPerBlock>>>(h_all_M,h_all_STM,h_all_size_A,h_all_size_B,d);
    testCUDA(cudaEventRecord(stop));
	testCUDA(cudaEventSynchronize(stop));
    testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
    printf("elapsed time : %f ms\n",TimeVar);
    testCUDA(cudaMemcpy(all_STM, h_all_STM, N*d*sizeof(int), cudaMemcpyDeviceToHost));

    // for(int i = 1;i<N*d;i++){
    //     printf("STM[%d]=%d\n",i,all_STM[i]);
    // }

    printf("_______ Check résultats___________\n");
    all_sorted=1;
    for(int i = 0;i<N*d;i+=d){
        sorted = is_sorted(&all_STM[i],d);
        if(sorted ==0){
            cout<<"Check sorted : "<<sorted<<endl;
            all_sorted = 0;
        }
    }
    if(all_sorted==1){
        printf("All table are sorted !\n");
    }
    else{
        printf("There is a table not sorted !\n");
    }

    printf("_______ Cleaning ___________\n");
    // clean copy 
    free(all_M);
    free(all_STM);
    free(all_size_A);
    free(all_size_B);
    testCUDA(cudaFree(h_all_M));
    testCUDA(cudaFree(h_all_STM));
    testCUDA(cudaFree(h_all_size_A));
    testCUDA(cudaFree(h_all_size_B));

    // clean zero copy
    testCUDA(cudaFreeHost(host_all_M));
    testCUDA(cudaFreeHost(host_all_STM));
    testCUDA(cudaFreeHost(host_all_size_A));
    testCUDA(cudaFreeHost(host_all_size_B));
    
    testCUDA(cudaEventDestroy(start));
    testCUDA(cudaEventDestroy(stop));
    #endif
	return 0;
}
