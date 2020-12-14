/****************************************************************************
 * Copyright (C) 2020 by Arthur Zucker @ Apavou Clément                     *
 ****************************************************************************/

/**
 * @file main.cu
 * @author Arthur Zucker & Clément Apavou  
 * @date 12 Dec 2020
 * @brief Main file used to produce results for each questions
 *
 * In this porject, we tackled the MERGE SORT problem on GPU
 * using CUDA. We answered questions from a subject. If you want to 
 * see the original Merge sort articles, 
 * @see https://www.researchgate.net/profile/Oded-Green/publication/254462662_GPU_merge_path_a_GPU_merging_algorithm/links/543eeaa00cf2e76f02244884/GPU-merge-path-a-GPU-merging-algorithm.pdf
 * @see https://arxiv.org/pdf/1406.2628.pdf 
 */


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
texture <int> texture_referenceA ;
texture <int> texture_referenceB ;
#define QUESTION 1  /**< Choose from {1,2,3,4,5} depending on the question */
#define INFO 0      /**< Set to 1 if you need to see GPU infromations. */


int main(int argc, char* argv[]) {
    //___________ Basic initialisation ___________
	srand((unsigned int)time(NULL));
	int nDevices;
	cudaGetDeviceCount(&nDevices);
    // int Tmax;
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
        // Tmax = prop.maxThreadsPerMultiProcessor*prop.multiProcessorCount;
	}
    //Tmax =1024;
	cudaSetDevice(0);
    testCUDA(cudaSetDeviceFlags(cudaDeviceMapHost));
    //____________________________________________

    //___________ Initialising size of arrays  ___________
    #if QUESTION==2 || QUESTION ==1
        int sizeA,sizeB;
        if (argc < 2) {sizeA = rand()%1024;sizeB = rand()%(1024-sizeA);} // If no arguments are provided, set random sizes
        else if(argc == 2){sizeA=atoi(argv[1]);sizeB=atoi(argv[1]);}
        else{sizeA=atoi(argv[1]);sizeB=atoi(argv[2]);}
        int sizeM = sizeA+sizeB;
        printf("|A| = %d, |B| = %d, |M| = %d\n",sizeA,sizeB,sizeM);
        #if QUESTION ==1
            int *hostA,*thostA,*hostB,*thostB,*hostM,*thostM;
        #endif
        int *seqM = (int *) malloc(sizeM*sizeof(int));
        int *A = (int *) malloc(sizeA*sizeof(int));
        int *B = (int *) malloc(sizeB*sizeof(int));
        int *M = (int *) malloc(sizeM*sizeof(int));
        A[0]=rand()%20;
        B[0]=rand()%20;
        for(int i=1;i<sizeA;i++){A[i]=A[i-1]+rand()%20+1;}
        for(int i=1;i<sizeB;i++){B[i]=B[i-1]+rand()%20+1;}
    #endif
    #if QUESTION == 5
        int sizeM;
        if (argc < 2) {sizeM = rand()%1024;} 
        if (argc == 2) {sizeM=atoi(argv[1]);} // If no arguments are provided, set random sizes
        printf("|M| = %d\n",sizeM);
    #endif
    //___________________________ Useful time stamps _________________________________
    cudaEvent_t start, stop;
    testCUDA(cudaEventCreate(&start));
	testCUDA(cudaEventCreate(&stop));
    float TimeVar=0;


    //___________________________ Question 1 _________________________________
    #if QUESTION == 1
        FILE *f = fopen("../results/results1.csv", "w"); 
        fprintf(f, "type,memory,time\n");
        // Copy 
        testCUDA(cudaMalloc((void **)&thostA,sizeA*sizeof(int)));
        testCUDA(cudaMalloc((void **)&thostB,sizeB*sizeof(int)));
        testCUDA(cudaMalloc((void **)&thostM,sizeM*sizeof(int)));

        testCUDA(cudaMemcpy(thostA, A, sizeA*sizeof(int), cudaMemcpyHostToDevice));
        testCUDA(cudaMemcpy(thostB, B, sizeB*sizeof(int), cudaMemcpyHostToDevice));
        // texture memory
        testCUDA (cudaBindTexture(0,texture_referenceA, thostA,sizeA*sizeof(int)));
        testCUDA (cudaBindTexture(0,texture_referenceB, thostB,sizeB*sizeof(int)));
        //____________________________________________
        // Zero copy
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
 
        //_______________ Sequential _________________
        printf("_______________ Sequential _________________\n");
        clock_t begin = clock();
        merged_path_seq(A,B,seqM,sizeA,sizeB);
        clock_t end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        printf("elapsed time : %f ms\n",time_spent*1000);
        fprintf(f, "Sequential,CPU,%f\n",time_spent*1000);
        cout<<"Check sorted : "<<is_sorted(seqM,sizeM)<<endl;
        //____________________________________________


        //_____________________________ Zero copy ______________________________________________________________
        printf("__________________________Zero copy________________________________\n");
        printf("_______________ Zero copy Normal ___________________\n");
        testCUDA(cudaEventRecord(start,0));
        mergedSmall_k<<<1,1024>>>(hostA,hostB,hostM,sizeA,sizeB,sizeM);
        testCUDA(cudaEventRecord(stop,0));
        testCUDA(cudaEventSynchronize(stop));
        testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
        printf("elapsed time : %f ms\n",TimeVar);
        fprintf(f, "Normal,ZeroCpy,%f\n",TimeVar);
        cout<<"Check sorted : "<<is_sorted(hostM,sizeM)<<endl;
        //____________________________________________

        for(int i=0;i<sizeM;i++){hostM[i]=0;}

        //___________ Zero copy Shared _________________________
        printf("________________ Zero copy Shared ___________________\n");
        testCUDA(cudaEventRecord(start));
        mergeSmall_k_shared<<<1,sizeM,sizeM*sizeof(int)>>>(hostA,hostB,hostM,sizeA,sizeB,sizeM);
        testCUDA(cudaEventRecord(stop));
        testCUDA(cudaEventSynchronize(stop));
        testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
        printf("elapsed time : %f ms\n",TimeVar);
        fprintf(f, "Shared,ZeroCpy,%f\n",TimeVar);
        cout<<"Check sorted : "<<is_sorted(hostM,sizeM)<<endl;
        //____________________________________________

        for(int i=0;i<sizeM;i++){hostM[i]=0;}

        //___________ Zero copy LDG ____________________________
        printf("_____________________ Zero copy LDG ___________________\n");
        testCUDA(cudaEventRecord(start,0));
        mergedSmall_k_ldg<<<1,1024>>>(hostA,hostB,hostM,sizeA,sizeB,sizeM);
        testCUDA(cudaEventRecord(stop,0));
        testCUDA(cudaEventSynchronize(stop));
        testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
        printf("elapsed time : %f ms\n",TimeVar);
        fprintf(f, "Ldg,ZeroCpy,%f\n",TimeVar);
        cout<<"Check sorted : "<<is_sorted(hostM,sizeM)<<endl;
        //____________________________________________

        for(int i=0;i<sizeM;i++){hostM[i]=0;}

        //___________ Texture ________________________
        printf("________________ Texture ___________________\n");
        testCUDA(cudaEventRecord(start,0));
        mergedSmall_k<<<1,1024>>>(hostA,hostB,hostM,sizeA,sizeB,sizeM);
        testCUDA(cudaEventRecord(stop,0));
        testCUDA(cudaEventSynchronize(stop));
        testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
        printf("elapsed time : %f ms\n",TimeVar);
        fprintf(f, "Texture,ZeroCpy,%f\n",TimeVar);
        cout<<"Check sorted : "<<is_sorted(hostM,sizeM)<<endl;
        //____________________________________________
        
        //_____________________________ Copy ______________________________________________________________
        printf("__________________________Copy________________________________\n");
        printf("_______________copy Normal ___________________\n");
        testCUDA(cudaEventRecord(start,0));
        mergedSmall_k<<<1,1024>>>(thostA,thostB,thostM,sizeA,sizeB,sizeM);
        testCUDA(cudaEventRecord(stop,0));
        testCUDA(cudaEventSynchronize(stop));
        testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
        printf("elapsed time : %f ms\n",TimeVar);
        fprintf(f, "Normal,MemCpy,%f\n",TimeVar);
        testCUDA(cudaMemcpy(M, thostM, sizeM*sizeof(int), cudaMemcpyDeviceToHost)); // retrieve M on the device
        cout<<"Check sorted : "<<is_sorted(M,sizeM)<<endl;
        //____________________________________________

        for(int i=0;i<sizeM;i++){M[i]=0;}

        //___________ copy Shared _________________________
        printf("________________copy Shared ___________________\n");
        testCUDA(cudaEventRecord(start));
        mergeSmall_k_shared<<<1,sizeM,sizeM*sizeof(int)>>>(thostA,thostB,thostM,sizeA,sizeB,sizeM);
        testCUDA(cudaEventRecord(stop));
        testCUDA(cudaEventSynchronize(stop));
        testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
        printf("elapsed time : %f ms\n",TimeVar);
        fprintf(f, "Shared,MemCpy,%f\n",TimeVar);
        testCUDA(cudaMemcpy(M, thostM, sizeM*sizeof(int), cudaMemcpyDeviceToHost));
        cout<<"Check sorted : "<<is_sorted(M,sizeM)<<endl;
        //____________________________________________

        for(int i=0;i<sizeM;i++){M[i]=0;}

        //___________ copy LDG ____________________________
        printf("_____________________copy LDG ___________________\n");
        testCUDA(cudaEventRecord(start,0));
        mergedSmall_k_ldg<<<1,1024>>>(thostA,thostB,thostM,sizeA,sizeB,sizeM);
        testCUDA(cudaEventRecord(stop,0));
        testCUDA(cudaEventSynchronize(stop));
        testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
        printf("elapsed time : %f ms\n",TimeVar);
        fprintf(f, "Ldg,MemCpy,%f\n",TimeVar);
        testCUDA(cudaMemcpy(M, thostM, sizeM*sizeof(int), cudaMemcpyDeviceToHost));
        cout<<"Check sorted : "<<is_sorted(M,sizeM)<<endl;
        //____________________________________________

        testCUDA(cudaUnbindTexture ( texture_referenceA ));
        testCUDA(cudaUnbindTexture ( texture_referenceB ));
        cudaFree(thostA);
        cudaFree(thostB);
        cudaFree(thostM);
        testCUDA(cudaFreeHost(hostA));
        testCUDA(cudaFreeHost(hostB));
        testCUDA(cudaFreeHost(hostM));
        fclose(f); 
    #endif

    //___________________________ Question 2_________________________________
    #if QUESTION==2
        FILE *f = fopen("../results/results2.csv", "w"); 
        fprintf(f, "Kernel,type,memory,time\n");
        int *__restrict__ path;
        int nb_threads = 128;
        int nb_blocks = (sizeM+nb_threads-1)/nb_threads;
        //if(sizeM<1024) nb_blocks=1024;
        int *hA,*hB,*hM;
        testCUDA(cudaMalloc((void **)&hA,sizeA*sizeof(int)));
        testCUDA(cudaMalloc((void **)&hB,sizeB*sizeof(int)));
        testCUDA(cudaMalloc((void **)&hM,sizeM*sizeof(int)));
        testCUDA(cudaMemcpy(hA, A, sizeA*sizeof(int), cudaMemcpyHostToDevice));
        testCUDA(cudaMemcpy(hB, B, sizeB*sizeof(int), cudaMemcpyHostToDevice));
        testCUDA(cudaMalloc((void **)&path,2*(nb_blocks+1)*sizeof(int)));

        //_________________________ Path and Merge global copy ___________________
        printf("__________________ Path big normal __________________\n");
        testCUDA(cudaEventRecord(start,0));
        pathBig_k<<<nb_blocks,nb_threads>>>(hA,hB,path,sizeA,sizeB,sizeM);
        testCUDA(cudaEventRecord(stop,0));
        testCUDA(cudaEventSynchronize(stop));
        testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
        printf("elapsed time : %f ms\n",TimeVar);
        fprintf(f, "Path,Normal,MemCopy,%f\n",TimeVar);
        //____________________________________________
        printf("__________________ Merg big normal _________________\n");
        testCUDA(cudaEventRecord(start,0));
        merged_Big_k<<<nb_blocks,nb_threads>>>(hA,hB,hM,path,sizeM);
        testCUDA(cudaEventRecord(stop,0));
        testCUDA(cudaEventSynchronize(stop));
        testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
        printf("elapsed time : %f ms\n",TimeVar);
        fprintf(f, "Merge,Normal,MemCopy,%f\n",TimeVar);
        testCUDA(cudaMemcpy(M, hM, sizeM*sizeof(int), cudaMemcpyDeviceToHost));
        cout<<"Check sorted : "<<is_sorted(M,sizeM)<<endl;

        //_________________________ Path and Merge zero copy ___________________
        int *hzA,*hzB,*hzM;
        testCUDA(cudaHostAlloc(&hzA,sizeA*sizeof(int),cudaHostAllocMapped)); //cudaHostAllocWriteCombined
        testCUDA(cudaHostAlloc(&hzB,sizeB*sizeof(int),cudaHostAllocMapped));
        hzA[0]=rand()%20;
        hzB[0]=rand()%20;
        for(int i=1;i<sizeA;i++){hzA[i]=hzA[i-1]+rand()%20+1;}
        for(int i=1;i<sizeB;i++){hzB[i]=hzB[i-1]+rand()%20+1;}
        testCUDA(cudaHostAlloc(&hzM,sizeM*sizeof(int),cudaHostAllocMapped));

        printf("__________________ Path big zero copy __________________\n");
        testCUDA(cudaEventRecord(start,0));
        pathBig_k<<<nb_blocks,nb_threads>>>(hzA,hzB,path,sizeA,sizeB,sizeM);
        testCUDA(cudaEventRecord(stop,0));
        testCUDA(cudaEventSynchronize(stop));
        testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
        printf("elapsed time : %f ms\n",TimeVar);
        fprintf(f, "Path,Normal,ZeroCpy,%f\n",TimeVar);
        //____________________________________________
        printf("__________________ Merg big zero copy _________________\n");
        testCUDA(cudaEventRecord(start,0));
        merged_Big_k<<<nb_blocks,nb_threads>>>(hzA,hzB,hzM,path,sizeM);
        testCUDA(cudaEventRecord(stop,0));
        testCUDA(cudaEventSynchronize(stop));
        testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
        printf("elapsed time : %f ms\n",TimeVar);
        fprintf(f, "Merge,Normal,ZeroCpy,%f\n",TimeVar);
        cout<<"Check sorted : "<<is_sorted(hzM,sizeM)<<endl;

        //_________________________ Path and Merge naive ___________________
        printf("__________________ Path big for naive merge __________________\n");
        testCUDA(cudaEventRecord(start,0));
        pathBig_k<<<nb_blocks,1>>>(hA,hB,path,sizeA,sizeB,sizeM);
        testCUDA(cudaEventRecord(stop,0));
        testCUDA(cudaEventSynchronize(stop));
        testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
        printf("elapsed time : %f ms\n",TimeVar);
        fprintf(f, "Path,Naive,MemCpy,%f\n",TimeVar);
        //____________________________________________
        printf("__________________ Merg big NAIVE_________________\n");
        testCUDA(cudaEventRecord(start,0));
        merged_Big_k_naive<<<nb_blocks,1>>>(hA,hB,hM,path,sizeM);
        testCUDA(cudaEventRecord(stop,0));
        testCUDA(cudaEventSynchronize(stop));
        testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
        printf("elapsed time : %f ms\n",TimeVar);
        fprintf(f, "Path,Naive,MemCpy,%f\n",TimeVar);
        testCUDA(cudaMemcpy(M, hM, sizeM*sizeof(int), cudaMemcpyDeviceToHost));
        cout<<"Check sorted : "<<is_sorted(M,sizeM)<<endl;

        //_________________________ Path and Merge naive zero copy___________________
        printf("__________________ Path big for naive merge zero copy___________________\n");
        testCUDA(cudaEventRecord(start,0));
        pathBig_k<<<nb_blocks,1>>>(hzA,hzB,path,sizeA,sizeB,sizeM);
        testCUDA(cudaEventRecord(stop,0));
        testCUDA(cudaEventSynchronize(stop));
        testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
        printf("elapsed time : %f ms\n",TimeVar);
        fprintf(f, "Path,Naive,ZeroCpy,%f\n",TimeVar);
        //____________________________________________
        printf("__________________ Merg big NAIVE zero copy__________________\n");
        testCUDA(cudaEventRecord(start,0));
        merged_Big_k_naive<<<nb_blocks,1>>>(hzA,hzB,hzM,path,sizeM);
        testCUDA(cudaEventRecord(stop,0));
        testCUDA(cudaEventSynchronize(stop));
        testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
        printf("elapsed time : %f ms\n",TimeVar);
        fprintf(f, "Merge,Naive,ZeroCpy,%f\n",TimeVar);
        cout<<"Check sorted : "<<is_sorted(hzM,sizeM)<<endl;
        fclose(f); 
    #endif

    //___________________________ Question 3_________________________________
    #if QUESTION==3
        int *__restrict__ hD;
        int *__restrict__ hsD;
        int *D  ;
        int *sD ;
        int padding = 0;
        //int nb_threads = 128; // changing it might be smart
        //int nb_blocks = (sizeM+nb_threads-1)/nb_threads;
        printf("__________________ sort M __________________\n");
        int threads_per_blocks = 128;
        FILE *f = fopen("../results/results3.csv", "w"); 
        fprintf(f, "d,time\n");
        for(int d=2;d<262144*2*2;d*=4){
            testCUDA(cudaMalloc((void **)&hsD,d*sizeof(int)));
            testCUDA(cudaMalloc((void **)&hD ,d*sizeof(int)));
            
            //code to launch on a size != than a power of 2
            if(d != 0 && (d & (d-1)) == 0){
                //printf("|M| is a power of 2\n");
                D  = (int *) malloc(d*sizeof(int));
                sD = (int *) malloc(d*sizeof(int));
                for(int i=0;i<d;i++){D[i]=rand()%d*50+1;}
            }
            else{
                //printf("|M| was not a power of 2, it will be changed\n");
                int power = 1;
                while(power < d) power*=2;
                //printf("new |M| with padding : %d\n",power);
                D  = (int *) malloc(power*sizeof(int));
                sD = (int *) malloc(power*sizeof(int));
                for(int i=0;i<d;i++){D[i]=rand()%d*5+1;}
                for(int i = d;i<power;i++){D[i] = ( int) -1 >> 1;}
                padding = power-d;
                d = power;
            }
            
            // printf("Assigning M\n");  
            testCUDA(cudaMemcpy(hD, D, d*sizeof(int), cudaMemcpyHostToDevice));
            testCUDA(cudaEventRecord(start,0));
            sort_array(hD,hsD,d,threads_per_blocks);
            testCUDA(cudaEventRecord(stop,0));
            testCUDA(cudaEventSynchronize(stop));
            testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
            printf("d = %10d | t =  %4.10f ms | ",d,TimeVar);
            fprintf(f, "%d,%f\n",d,TimeVar);
            testCUDA(cudaMemcpy(sD, hsD, d*sizeof(int), cudaMemcpyDeviceToHost));
            cout<<" Sorted : "<<is_sorted(sD,d);
            //____________________Compare with qsort ________________________
            clock_t begin = clock();
            qsort(D, d, sizeof(int), cmpfunc);
            clock_t end = clock();
            double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
            printf("\tquicksort t = %f ms | ",time_spent*1000);
            int sorted = 1;
            for(int i=0;i<d;i++) {
                if(D[i]!=sD[i]){
                    printf("ERROR    i=%d : %d != %d\n",i,D[i],sD[i]);
                    sorted = 0;
                    break;
                }
            }
            if(sorted) printf("arrays are equal\n");
            cudaFree(hD);
            cudaFree(hsD);
            free(D);
            free(sD);
        }
        fclose(f); 
    #endif
        
    //___________________________ Question 4_________________________________
    #if QUESTION==4
        FILE *f = fopen("../results/results4.csv", "w"); 
        fprintf(f, "type,memory,time\n");
        // N arrays containing Ai and Bi such as |Ai| + |Bi| = d
        // N arrays of size d
        int N = 10000; // max 1000000
        int d = 500; 
        if (argc > 2) {
            N = atoi(argv[1]);
            d = atoi(argv[2]);
            if(d>1024) d=1024; 
        } // If no arguments are provided, set random sizes
        printf("N = %d | d = %d\n",N,d);
        
        
        // ________________________________________Zero Copy______________________________________________ 

        printf("_______________________________Zero copy____________________________________\n");
        int* host_all_M;
        int* host_all_STM;
        int* host_all_size_A;
        int* host_all_size_B;

        // allocation on the device for save all size of Ai and Bi 
        // we choose a 1D representation,  we stocked Ai and Bi in one table M : M = (A1|B1|...|AN|BN) 
        testCUDA(cudaHostAlloc(&host_all_size_A,N*sizeof(int),cudaHostAllocMapped));
        testCUDA(cudaHostAlloc(&host_all_size_B,N*sizeof(int),cudaHostAllocMapped));

        // Initialisation of size Ai and Bi such as |Ai| + |Bi| = d 
        int size_all_A=0;
        int size_all_B=0;
        int sizeA;
        int sizeB;
        for(int i = 0;i<N;i++){ 
            sizeA = rand()%d+1;
            sizeB = (d-sizeA);
            host_all_size_A[i] = sizeA;
            host_all_size_B[i] = sizeB;
            size_all_A += sizeA;
            size_all_B +=sizeB;
        }

        // we stocked Ai and Bi in one table M : M = (A1|B1|...|AN|BN) 
        // allocation on device for M and STM of size N*d (N arrays of size d)
        // M will contains N arrays of Ai and Bi not sorted  
        // STM (Sorted M) will contains Mi sorted i.e Ai and Bi merge and sort 
        testCUDA(cudaHostAlloc(&host_all_STM,N*d*sizeof(int),cudaHostAllocMapped));    
        testCUDA(cudaHostAlloc(&host_all_M,N*d*sizeof(int),cudaHostAllocMapped));  

        // Start initialisation of the first arrays A0 and B0
        if(host_all_size_A[0]!=0){
            host_all_M[0]=rand()%20+1;
            for(int j = 1;j<host_all_size_A[0];j++){
                host_all_M[j]=host_all_M[j-1]+rand()%20+1;
            }
        }
        if(host_all_size_B[0]!=0){
            host_all_M[host_all_size_A[0]]=rand()%20+1;
            for(int j = host_all_size_A[0]+1;j<host_all_size_B[0]+host_all_size_A[0];j++){
                host_all_M[j]=host_all_M[j-1]+rand()%20+1;
            }
        }
            
        // Initialisation of all arrays 
        int tmp_A=host_all_size_A[0];
        int tmp_B=host_all_size_B[0];
        for(int i = 1;i<N;i++){ 
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
        
        printf("_________________ LDG_____________________\n");

        int numBlocks = N; //big number
        int threadsPerBlock = d; // multiple of d
        testCUDA(cudaEventRecord(start));
        mergeSmallBatch_k_ldg<<<numBlocks,threadsPerBlock>>>(host_all_M,host_all_STM,host_all_size_A,host_all_size_B,d);
        testCUDA(cudaEventRecord(stop));
        testCUDA(cudaEventSynchronize(stop));
        testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
        printf("elapsed time : %f ms\n",TimeVar);
        fprintf(f, "Ldg,ZeroCpy,%f\n",TimeVar);
        // _______________Check results_______________
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

        for(int i=0;i<d;i++){host_all_STM[i]=0;}

        printf("_________________ Shared_____________________\n");

        numBlocks = N; //big number
        threadsPerBlock = d; // multiple de d
        testCUDA(cudaEventRecord(start));
        mergeSmallBatch_k_shared<<<numBlocks,threadsPerBlock,d*sizeof(int)>>>(host_all_M,host_all_STM,host_all_size_A,host_all_size_B,d);
        testCUDA(cudaEventRecord(stop));
        testCUDA(cudaEventSynchronize(stop));
        testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
        printf("elapsed time : %f ms\n",TimeVar);
        fprintf(f, "Shared,ZeroCpy,%f\n",TimeVar);
        // _______________Check results_______________
        all_sorted=1;
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

        for(int i=0;i<d;i++){host_all_STM[i]=0;}

        printf("_________________ Normal_____________________\n");

        numBlocks = N; //big number
        threadsPerBlock = d; // multiple de d
        testCUDA(cudaEventRecord(start));
        mergeSmallBatch_k<<<numBlocks,threadsPerBlock>>>(host_all_M,host_all_STM,host_all_size_A,host_all_size_B,d);
        testCUDA(cudaEventRecord(stop));
        testCUDA(cudaEventSynchronize(stop));
        testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
        printf("elapsed time : %f ms\n",TimeVar);
        fprintf(f, "Normal,ZeroCpy,%f\n",TimeVar);
        // _______________Check results_______________
        all_sorted=1;
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

        // ________________________________________Copy______________________________________________ 

        printf("__________________________________Copy_______________________________________\n");

        int* all_M = (int *) malloc(N*d*sizeof(int));
        int* all_STM = (int *) malloc(N*d*sizeof(int));
        int* all_size_A = (int *) malloc(N*sizeof(int));
        int* all_size_B = (int *) malloc(N*sizeof(int));
        int* h_all_M;
        int* h_all_STM;
        int* h_all_size_A;
        int* h_all_size_B;

        // allocation on device for save size
        testCUDA(cudaMalloc((void **)&h_all_size_A,N*sizeof(int)));
        testCUDA(cudaMalloc((void **)&h_all_size_B,N*sizeof(int)));

        // Initialisation size
        size_all_A=0;
        size_all_B=0;
        for(int i = 0;i<N;i++){ 
            sizeA = rand()%d+1;
            sizeB = (d-sizeA);
            all_size_A[i] = sizeA;
            all_size_B[i] = sizeB;
            size_all_A += sizeA;
            size_all_B +=sizeB;
        }

        // copy of all size on device
        testCUDA(cudaMemcpy(h_all_size_A, all_size_A, N*sizeof(int), cudaMemcpyHostToDevice));
        testCUDA(cudaMemcpy(h_all_size_B, all_size_B, N*sizeof(int), cudaMemcpyHostToDevice));
        
        // allocation on device of M and STM
        testCUDA(cudaMalloc((void **)&h_all_M,N*d*sizeof(int)));
        testCUDA(cudaMalloc((void **)&h_all_STM,N*d*sizeof(int)));

        // Start initialisation of the first arrays A0 and B0
        if(all_size_A[0]!=0){
            all_M[0]=rand()%20+1;
            for(int j = 1;j<all_size_A[0];j++){
                all_M[j]=all_M[j-1]+rand()%20+1;
            }
        }
        if(all_size_B[0]!=0){
            all_M[all_size_A[0]]=rand()%20+1;
            for(int j = all_size_A[0]+1;j<all_size_B[0]+all_size_A[0];j++){
                all_M[j]=all_M[j-1]+rand()%20+1;
            }
        }
        tmp_A=all_size_A[0];
        tmp_B=all_size_B[0];

        // Initialisation of all arrays 
        for(int i = 1;i<N;i++){ 
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
        // copy all_M on h_all_M on device
        testCUDA(cudaMemcpy(h_all_M, all_M, N*d*sizeof(int), cudaMemcpyHostToDevice));

        printf("_________________ LDG_____________________\n");

        numBlocks = N; //big number
        threadsPerBlock = d; // multiple de d
        testCUDA(cudaEventRecord(start));
        mergeSmallBatch_k_ldg<<<numBlocks,threadsPerBlock>>>(h_all_M,h_all_STM,h_all_size_A,h_all_size_B,d);
        testCUDA(cudaEventRecord(stop));
        testCUDA(cudaEventSynchronize(stop));
        testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
        printf("elapsed time : %f ms\n",TimeVar);
        fprintf(f, "Ldg,Memcpy,%f\n",TimeVar);
        // retrieve STM on device
        testCUDA(cudaMemcpy(all_STM, h_all_STM, N*d*sizeof(int), cudaMemcpyDeviceToHost));

        // _______________Check results_______________
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

        for(int i=0;i<d;i++){all_STM[i]=0;}

        printf("_________________ Shared_____________________\n");

        numBlocks = N; //big number
        threadsPerBlock = d; // multiple de d
        testCUDA(cudaEventRecord(start));
        mergeSmallBatch_k_shared<<<numBlocks,threadsPerBlock,d*sizeof(int)>>>(h_all_M,h_all_STM,h_all_size_A,h_all_size_B,d);
        testCUDA(cudaEventRecord(stop));
        testCUDA(cudaEventSynchronize(stop));
        testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
        printf("elapsed time : %f ms\n",TimeVar);
        fprintf(f, "Shared,Memcpy,%f\n",TimeVar);
        testCUDA(cudaMemcpy(all_STM, h_all_STM, N*d*sizeof(int), cudaMemcpyDeviceToHost));

        // _______________Check results_______________
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

        for(int i=0;i<d;i++){all_STM[i]=0;}

        printf("_________________ Normal_____________________\n");

        numBlocks = N; //big number
        threadsPerBlock = d; // multiple of d
        testCUDA(cudaEventRecord(start));
        mergeSmallBatch_k<<<numBlocks,threadsPerBlock>>>(h_all_M,h_all_STM,h_all_size_A,h_all_size_B,d);
        testCUDA(cudaEventRecord(stop));
        testCUDA(cudaEventSynchronize(stop));
        testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
        printf("elapsed time : %f ms\n",TimeVar);
        fprintf(f, "Normal,Memcpy,%f\n",TimeVar);
        testCUDA(cudaMemcpy(all_STM, h_all_STM, N*d*sizeof(int), cudaMemcpyDeviceToHost));

        // _______________Check results_______________
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

        // test on quicksort sequential to compare 
        printf("______________________________Quicksort sequential___________________________\n");

        clock_t begin = clock();
        for(int i=0;i<N*d;i+=d)
            qsort(&all_M[i], d, sizeof(int), cmpfunc);
        clock_t end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        printf("elapsed time : %f ms\n",time_spent*1000);
        fprintf(f, "Quicksort,CPU,%f\n",TimeVar);
        // _______________Check results_______________
        all_sorted=1;
        for(int i = 0;i<N*d;i+=d){
            sorted = is_sorted(&all_M[i],d);
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

        // ______________________Clean Question 4_____________________________
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
        fclose(f); 
    #endif

    //___________________________ Question 5__________________________________
   
    #if QUESTION == 5
        // We chose to use copy because it's faster than zero copy
        FILE *f = fopen("../results/results5.csv", "w"); 
        fprintf(f, "N,d,time\n");
        // test for several value of N and d
        int Nmax = 1000000;
        if(argc == 2){
            if(atoi(argv[1])<Nmax && atoi(argv[1])> 1000 ) Nmax = atoi(argv[1]);
        }
        for(int N = 10; N<Nmax; N*=10){//10000000 max 
            for (int d = 2; d<=1024; d*=2){
                int* all_M = (int *) malloc(N*d*sizeof(int));
                int* all_STM = (int *) malloc(N*d*sizeof(int));
                int* all_size_A = (int *) malloc(N*sizeof(int));
                int* all_size_B = (int *) malloc(N*sizeof(int));
                int* h_all_M;
                int* h_all_STM;
                int* h_all_size_A;
                int* h_all_size_B;

                // allocation on device for save size
                testCUDA(cudaMalloc((void **)&h_all_size_A,N*sizeof(int)));
                testCUDA(cudaMalloc((void **)&h_all_size_B,N*sizeof(int)));

                // Initialisation size
                int size_all_A=0;
                int size_all_B=0;
                int sizeA;
                int sizeB;
                for(int i = 0;i<N;i++){ 
                    sizeA = rand()%d+1;
                    sizeB = (d-sizeA);
                    all_size_A[i] = sizeA;
                    all_size_B[i] = sizeB;
                    size_all_A += sizeA;
                    size_all_B +=sizeB;
                }

                // copy of all size on device
                testCUDA(cudaMemcpy(h_all_size_A, all_size_A, N*sizeof(int), cudaMemcpyHostToDevice));
                testCUDA(cudaMemcpy(h_all_size_B, all_size_B, N*sizeof(int), cudaMemcpyHostToDevice));
                
                // allocation on device of M and STM
                testCUDA(cudaMalloc((void **)&h_all_M,N*d*sizeof(int)));
                testCUDA(cudaMalloc((void **)&h_all_STM,N*d*sizeof(int)));

                // Start initialisation of the first arrays A0 and B0
                if(all_size_A[0]!=0){
                    all_M[0]=rand()%20+1;
                    for(int j = 1;j<all_size_A[0];j++){
                        all_M[j]=all_M[j-1]+rand()%20+1;
                    }
                }
                if(all_size_B[0]!=0){
                    all_M[all_size_A[0]]=rand()%20+1;
                    for(int j = all_size_A[0]+1;j<all_size_B[0]+all_size_A[0];j++){
                        all_M[j]=all_M[j-1]+rand()%20+1;
                    }
                }
                int tmp_A=all_size_A[0];
                int tmp_B=all_size_B[0];

                // Initialisation of all arrays 
                for(int i = 1;i<N;i++){ 
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
                // copy all_M on h_all_M on the device
                testCUDA(cudaMemcpy(h_all_M, all_M, N*d*sizeof(int), cudaMemcpyHostToDevice));

                int numBlocks = N; //big number
                int threadsPerBlock = d; // multiple of d
                testCUDA(cudaEventRecord(start));
                mergeSmallBatch_k_shared<<<numBlocks,threadsPerBlock,d*sizeof(int)>>>(h_all_M,h_all_STM,h_all_size_A,h_all_size_B,d);
                testCUDA(cudaEventRecord(stop));
                testCUDA(cudaEventSynchronize(stop));
                testCUDA(cudaEventElapsedTime(&TimeVar, start, stop));
                printf("elapsed time : N = %10d, d = %10d: %f ms\t",N,d,TimeVar);
                fprintf(f, "%d,%d,%f\n",N,d,TimeVar);
                testCUDA(cudaMemcpy(all_STM, h_all_STM, N*d*sizeof(int), cudaMemcpyDeviceToHost));

                // _______________Check results_______________
                int all_sorted=1;
                int sorted;
                for(int i = 0;i<N*d;i+=d){
                    sorted = is_sorted(&all_STM[i],d);
                    if(sorted ==0){
                        cout<<"Check sorted : "<<sorted<<endl;
                        all_sorted = 0;
                    }
                }
                if(all_sorted==1){
                    printf("Each array is sorted !\n");
                }
                else{
                    printf("There is an unsorted array !\n");
                }

                free(all_M);
                free(all_STM);
                free(all_size_A);
                free(all_size_B);
                testCUDA(cudaFree(h_all_M));
                testCUDA(cudaFree(h_all_STM));
                testCUDA(cudaFree(h_all_size_A));
                testCUDA(cudaFree(h_all_size_B));
            }
        }
        fclose(f); 
    #endif
    
    //___________ Cleaning up ____________________
    #if QUESTION == 2||QUESTION==1
    free(A);
    free(B);
    free(M);
    #endif 
	testCUDA(cudaEventDestroy(start));
    testCUDA(cudaEventDestroy(stop));
    cudaDeviceReset();
	return 0;
}
