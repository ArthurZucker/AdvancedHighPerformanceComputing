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

#define TEXTURE 1 //set to 0 to use normal memory, else it will use texture memory for A and B
// Function that catches the error 
void testCUDA(cudaError_t error, const char *file, int line)  {
	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       printf("%s\n",cudaGetErrorString(error));
       exit(EXIT_FAILURE);
	} 
}

// Has to be defined in the compilation in order to get the correct value 
// of the macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))



texture <int> texture_referenceA ;
texture <int> texture_referenceB ;

void merged_path_seq(int **A,int **B, int **M,int a, int b);
__global__ void merged_path_par(int **A,int **B, int **M);
__global__ void mergedSmall_k(int **A,int **B, int **M);
__global__ void mergeSmallBatch_k(int **A,int **B, int **M);
__device__ void pathBig_k(int **A,int **B, int **M);
__device__ void pathBig_k(int **A,int **B, int **M);

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
    
    if (argc < 3) {sizeA = rand()%1024;sizeB = rand()%1024-sizeA;} // If no arguments are provided, set random sizes
    else{sizeA=atoi(argv[1]);sizeB=atoi(argv[2]);}
    printf("|A| = %d, |B| = %d\n",sizeA,sizeB);
    int sizeM = sizeA+sizeB;
    int *hostA;
    int *hostB;
    int *hostM;
    
    
    
    //___________ TO DO: explain texture memory ___________
    #if TEXTURE == 1
    int *A = (int *) malloc(sizeA*sizeof(int));
    int *B = (int *) malloc(sizeB*sizeof(int));
    A[0]=rand()%20;
    B[0]=rand()%20;
    for(int i=1;i<sizeA;i++){A[i]=A[i-1]+rand()%20+1;}
    for(int i=1;i<sizeB;i++){B[i]=B[i-1]+rand()%20+1;}
    
    testCUDA(cudaMalloc((void **)&hostA,sizeA*sizeof(int)));
    testCUDA(cudaMalloc((void **)&hostB,sizeB*sizeof(int)));
    
    testCUDA(cudaMemcpy(hostA, A, sizeA*sizeof(int), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(hostB, B, sizeB*sizeof(int), cudaMemcpyHostToDevice));
    

    
    
    testCUDA (cudaBindTexture(0,texture_referenceA, hostA,sizeA*sizeof(int)));
    testCUDA (cudaBindTexture(0,texture_referenceB, hostB,sizeB*sizeof(int)));
    printf("texture step passed\n");
    #elif
    testCUDA(cudaHostAlloc(&hostA,sizeA*sizeof(int),cudaHostAllocWriteCombined));
    testCUDA(cudaHostAlloc(&hostB,sizeB*sizeof(int),cudaHostAllocWriteCombined));
    #endif
        // WC memory can be transferred across the PCI Express bus more quickly on some system configurations, but cannot be read 
        // efficiently by most CPUs. WC memory is a good option for buffers that will be written by the CPU and read by the device 
        // via mapped pinned memory or host->device transfers.
    testCUDA(cudaHostAlloc(&hostM,sizeM*sizeof(int),cudaHostAllocMapped)); // in order to do zero copy
    // alternative for M : testCUDA(cudaMalloc(&hostM,sizeM*sizeof(int)));
    
    
    //____________________________________________
    //___________ Initialize host table ___________
    
    
    // tex1Dfetch()
    
    
    //____________________________________________
    //___________ Cleaning up ____________________
    
    #if TEXTURE == 1
    testCUDA(cudaUnbindTexture ( texture_referenceA ));
    testCUDA(cudaUnbindTexture ( texture_referenceB ));
    #elif
    testCUDA(cudaFreeHost(hostA));
    testCUDA(cudaFreeHost(hostB));
    testCUDA(cudaFreeHost(hostM));
    #endif
    
    
    //____________________________________________
	return 0;
}

void merged_path_seq(int **A,int **B, int **M,int a, int b){
	int m = a+b;
	int i = 0;
	int j = 0;
	while(i+j<m){
		if(i>=a){
			M[i+j]=B[i];
			j++;}
		else if(j>=m ||A[i]<B[j]){
			M[i+j]=A[i];
			i++;}
		else{
			M[i+j]=B[i];
			j++;}
	}      
	return;
}
__global__ void mergedSmall_k(int **A,int **B, int **M){
    int i = threadIdx.x;
	return;       
}