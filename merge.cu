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

// Function that catches the error 
void testCUDA(cudaError_t error, const char *file, int line)  {
	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       exit(EXIT_FAILURE);
	} 
}

// Has to be defined in the compilation in order to get the correct value 
// of the macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))





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
    int *hostA = NULL;
    int *hostB = NULL;
    int *hostC = NULL;
    
    //____________________________________________
    //___________ Initialize host table ___________
    
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