/**************************************************************
This code is an implementation of the merging of two arrays
as describes in the subject
Both the sequential and parralele versions will be detailed in 
order to asses the performances
***************************************************************/
#include <stdio.h>






void merged_path_seq(int **A,int **B, int **M,int a, int b);
__global__ void merged_path_par(int **A,int **B, int **M);
__global__ void mergedSmall_k(int **A,int **B, int **M);
__device__ void pathBig_k(int **A,int **B, int **M);
__device__ void pathBig_k(int **A,int **B, int **M);

int main() {
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
	}
}

void merged_path_seq(int **A,int **B, int **M,int a, int b){
	int m = a+b;
	int i = 0;
	int j = 0;
	while(i+j<m):
		if(i>=a):
			M[i+j]=B[i];
			j++;
		else if(j>=n ||A[i]<B[j):
			M[i+j]=A[i];
			i++;
		else:
			M[i+j]=B[i];	
			j++;
	return
}