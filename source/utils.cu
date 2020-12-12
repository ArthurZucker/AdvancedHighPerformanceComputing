/**
 * @file utils.cu
 * @author Arthur Zucker & Clément Apavou  
 * @date 12 Dec 2020
 * @brief contains all usefull non-kernel function
 */

#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>  
#include <iostream>
#include <algorithm>
#include <iterator>
#include <assert.h>
#include <time.h>

using namespace std;
void testCUDA(cudaError_t error, const char *file, int line)  {
    /**
    * CUDA provided error verifier
    */
	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       printf("%s\n",cudaGetErrorString(error));
       exit(EXIT_FAILURE);
	} 
}

// check if the table seqM of size sizeM is sorted
int is_sorted(const int *seqM,const int sizeM){
    /**
    * Checks if an array is sorted
    * @param seqM is an array
    * @param sizeM is its size
    * @return 0 if not sorted, 1 if sorted
    * @note this function prints different information to check whether 
    * M is actually sorted or empty, and if it is empty it will show from where to where
    * If the array is not sorted, it keep on searching, in order to know if only half of it is sorted or not
    */
    int sorted =1;
    for(int i=0;i<sizeM-1;i++){
        if(seqM[i+1]<seqM[i]){
            printf("\terror index : %d, value = %d, next value = %d\t",i,seqM[i],seqM[i+1]);
            sorted =  0;
        }
        else if(seqM[i+1]==0){
            i++;
            printf("null value encountered");
            while(seqM[i+1]==0 && i<sizeM-1) i++;
            printf(" ... 0 : %d\n",i);
        }
    }
    return sorted;
    
}

int cmpfunc (const void * a, const void * b) {
    /**
    * Comparison function used for quick sort
    */
    return ( *(int*)a - *(int*)b );
 }
 

void print_t(const int *seqM,const int sizeM){
    /**
    * Prints the content of an array
    * @param seqM is an array
    * @param sizeM is its size
    */
    cout<<"[";
    for (int i = 0; i <sizeM-1; i++) 
        cout << seqM[i] << ',';
    cout<<seqM[sizeM-1]<<"]"<<endl;
    return;
}
