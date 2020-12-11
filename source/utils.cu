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
	if (error != cudaSuccess) {
	   printf("There is an error in file %s at line %d\n", file, line);
       printf("%s\n",cudaGetErrorString(error));
       exit(EXIT_FAILURE);
	} 
}
int is_sorted(const int *seqM,const int sizeM){
    int sorted =1;
    for(int i=0;i<sizeM-1;i++){
        if(seqM[i+1]<seqM[i]){
            printf("\n\terror index : %d, value = %d, next value = %d",i,seqM[i],seqM[i+1]);
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
    return ( *(int*)a - *(int*)b );
 }
 
void print_t(const int *seqM,const int sizeM){
    cout<<"[";
    for (int i = 0; i <sizeM-1; i++) 
        cout << seqM[i] << ',';
    cout<<seqM[sizeM-1]<<"]"<<endl;
    return;
}
