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
    for(int i=0;i<sizeM-1;i++){
        if(seqM[i+1]<seqM[i])return 0;
    }
     return 1;
    
}

void print_t(const int *seqM,const int sizeM){
    cout<<"[";
    for (int i = 0; i <sizeM-1; i++) 
        cout << seqM[i] << ',';
    cout<<seqM[sizeM-1]<<"]"<<endl;
    return;
}
