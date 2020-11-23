#ifndef UTILS_H
#define UTILS_H
void testCUDA(cudaError_t error, const char *file, int line) ;
void print_t(int *seqM,int sizeM);
int is_sorted(int *seqM,int sizeM);
#endif