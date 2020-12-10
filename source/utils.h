#ifndef UTILS_H
#define UTILS_H
void testCUDA(cudaError_t error, const char *file, int line) ;
void print_t(const int *seqM,const int sizeM);
int is_sorted(const int *seqM,const int sizeM);

int cmpfunc (const void * a, const void * b);
#endif