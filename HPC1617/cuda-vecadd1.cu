/****************************************************************************
 *
 * cuda-vecadd1.cu - Sum two arrays with CUDA, using thread blocks
 *
 * Written in 2017 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * To the extent possible under law, the author(s) have dedicated all 
 * copyright and related and neighboring rights to this software to the 
 * public domain worldwide. This software is distributed without any warranty.
 *
 * You should have received a copy of the CC0 Public Domain Dedication
 * along with this software. If not, see 
 * <http://creativecommons.org/publicdomain/zero/1.0/>. 
 *
 * ---------------------------------------------------------------------------
 *
 * Based on the example shown in the CUDA toolkit documentation
 * http://docs.nvidia.com/cuda/cuda-c-programming-guide/
 *
 * Compile with:
 * nvcc cuda-vecadd1.cu -o cuda-vecadd1
 *
 * Run with:
 * ./cuda-vecadd1
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>

__global__ void add( int *a, int *b, int *c )
{
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

void vec_init( int *a, int n )
{
    int i;
    for (i=0; i<n; i++) {
        a[i] = i;
    }
}

#define N 512

int main( void ) 
{
    int *a, *b, *c;	          /* host copies of a, b, c */ 
    int *d_a, *d_b, *d_c;	  /* device copies of a, b, c */
    int i;
    const size_t size = N*sizeof(int);
    /* Allocate space for device copies of a, b, c */
    cudaMalloc((void **)&d_a, size); 
    cudaMalloc((void **)&d_b, size); 
    cudaMalloc((void **)&d_c, size);
    /* Allocate space for host copies of a, b, c */
    a = (int*)malloc(size); vec_init(a, N);
    b = (int*)malloc(size); vec_init(b, N);
    c = (int*)malloc(size);
    /* Copy inputs to device */
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    /* Launch add() kernel on GPU */
    printf("Adding %d elements\n", N);
    add<<<N,1>>>(d_a, d_b, d_c);
    /* Copy result back to host */
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    /* Check results */
    for (i=0; i<N; i++) {
        if ( c[i] != a[i] + b[i] ) {
            printf("Error at index %d: a[%d]=%d, b[%d]=%d, c[%d]=%d\n",
                   i, i, a[i], i, b[i], i, c[i]);
            return -1;
        }
    }    
    printf("Test OK\n");
    /* Cleanup */
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
