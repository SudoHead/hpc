/****************************************************************************
 *
 * cuda-vecadd0.cu - Sum two integers with CUDA
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
 * nvcc cuda-vecadd0.cu -o cuda-vecadd0
 *
 * Run with:
 * ./cuda-vecadd0
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>

__global__ void add( int *a, int *b, int *c )
{
    *c = *a + *b;
}

int main( void ) 
{
    int a, b, c;	          /* host copies of a, b, c */ 
    int *d_a, *d_b, *d_c;	  /* device copies of a, b, c */
    const size_t size = sizeof(int);
    /* Allocate space for device copies of a, b, c */
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    /* Setup input values */
    a = 2; b = 7;
    /* Copy inputs to device */
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);
    /* Launch add() kernel on GPU */
    add<<<1,1>>>(d_a, d_b, d_c);
    /* Copy result back to host */
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
    /* check result */
    if ( c != a + b ) {
        printf("Expected %d, got %d\n", a+b, c);
        return -1;
    } else {
        printf("Test OK\n");
    }
    /* Cleanup */
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
