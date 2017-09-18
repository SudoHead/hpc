/****************************************************************************
 *
 * cuda-stencil1d.cu - 1D stencil example with CUDA
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
 * This implementation does not used per-block shared memory, and therefore
 * is less efficient than the version which does (cuda-stencil1d-shared.cu).
 *
 * Compile with:
 * nvcc cuda-stencil1d.cu -o cuda-stencil1d
 *
 * Run with:
 * ./cuda-stencil1d
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define BLKSIZE 512
#define RADIUS 3
#define N (BLKSIZE*1024)

__global__ void stencil1d(int *in, int *out) 
{
    int index = threadIdx.x + blockIdx.x * blockDim.x + RADIUS;
    int result = 0, offset;
    for (offset = -RADIUS ; offset <= RADIUS ; offset++) {
        result += in[index + offset];
    }
    /* Store the result */
    out[index] = result;
}

int main( void ) 
{
    int *h_in, *h_out;	  /* host copies of in and out */
    int *d_in, *d_out;	  /* device copies of in and out */
    int i;
    const size_t size = (N+2*RADIUS)*sizeof(int);

    assert( N % BLKSIZE == 0 );

    /* Allocate space for device copies of d_in and d_out */
    cudaMalloc((void **)&d_in, size);
    cudaMalloc((void **)&d_out, size);
    /* Allocate space for host copies of h_in and h_out */
    h_in = (int*)malloc(size);
    h_out = (int*)malloc(size);
    /* Set all elements of vector h_in to one */
    for (i=0; i<N+2*RADIUS; i++) {
        h_in[i] = 1;
    }
    /* Copy inputs to device */
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    /* Launch stencil1d() kernel on GPU */
    stencil1d<<<(N + BLKSIZE-1)/BLKSIZE, BLKSIZE>>>(d_in, d_out);
    /* Copy result back to host */
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    /* Check the result */
    for (i=RADIUS; i<N+RADIUS; i++) {
        if ( h_out[i] != 7 ) {
            printf("Error at index %d: h_out[%d] == %d != 7\n", i, i, h_out[i]);
            return -1;
        }
    }
    printf("Test OK\n");
    /* Cleanup */
    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
