/****************************************************************************
 *
 * cuda-reduce01cu - Reduction with CUDA
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
 * This program realizes a not-so-simple sum-reduction on the
 * GPU. Each thread block copies a portion of the array in shared
 * memory; then, all threads within the same block cooperate to
 * compute the sum of the local data by organizing the computation as
 * a tree. The final reduction is executed on the CPU.
 *
 * Compile with:
 * nvcc cuda-reduce1.cu -o cuda-reduce1
 *
 * Run with:
 * ./cuda-reduce1
 *
 ****************************************************************************/

#include <stdio.h>
#include <assert.h>

#define BLKSIZE 512
#define N_OF_BLOCKS 1024
/* N must be an integer multiple of BLKSIZE */
#define N ((N_OF_BLOCKS)*(BLKSIZE))

/* d_sums is an array of N_OF_BLOCKS integers that reside in device
   memory; therefore, there is no need to cudaMalloc'ate it */
__device__ int d_sums[N_OF_BLOCKS];
int h_sums[N_OF_BLOCKS];

/* This kernel copies a portion of array a[] of n elements into
   thread-local shared memory. All threads cooperate to compute the
   sum of the local data; thread 0 stores the computed value on the
   appropriate entry of d_sums[]. Different thread blocks access
   different elements of d_sums[], so no race condition is
   possible. */
__global__ void sum( int *a, int n )
{
    __shared__ int temp[BLKSIZE];
    int lindex = threadIdx.x;
    int bindex = blockIdx.x;
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int bsize = blockDim.x / 2;

    temp[lindex] = a[gindex];

    /* wait for all threads to finish the copy operation */
    __syncthreads(); 

    /* All threads within the block cooperate to compute the local sum */
    while ( bsize > 0 ) {
        if ( lindex < bsize ) {
            temp[lindex] += temp[lindex + bsize];
        }
	bsize = bsize / 2;
        /* threads must synchronize before performing the next
           reduction step */
        __syncthreads(); 
    }

    if ( 0 == lindex ) {
        d_sums[bindex] = temp[0];
    }
}

int main( void ) 
{
    int *h_a;
    int *d_a;
    int i, s=0;
    assert( 0 == N % BLKSIZE );
    /* Allocate space for device copies of d_a */
    cudaMalloc((void **)&d_a, N*sizeof(int));
    /* Allocate space for host copies of h_a */
    h_a = (int*)malloc(N * sizeof(int));
    /* Set all elements of vector h_a to 2, so that we know that the
       result of the sum must be 2*N */
    for (i=0; i<N; i++) {
        h_a[i] = 2;
    }
    /* Copy inputs to device */
    cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);
    /* Launch sum() kernel on the GPU */
    sum<<<N_OF_BLOCKS, BLKSIZE>>>(d_a, N);
    /* Copy the d_sums[] array from device memory to host memory h_sums[] */
    cudaMemcpyFromSymbol(h_sums, d_sums, N_OF_BLOCKS*sizeof(int));
    /* Perform the final reduction on the CPU */
    s = 0;
    for (i=0; i<N_OF_BLOCKS; i++) {
        s += h_sums[i];
    }
    /* Check result */
    if ( s != 2*N ) {
        printf("Check failed: Expected %d, got %d\n", 2*N, s);
        return -1;
    }
    printf("Check OK: computed sum = %d\n", s);
    /* Cleanup */
    free(h_a);
    cudaFree(d_a);
    return 0;
}
