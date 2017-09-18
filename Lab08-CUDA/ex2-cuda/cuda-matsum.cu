/****************************************************************************
 *
 * cuda-matsum.cu - Dense matrix-matrix addition with CUDA
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
 * Simple implementation of dense square matrix-matrix addition with CUDA.
 *
 * Compile with:
 * nvcc cuda-matsum.cu -o cuda-matsum -lm
 *
 * Run with:
 * ./cuda-matsum
 *
 ****************************************************************************/

#include <stdio.h>
#include <math.h>

#define BLKSIZE 16

void matsum( float *p, float *q, float *r, int n )
{
    int i, j;
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            r[i*n + j] = p[i*n + j] + q[i*n + j];
        }
    }
}

/* Initialize square matrix p */
void fill( float *p, int n )
{
    int i;
    for (i=0; i<n*n; i++) {
        p[i] = i;
    }
}


__global__ void cudaMatsum(float *p, float *q, float *r, int n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if(row > n/BLKSIZE) {
    row = n/BLKSIZE;
    col += BLKSIZE;
  }

  int index = row *n + col;

  r[index] = p[index] + q[index];
}

int main( int argc, char* argv[] )
{
    float *p, *q, *r;
    int i, j, k, n = 1024;
    size_t size;

    if ( argc > 2 ) {
        printf("Usage: %s [n]\n", argv[0]);
        return -1;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    }

    size = n*n*sizeof(*p);

    /* Allocate space for host copies of p, q, r */
    p = (float*)malloc(size);
    fill(p, n);
    q = (float*)malloc(size);
    fill(q, n);
    r = (float*)malloc(size);

    float *d_p, *d_q, *d_r;

    cudaMalloc((void**)&d_p, size);
    cudaMalloc((void**)&d_q, size);
    cudaMalloc((void**)&d_r, size);

    cudaMemcpy(d_p, p, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q, size, cudaMemcpyHostToDevice);

    int blocks = (n*n) % BLKSIZE == 0 ? (n*n)/BLKSIZE : (n*n)/BLKSIZE + 1;

    printf("blocks = %d\n",blocks );

    cudaMatsum<<<blocks, BLKSIZE>>>(d_p, d_q, d_r, n);

    cudaMemcpy(r, d_r, size, cudaMemcpyDeviceToHost);

    //matsum(p, q, r, n);

    /* Check result */
    k = 0;
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            if (fabsf(r[i*n+j] - 2.0*k) > 1e-5) {
                printf("Check failed: r[%d][%d] = %f, expeted %f\n", i, j, r[i*n+j], 2.0*k);
                return -1;
            }
            k++;
        }
    }
    printf("Check OK\n");

    /* Cleanup */
    free(p);
    free(q);
    free(r);
    return 0;
}
