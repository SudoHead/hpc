/****************************************************************************
 *
 * cuda-dot.cu - Dot product with CUDA
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
 * Compile with:
 * nvcc cuda-dot.cu -o cuda-dot -lm
 *
 * Run with:
 * ./cuda-dot [len]
 *
 * Example:
 * ./cuda-dot
 *
 ****************************************************************************/
#include <stdio.h>
#include <math.h>

#define BLKSIZE 512

float dot( float *x, float *y, int n )
{
    int i;
    float result = 0.0;
    for (i = 0; i < n; i++) {
        result += x[i] * y[i];
    }
    return result;
}

__device__ float tmp[BLKSIZE];

__global__ void dotKer(float *x, float *y, float *res, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < BLKSIZE) {
        float r = 0;
        for(int i = 0 ; i < n/BLKSIZE ; i++) {
            r += x[index + i*BLKSIZE] * y[index + i*BLKSIZE];
        }
        tmp[index] = r;
	}
}

void vec_init( float *x, float *y, int n )
{
    int i;
    for (i=0; i<n; i++) {
        x[i] = i + 1.0;
        y[i] = 1.0 / x[i];
    }
}

int main( int argc, char* argv[] ) 
{
    float *x, *y, result=0;
    const int default_len = 1024*1024;
    int n;

    if ( argc > 2 ) {
        printf("\nUsage: %s [len]\n\nCompute the dot product of two arrays of length \"len\" (default: %d)\n\n", argv[0], default_len);
        return -1;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    } else {
        n = default_len;
    }

    const size_t size = n*sizeof(*x);


    /* Allocate space for host copies of x, y */
    x = (float*)malloc(size);
    y = (float*)malloc(size);
    vec_init(x, y, n);
	
	float *d_x, *d_y, *d_result;


        cudaMalloc((void**)&d_x, size);
        cudaMalloc((void**)&d_y, size);
        cudaMalloc((void**)&d_result, BLKSIZE*sizeof(float));

        cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);


	dotKer<<<1,BLKSIZE>>>(d_x, d_y, d_result, n);

	//cudaMemcpy(&result, d_result, size, cudaMemcpyDeviceToHost);

	float *tmp_cpu = (float*)malloc(BLKSIZE*sizeof(float));

	cudaMemcpyToSymbol(tmp_cpu, tmp, BLKSIZE*sizeof(float));

    //cudaMemcpy(tmp_cpu, d_result, BLKSIZE*sizeof(float), cudaMemcpyDeviceToHost);

	for(int i =0; i< BLKSIZE; i++) {
		result+=tmp_cpu[i];
	}

    /* Check result */
    printf("Result: %f\n", result);
    if ( fabs(result - n) < 1e-5 ) {
        printf("Check OK\n");
    } else {
        printf("Check failed: got %f, expected %f\n", result, (float)n);
    }

    /* Cleanup */
    free(x); 
    free(y);
    return 0;
}
