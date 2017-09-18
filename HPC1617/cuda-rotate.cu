/****************************************************************************
 *
 * cuda-rotate.cu - Rotate N points using constant memory
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
 * nvcc cuda-rotate.cu -o cuda-rotate -lm
 *
 * Run with:
 * ./cuda-rotate
 *
 ****************************************************************************/
#include <stdio.h>
#include <math.h>

#define PLANE_ANG 360
#define BLKSIZE 256

__constant__ float c_sin[PLANE_ANG];
__constant__ float c_cos[PLANE_ANG];

/* Rotate all points of coords (px[i], py[i]) through an angle |angle|
   counterclockwise around the origin. 0 <= |angle| <= 359 */
__global__ void rotate_kernel(float* px, float *py, int n, int angle)  
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    angle = angle % PLANE_ANG; /* ensures 0 <= angle <= 359 */
    if (index < n ) {
        /* compute coordinates (prx, pry) of the rotated point (px[i], py[i]) */
        float prx = px[index] * c_cos[angle] - py[index]*c_sin[angle];
        float pry = py[index] * c_cos[angle] + px[index]*c_sin[angle];
        px[index] = prx;
        py[index] = pry;
    }
}

int main(int argc, char* argv[])
{
    float sin_table[PLANE_ANG], cos_table[PLANE_ANG];
    float *h_px, *h_py; /* coordinates in host memory */
    float *d_px, *d_py; /* coordinates in device memory */
    int i, a;
    const size_t NPOINTS = 1024*1024;
    const int ANGLE = 72;

    /* pre-compute the table of sin and cos; note that sin() and cos()
       expect the angle to be in radians */
    for (a=0; a<PLANE_ANG; a++) {
        sin_table[a] = sin(a * M_PI / 180.0f );
        cos_table[a] = cos(a * M_PI / 180.0f );
    }

    /* Ship the pre-computed tables to constant memory */
    cudaMemcpyToSymbol(c_sin, sin_table, sizeof(sin_table));
    cudaMemcpyToSymbol(c_cos, cos_table, sizeof(cos_table));

    const size_t size = NPOINTS * sizeof(*h_px);

    /* Create NPOINTS instances of the point (1,0) */
    h_px = (float*)malloc( size );
    h_py = (float*)malloc( size );
    for (i=0; i<NPOINTS; i++) {
        h_px[i] = 1.0f;
        h_py[i] = 0.0f;
    }

    /* Copy the points to device memory */
    cudaMalloc((void**)&d_px, size);
    cudaMalloc((void**)&d_py, size);
    cudaMemcpy(d_px, h_px, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, h_py, size, cudaMemcpyHostToDevice);

    /* Rotate all points by 72 degrees counterclockwise on the GPU */
    rotate_kernel<<< (NPOINTS + BLKSIZE-1)/BLKSIZE, BLKSIZE >>>(d_px, d_py, NPOINTS, ANGLE);

    /* Copy result back to host memory */
    cudaMemcpy(h_px, d_px, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_py, d_py, size, cudaMemcpyDeviceToHost);

    /* Check results */
    for (i=0; i<NPOINTS; i++) {
        if ( fabs(h_px[i] - cos_table[ANGLE]) > 1e-5 ||
             fabs(h_py[i] - sin_table[ANGLE]) > 1e-5 ) {
            printf("Test failed: (h_px[%d], h_py[%d]) expected (%f, %f) but got (%f, %f)\n",
                   i, i, cos_table[ANGLE], sin_table[ANGLE], h_px[i], h_py[i]);
            return -1;
        }
    }
    printf("Test OK\n");

    /* free memory */
    free(h_px);
    free(h_py);
    cudaFree(d_px);
    cudaFree(d_py);
    return 0;
}


