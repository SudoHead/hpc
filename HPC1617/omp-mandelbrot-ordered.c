/****************************************************************************
 *
 * mandelbrot.c - displays the Mandelbrot set using ASCII characters
 *
 * Written in 2016, 2017 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * To the extent possible under law, the author(s) have dedicated all 
 * copyright and related and neighboring rights to this software to the 
 * public domain worldwide. This software is distributed without any warranty.
 *
 * You should have received a copy of the CC0 Public Domain Dedication
 * along with this software. If not, see 
 * <http://creativecommons.org/publicdomain/zero/1.0/>. 
 *
 * --------------------------------------------------------------------------
 *
 * This program computes and displays the Mandelbrot set using ASCII characters.
 * Since characters are displayed one at a time, 
 *
 * Compile with
 * gcc -std=c99 -Wall -Wpedantic -fopenmp omp-mandelbrot-ordered -o omp-mandelbrot-ordered
 *
 * and run with:
 * OMP_NUM_THREADS=4 ./omp-mandelbrot-ordered
 *
 ****************************************************************************/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>

const int maxit = 500000; /* A very large value just to make the program spend more time */
const int xsize = 78, ysize = 62;

/*
 * Iterate the recurrence:
 *
 * z_0 = 0;
 * z_{n+1} = z_n^2 + cx + i*cy;
 *
 * Returns the first n such that z_n > |bound|, or |maxit| if z_n is below
 * |bound| after |maxit| iterations.
 */
int iterate( float cx, float cy )
{
    float x = cx, y = cy;
    int it;
    for ( it = 0; (it < maxit) && (x*x + y*y < 2*2); it++ ) {
        float xx = x*x - y*y + cx;
        y = 2.0*x*y + cy;
        x = xx;
    }
    return it;
}

int main( int argc, char *argv[] )
{
    int x, y;
    float tstart, elapsed;
    const char charset[] = ".,c8M@jawrpogOQEPGJ";
    
    tstart = hpc_gettime();
#pragma omp parallel for private(x,y) collapse(2) ordered
    for ( y = 0; y < ysize; y++ ) {
        for ( x = 0; x < xsize; x++ ) {
            float cx = -2.5 + 3.5 * (float)x / (xsize - 1);
            float cy = 1 - 2.0 * (float)y / (ysize - 1);
            int v = iterate(cx, cy);
#pragma omp ordered 
            {
                char c = ' ';
                if (v > 0 && v < maxit) {
                    c = charset[v % (sizeof(charset)-1)];
                }
                putchar(c);
                if (x+1 == xsize) puts("|");
            }            
        }
    }
    elapsed = hpc_gettime() - tstart;
    printf("Elapsed time %f\n", elapsed);
    printf("Click to finish\n");
    return 0;
}
