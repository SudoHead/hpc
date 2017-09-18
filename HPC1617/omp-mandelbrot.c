/****************************************************************************
 *
 * omp-mandelbrot.c - displays the Mandelbrot set
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
 * This program computes and display the Mandelbrot set. This program
 * requires the gfx library from
 * http://www.nd.edu/~dthain/courses/cse20211/fall2011/gfx (the
 * library should be included in the archive containing this source file)
 *
 * Compile with
 * gcc -std=c99 -Wall -Wpedantic -fopenmp omp-mandelbrot.c gfx.c -o omp-mandelbrot -lX11
 *
 * and run with:
 * OMP_NUM_THREADS=4 ./omp-mandelbrot
 *
 * At the end, click the left mouse button to close the graphical window.
 *
 ****************************************************************************/

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include "gfx.h"

const int maxit = 10000;
const int xsize = 800, ysize = 600;

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
    float x = 0.0, y = 0.0, xx;
    int it;
    for ( it = 0; (it < maxit) && (x*x + y*y < 2*2); it++ ) {
	xx = x*x - y*y + cx;
	y = 2.0*x*y + cy;
	x = xx;
    }
    return it;
}

/*
 * Draw a pixel at window coordinates (x, y) with the appropriate
 * color; (0,0) is the upper left corner of the window, y grows
 * downward.
 */
void drawpixel( int x, int y )
{
    float cx = -2.5 + 3.5 * (float)x / (xsize - 1);
    float cy = 1 - 2.0 * (float)y / (ysize - 1);
    int v = iterate( cx, cy);
/* only one thread should access the display at any given time */
#pragma omp critical 
    {
	int c = 255-(int)(255.0 * v / maxit);
	gfx_color( 0, c, c );
	gfx_point( x, y );
    }
}

int main( int argc, char *argv[] )
{
    int x, y;
    double tstart, elapsed;

    gfx_open( xsize, ysize, "Mandelbrot Set");
    tstart = hpc_gettime();
/* #pragma omp parallel for private(x,y) schedule(dynamic,64) */
#pragma omp parallel for private(x,y) 
    for ( y = 0; y < ysize; y++ ) {
	for ( x = 0; x < xsize; x++ ) {
	    drawpixel( x, y );
	}
    }
    elapsed = hpc_gettime() - tstart;
    printf("Elapsed time %f\n", elapsed);
    printf("Click to finish\n");
    gfx_wait();
    return 0;
}
