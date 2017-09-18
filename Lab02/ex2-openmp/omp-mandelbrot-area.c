/******************************************************************************
 *
 *  PROGRAM: Mandelbrot area
 *
 *  PURPOSE: Program to compute the area of a  Mandelbrot set.
 *           Correct answer should be around 1.510659.
 *           WARNING: this program may contain errors
 *
 *  USAGE:   Program runs without input ... just run the executable
 *            
 *  HISTORY: Written:  (Mark Bull, August 2011).
 *           Changed "complex" to "d_complex" to avoid collsion with 
 *           math.h complex type (Tim Mattson, September 2011)
 *           Code cleanup (Moreno Marzolla, February 2017)
 *
 * --------------------------------------------------------------------------
 *
 * Compile with:
 * gcc -fopenmp -std=c99 -Wall -Wpedantic omp-mandelbrot-area.c -o omp-mandelbrot-area
 * 
 * Run with:
 * ./omp-mandelbrot-area
 *
 ******************************************************************************/

#include <stdio.h>
#include <omp.h>

#define NPOINTS 1000
#define MAXITER 10000

struct d_complex {
    double r;
    double i;
};

/*
 * Does the iteration z=z*z+c, until |z| > 2 when point is known to be
 * outside set If loop count reaches MAXITER, point is considered to
 * be inside the set. Returns 1 iff outside the set.
 */
int outside(struct d_complex c)
{
    struct d_complex z = c;
    int iter;
    double temp;
    
    #pragma omp parallel for
    for (iter=0; iter<MAXITER; iter++){
        temp = (z.r*z.r)-(z.i*z.i)+c.r;
        z.i = z.r*z.i*2+c.i;
        z.r = temp;
        if ((z.r*z.r+z.i*z.i)>4.0) {
            return 1;
        }
    }
    return 0;
}

int main( void ) 
{
    int i, j, numoutside = 0;
    double area, error;
    const double eps = 1.0e-5;
    double tstart, tend;

    /*
     * Loop over grid of points in the complex plane which contains
     * the Mandelbrot set, testing each point to see whether it is
     * inside or outside the set.
     *
     * This loop should be parallelized using OpenMP.
     */
    tstart = omp_get_wtime();
    #pragma omp parallel for collapse(2)
    for (i=0; i<NPOINTS; i++) {
        for (j=0; j<NPOINTS; j++) {
            struct d_complex c;
            c.r = -2.0+2.5*(double)(i)/(double)(NPOINTS)+eps;
            c.i = 1.125*(double)(j)/(double)(NPOINTS)+eps;
            numoutside += outside(c);
        }
    }
    tend = omp_get_wtime();
    
    /* Calculate area of set and error estimate and output the results */  
    area = 2.0*2.5*1.125*(double)(NPOINTS*NPOINTS-numoutside)/(double)(NPOINTS*NPOINTS);
    error = area/(double)NPOINTS;
    
    printf("Elapsed time %f\n", tend - tstart);
    printf("Area of Mandlebrot set = %12.8f +/- %12.8f\n", area, error);
    printf("Correct answer should be around 1.510659\n");
    return 0;
}


