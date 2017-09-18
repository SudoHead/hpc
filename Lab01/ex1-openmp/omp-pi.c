/****************************************************************************
 *
 * omp-pi.c - Compute the approximate value of PI
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
 * --------------------------------------------------------------------------
 *
 * This program implements a serial algorithm for computing the
 * approximate value of PI using a Monte Carlo method.
 *
 * This file uses some C99 idioms (specifically, the %zu printf modifier
 * that is the portable way to print a value of type size_t). Therefore,
 * this file must be compiled with:
 * gcc -std=c99 -fopenmp -Wall -pedantic omp-pi.c -o omp-pi -lm
 *
 * After parallelization, it can be run with:
 * OMP_NUM_THREADS=4 ./omp-pi 20000
 *
 ****************************************************************************/

/* The rand_r function is only available if _XOPEN_SOURCE is defined */
#define _XOPEN_SOURCE
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* for fabs */

/* Generate |n| random points inside the square with corners (-1, -1),
   (1, 1); return the number of points that fall inside the circle
   centered ad the origin with radius 1. */
size_t generate_points( size_t n ) 
{
    size_t i, n_inside = 0;
    /* The C function rand() is not thread-safe, and therefore can not
       be used with OpenMP. We use rand_r with an explicit
       seed which is maintained by each thread. */
    unsigned int my_seed = 17 + omp_get_thread_num();
    for (i=0; i<n; i++) {
        double x = (rand_r(&my_seed)/(double)RAND_MAX * 2.0) - 1.0;
        double y = (rand_r(&my_seed)/(double)RAND_MAX * 2.0) - 1.0;
        if ( x*x + y*y < 1.0 ) {
            n_inside++;
        }
    }
    return n_inside;
}

int main( int argc, char *argv[] )
{
    size_t npoints = 10000, ninside;
    double pi_approx;
    const double pi_exact = 3.14159265358979323846;

    if ( argc > 2 ) {
        printf("Usage: %s [npoints]\n", argv[0]);
        return -1;
    }

    if ( argc == 2 ) {
        npoints = atol(argv[1]);
    }
     
    printf("Generating %zu points...\n", npoints);
    ninside = generate_points(npoints);
    pi_approx = 4.0 * ninside / (double)npoints;
    printf("PI approximation %f, exact %f, error %f%%\n", pi_approx, pi_exact, 100.0*fabs(pi_approx - pi_exact)/pi_exact);

    return 0;
}
