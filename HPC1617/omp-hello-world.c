/*
 * omp-hello-world.c - First OpenMP demo
 *
 * Compile with:
 * gcc -fopenmp omp-hello-world.c -o omp-hello-world
 *
 * Run with:
 * OMP_NUM_THREADS=10 ./omp-hello-world
 *
 * Author: Moreno Marzolla <moreno.marzolla(at)unibo.it>
 * Last modified: 2016/10/24
 * License: CC0 1.0 Public Domain Dedication 
 *          http://creativecommons.org/publicdomain/zero/1.0/
 */

#include <stdio.h>

int main( void )
{
    #pragma omp parallel
    {
	printf("Hello, world!\n");
    }

    return 0;
}
