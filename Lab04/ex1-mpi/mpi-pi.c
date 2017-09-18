/****************************************************************************
 *
 * mpi-pi.c - Skeleton to compute the approximate value of PI using MPI
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
 * Compile with:
 * mpicc -Wall mpi-pi.c -o mpi-pi
 *
 * Run with:
 * mpirun -n 4 ./mpi-pi
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/* Generate |n| random points within the square with corners (-1, -1),
   (1, 1); return the number of points that fall inside the circle
   centered ad the origin with radius 1 */
int generate_points( int n ) 
{
    int i, n_inside = 0;
    double x, y;
    for (i=0; i<n; i++) {
        x = (rand()/(double)RAND_MAX * 2.0) - 1.0;
        y = (rand()/(double)RAND_MAX * 2.0) - 1.0;
        if ( x*x + y*y < 1.0 ) {
            n_inside++;
        }
    }
    return n_inside;
}

int main( int argc, char *argv[] )
{
    int my_rank, comm_sz, inside;
    int npoints = 1000000;
    double pi_approx;

    MPI_Init( &argc, &argv );	
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &comm_sz );

    if ( argc > 1 ) {
        npoints = atoi(argv[1]);
    }

    /* Initialize the pseudo-random number generator on each process */
    srand(11 + my_rank);

    /* This is not a true parallel version; the master does
       everything */
    if ( 0 == my_rank ) {
        inside = generate_points(npoints);
        pi_approx = 4.0 * inside / (double)npoints;
        printf("PI approximation is %f (true value is about 3.14159265358979323846)\n", pi_approx);
    }
    MPI_Finalize();		
    return 0;
}
