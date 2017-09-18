/****************************************************************************
 *
 * mpi-vecsum.c - Parallel vector sum using MPI. This program requires
 * that the vector size is a multiple of the number of MPI processes.
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
 * Compile with:
 * mpicc -std=c99 -Wall -Wpedantic mpi-vecsum.c -o mpi-vecsum
 *
 * Run with:
 * mpirun -n 4 ./mpi-vecsum
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

/*
 * Compute z[i] = x[i] + y[i], i=0, ... n-1
 */
void sum( double* x, double* y, double* z, int n )
{
    int i;
    for (i=0; i<n; i++) {
	z[i] = x[i] + y[i];
    }
}

int main( int argc, char* argv[] )
{
    double *x, *local_x, *y, *local_y, *z, *local_z;
    int n, local_n, i;
    int my_rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    if ( 2 == argc ) {
	n = atoi(argv[1]);
    } else {
	n = comm_sz * 1000;
    }

    /* MPI_Scatter requires that all blocks have the same size */
    if ( (0 == my_rank) && (n % comm_sz) ) {
	printf("Error: the vector size (%d) must be multiple of %d\n", n, comm_sz);
	MPI_Abort( MPI_COMM_WORLD, -1 );
    }

    local_n = n / comm_sz;

    x = y = z = NULL;

    if ( 0 == my_rank ) {
	/* The master allocates the vectors */
	x = (double*)malloc( n * sizeof(*x) );
	y = (double*)malloc( n * sizeof(*y) );
	z = (double*)malloc( n * sizeof(*z) );
	for ( i=0; i<n; i++ ) {
	    x[i] = i;
	    y[i] = n-1-i;
	}
    }

    /* All nodes (including the master) allocate the local vectors */
    local_x = (double*)malloc( local_n * sizeof(*local_x) );
    local_y = (double*)malloc( local_n * sizeof(*local_y) );
    local_z = (double*)malloc( local_n * sizeof(*local_z) );

    /* Scatter vector x */
    MPI_Scatter( x,		/* sendbuf */
		 local_n,	/* sendcount; how many elements to send to _each_ destination */
		 MPI_DOUBLE,	/* sent MPI_Datatype */
		 local_x,	/* recvbuf */
		 local_n,	/* recvcount (usually equal to sendcount) */
		 MPI_DOUBLE,	/* received MPI_Datatype */
		 0,		/* root */
		 MPI_COMM_WORLD /* communicator */
		 );

    /* Scatter vector y*/
    MPI_Scatter( y,		/* sendbuf */
		 local_n,	/* sendcount; how many elements to send to _each_ destination */
		 MPI_DOUBLE,	/* sent MPI_Datatype */
		 local_y,	/* recvbuf */
		 local_n,	/* recvcount (usually equal to sendcount) */
		 MPI_DOUBLE,	/* received MPI_Datatype */
		 0,		/* root */
		 MPI_COMM_WORLD /* communicator */
		 );

    /* All nodes compute the local result */
    sum( local_x, local_y, local_z, local_n );

    /* Gather results from all nodes */
    MPI_Gather( local_z,	/* sendbuf */
		local_n,	/* sendcount (how many elements each node sends */
		MPI_DOUBLE,	/* sendtype */
		z,		/* recvbuf */
		local_n,	/* recvcount (how many elements should be received from _each_ node */
		MPI_DOUBLE,	/* recvtype */
		0,		/* root (where to send) */
		MPI_COMM_WORLD	/* communicator */
		);

    /* Uncomment the following block if you want process 0 to print
       the result */

#if 0
    if ( 0 == my_rank ) {
	for ( i=0; i<n; i++ ) {
	    printf("%f ", z[i]);
	}
	printf("\n");
    }	
#endif

    MPI_Finalize();

    return 0;
}
