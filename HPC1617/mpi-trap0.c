/*****************************************************************************
 * mpi-trap0.c - MPI implementation of the trapezoid rule; reworked
 * version of the code from http://www.cs.usfca.edu/~peter/ipp/.
 *
 * This solution uses the naive approach: node 0 (the master) collects
 * all partial results, and computes the final value without using the
 * reduction primitive.
 *
 * last updated in 2017 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
 * mpicc -std=c99 -Wall -Wpedantic mpi-trap0.c -o mpi-trap0
 *
 * Run with:
 * mpirun -n 4 ./mpi-trap0 
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

/*
 * Function to be integrated
 */
double f( double x )
{
    return 4.0/(1.0 + x*x);
}

/*
 * Compute the area of function f(x) for x=[a, b] using the trapezoid
 * rule. The integration interval [a,b] is partitioned into n
 * subintervals of equal size.
 */
double trap( int my_rank, int comm_sz, double a, double b, int n )
{
    const double h = (b-a)/n;
    const int local_n_start = n * my_rank / comm_sz;
    const int local_n_end = n * (my_rank + 1) / comm_sz;
    double x = a + local_n_start * h;
    double my_result = 0.0;
    int i;

    for (i = local_n_start; i < local_n_end; i++) {
	my_result += h*(f(x) + f(x+h))/2.0;
	x += h;
    }
    return my_result;
}

int main( int argc, char* argv[] )
{
    double a, b, partial_result, result = 0.0;
    int n, i;
    int my_rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    /* All nodes receive the command line parameters, therefore the
       master does not need to send them. */
    if ( 4 == argc ) {
	a = atof(argv[1]);
	b = atof(argv[2]);
	n = atoi(argv[3]);
    } else {
	a = 0.0;
	b = 1.0;
	n = 1000000;
    }

    /* All nodes (incl. the master) compute their local result */
    partial_result = trap( my_rank, comm_sz, a, b, n );

    if ( 0 != my_rank ) {
	/* all participants (other than the master) send their local
	   result to the master */
	MPI_Send( &partial_result,	/* Buffer */
		  1,			/* Size */
		  MPI_DOUBLE,		/* Type */
		  0			/* dest */, 
		  0			/* tag */, 
		  MPI_COMM_WORLD	/* Communicator */
		  );
    } else {
	/* The master collects all partial results from other nodes */
	result = partial_result;
	for ( i=1; i<comm_sz; i++ ) {
	    MPI_Recv( &partial_result,	/* Buffer */
		      1,		/* Size */
		      MPI_DOUBLE,	/* Type */
		      i,		/* Sender */
		      0,		/* Tag */
		      MPI_COMM_WORLD,	/* Communicator */
		      MPI_STATUS_IGNORE /* Status */
		      );
	    result += partial_result;
	}
	printf("Area: %f\n", result);
    }

    MPI_Finalize();

    return 0;
}
