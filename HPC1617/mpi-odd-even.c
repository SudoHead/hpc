/****************************************************************************
 *
 * mpi-odd-even.c - Odd-even transposition sort in MPI
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
 * mpicc -std=c99 -Wall -Wpedantic mpi-odd-even.c -o mpi-odd-even
 *
 * Run with:
 * mpirun -n 4 ./mpi-odd-even 
 *
 * mpi-odd-even takes the vector length n as an optional parameter. n
 * must be a multiple of the communicator size (number of processes
 * executed). If n is omitted, the default vector length is
 * 1000*comm_size. The vector is initialized with random values.
 *
 ****************************************************************************/

#include "hpc.h"
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/*
 * Comparison function used by qsort. Should return -1, 0 or 1
 * depending whetner *x is less than, equal to, or greater than *y
 */
int compare( const void* x, const void* y )
{
    const double *vx = (const double*)x;
    const double *vy = (const double*)y;

    if ( *vx > *vy ) {
	return 1;
    } else {
	if ( *vx < *vy )
            return -1;
    	else
            return 0;
    }
#if 0
    /* This is a "branchless" version of the comparison below. 
       It is much harder to understand, and should be
       used with care. */
    return (*vx > *vy) - (*vx < *vy);
#endif
}

/*
 * Fill vector v with n random values drawn from the interval [0,1)
 */
void fill( double* v, int n )
{
    int i;
    for ( i=0; i<n; i++ ) {
	v[i] = rand() / (double)RAND_MAX;
    }
}

/* 
 * Check whether the n vector array v is sorted according to the
 * comparison function compare()
 */
void check( const double* v, int n )
{
    int i;
    for (i=1; i<n; i++) {
	if ( compare( &v[i-1], &v[i] ) > 0 ) {
	    printf("Check failed at element %d (v[%d]=%f, v[%d]=%f)\n",
		   i-1, i-1, v[i-1], i, v[i] );
	    MPI_Abort(MPI_COMM_WORLD, -1);
	}
    }
    printf("Check OK\n");
}

/* 
 * Merge two sorted vectors local_x and received_x, with local_n
 * elements each, keeping the lower half of the result in local_x and
 * discarding the rest.  buffer is used as temporary space, for which
 * the caller is responsible for allocating space for local_n elements
 * of type double.
 */
void merge_low( double* local_x, double *received_x, double* buffer, int local_n )
{
    int i_l=0, i_r=0, i_b=0;
    while ( i_b < local_n ) {
	if ( compare( local_x + i_l, received_x + i_r ) < 0 ) {
	    buffer[i_b] = local_x[i_l];
	    i_l++;
	} else {
	    buffer[i_b] = received_x[i_r];
	    i_r++;
	}
	i_b++;
    }

    memcpy( local_x, buffer, local_n*sizeof(double) );
}

/* 
 * Merge two sorted vectors local_x and received_x, with local_n
 * elements each, keeping the upper half of the result in local_x and
 * discarding the rest.  buffer is used as temporary space, for which
 * the caller is responsible for allocating space for local_n elements
 * of type double.
 */
void merge_high( double* local_x, double *received_x, double* buffer, int local_n )
{
    int i_l=local_n-1, i_r=local_n-1, i_b=local_n-1;
    while ( i_b >= 0 ) {
	if ( compare( local_x + i_l, received_x + i_r ) > 0 ) {
	    buffer[i_b] = local_x[i_l];
	    i_l--;
	} else {
	    buffer[i_b] = received_x[i_r];
	    i_r--;
	}
	i_b--;
    }

    memcpy( local_x, buffer, local_n*sizeof(double) );
}

/*
 * Performs a single exchange-compare step. Two adjacent nodes
 * exchange their data, merging with the local buffer. The neighbor on
 * the left of the current node keeps the lower half of the merged
 * vector, while the neighbor on the right keeps the upper half.
 */
void do_sort_exchange( int phase, double *local_x, double* received_x, double *buffer, int local_n, int my_rank, int even_partner, int odd_partner )
{
    MPI_Status status;

    if ( phase % 2 == 0 ) {
	/* even phase */
	if ( even_partner != MPI_PROC_NULL ) {
	    MPI_Sendrecv(local_x,	/* sendbuf */
			 local_n,	/* sendcount */
			 MPI_DOUBLE,	/* datatype */
			 even_partner,	/* dest */
			 0,		/* sendtag */
			 received_x,	/* recvbuf */
			 local_n,	/* recvcount */
			 MPI_DOUBLE,	/* datatype */
			 even_partner,	/* source */
			 0,		/* recvtag */
			 MPI_COMM_WORLD, /* comm */
			 &status	/* status */
			 );
	    if ( my_rank < even_partner ) {
		merge_low( local_x, received_x, buffer, local_n );
            } else {
		merge_high( local_x, received_x, buffer, local_n );
            }
	}
    } else {
	/* odd phase */
	if ( odd_partner != MPI_PROC_NULL ) {
	    MPI_Sendrecv(local_x, 
			 local_n, 
			 MPI_DOUBLE, 
			 odd_partner, 
			 0, 
			 received_x, 
			 local_n, 
			 MPI_DOUBLE, 
			 odd_partner, 
			 0, 
			 MPI_COMM_WORLD,
			 &status);
	    if ( my_rank < odd_partner ) {
		merge_low( local_x, received_x, buffer, local_n );	    
            } else {
		merge_high( local_x, received_x, buffer, local_n );	    
            }
	}
    }    
}

int main( int argc, char* argv[] )
{
    double *x = NULL, *local_x, *received_x, *buffer;
    double tstart, tstop;
    int n, local_n, 
	phase,		/* compare-exchange phase */
	odd_partner,	/* neighbor to use during odd phase */
	even_partner	/* neighbor to use during even phase */
	;
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
	printf("Error: the vector length must be multiple of %d\n", comm_sz);
	MPI_Abort(MPI_COMM_WORLD, -1);
    }

    local_n = n / comm_sz;

    if ( 0 == my_rank ) {
	printf("Vector size: %d\n", n );
	printf("Number of MPI processes: %d\n", comm_sz );

	/* The master initializes the vector to be sorted */
	x = (double*)malloc( n * sizeof(*x) );
	tstart = hpc_gettime();
	fill(x, n);
	tstop = hpc_gettime();
	printf("Init: %f\n", tstop - tstart );
    }

    /* All nodes initialize their local vectors */
    local_x = (double*)malloc( local_n * sizeof(*local_x) );
    received_x = (double*)malloc( local_n * sizeof(*received_x) );
    buffer = (double*)malloc( local_n * sizeof(*buffer) );

    /* Find partners */
    if (my_rank % 2 == 0) {
	even_partner = (my_rank < comm_sz-1 ? my_rank + 1 : MPI_PROC_NULL );
	odd_partner = (my_rank > 0 ? my_rank - 1 : MPI_PROC_NULL );
    } else {
	even_partner = (my_rank > 0 ? my_rank - 1 : MPI_PROC_NULL );
	odd_partner = (my_rank < comm_sz-1 ? my_rank + 1 : MPI_PROC_NULL );
    }

    /* The root starts the timer */
    if ( 0 == my_rank ) {
	tstart = hpc_gettime();
    }

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

    /* sort local buffer */
    qsort( local_x, local_n, sizeof(*local_x), compare );

    /* phases of odd-even sort */
    for ( phase = 0; phase < comm_sz; phase++ ) {
	do_sort_exchange( phase, local_x, received_x, buffer, local_n, my_rank, even_partner, odd_partner);
    }

    /* Gather results from all nodes */
    MPI_Gather( local_x,	/* sendbuf */
		local_n,	/* sendcount (how many elements each node sends */
		MPI_DOUBLE,	/* sendtype */
		x,		/* recvbuf */
		local_n,	/* recvcount (how many elements should be received from _each_ node */
		MPI_DOUBLE,	/* recvtype */
		0,		/* root (where to send) */
		MPI_COMM_WORLD	/* communicator */
		);

    /* The root checks the sorted vector */
    if ( 0 == my_rank ) {
	tstop = hpc_gettime();
	printf("Sort: %f\n", tstop - tstart );
	tstart = hpc_gettime();
	check(x, n);
	tstop = hpc_gettime();
	printf("Check: %f\n", tstop - tstart );
    }	

    MPI_Finalize();

    return 0;
}
