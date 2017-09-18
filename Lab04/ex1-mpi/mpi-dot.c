/****************************************************************************
 *
 * mpi-dot.c - Parallel dot product using MPI.
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
 * mpicc -Wall mpi-dot.c -o mpi-dot -lm
 *
 * Run with:
 * mpirun -n 4 ./mpi-dot
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

/*
 * Compute sum { x[i] * y[i] }, i=0, ... n-1
 */
double dot( double* x, double* y, int n )
{
    double s = 0.0;
    int i;
    for (i=0; i<n; i++) {
	s += x[i] * y[i];
    }
    return s;
}

int main( int argc, char* argv[] )
{
    double *x, *y, result = 0.0;
    int n = 1000;
    int my_rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    
    if ( 2 == argc ) {
	   n = atof(argv[1]);
    }    
    
    x = y = NULL;

    if ( 0 == my_rank ) {
    /* The master allocates the vectors */
        int i;
        x = (double*)malloc( n * sizeof(*x) );
        y = (double*)malloc( n * sizeof(*y) );
        for ( i=0; i<n; i++ ) {
            x[i] = i + 1.0;
            y[i] = 1.0 / x[i];
        }
    }
    
    double *local_x, *local_y, partial_result;
    int local_n = n / comm_sz;

    local_x = (double*)malloc( local_n * sizeof(double) );
    local_y = (double*)malloc( local_n * sizeof(double) );

    MPI_Scatter(x,
        local_n,
        MPI_DOUBLE,
        local_x,
        local_n,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD);

    MPI_Scatter(y,
        local_n,
        MPI_DOUBLE,
        local_y,
        local_n,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD);

    partial_result = dot(local_x, local_y, local_n);

    if(my_rank == 0 && n % comm_sz != 0) {
        for(int i = n - n%comm_sz; i < n; i++) {
            partial_result += x[i] * y[i];
        }
    }

    MPI_Reduce( &partial_result,
        &result,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        MPI_COMM_WORLD);

    if( my_rank == 0 ) {
        printf("Dot product: %f\n", result);
        if ( fabs(result - n) < 1e-5 ) {
            printf("Check OK\n");
        } else {
            printf("Check failed: got %f, expected %f\n", result, (double)n);
        }
    }

    MPI_Finalize();
    
    return 0;
}
