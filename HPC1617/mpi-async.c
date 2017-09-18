/****************************************************************************
 *
 * mpi-async.c - Simple point-to-point communication for MPI using
 * asynchronous primitives
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
 * ---------------------------------------------------------------------------
 *
 * Compile with:
 * mpicc -Wall mpi-async.c -o mpi-async
 *
 * Run with:
 * mpirun -n 2 mpi-async
 *
 * Process 0 sends an integer value to process 1
 *
 ****************************************************************************/

#include "mpi.h"
#include <stdio.h>

void big_computation( void )
{
    printf("Some big computation...\n");
}

int main( int argc, char *argv[])
{
    int rank, size, buf;
    MPI_Status status;
    MPI_Request req;
    MPI_Init(&argc, &argv);   
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    
    if ( size < 2 ) {
        printf("Error: you must run at least 2 processes\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    
    if (rank == 0) {
        buf = 123456;
        /* asynchronous send */
        MPI_Isend( &buf, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &req);
        big_computation();
        MPI_Wait(&req, &status);
        printf("Master terminates\n");
    }
    else if (rank == 1) {
        /* synchronous receive */
        MPI_Recv( &buf, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status );
        printf( "Received %d\n", buf );
    }
    
    MPI_Finalize();
    return 0;
}
