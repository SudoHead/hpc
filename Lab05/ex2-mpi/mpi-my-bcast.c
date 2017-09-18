/****************************************************************************
 *
 * mpi-my-bcast.c - Simulate MPI_Bcast using point-to-point communications
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
 * Compile with
 * mpicc -std=c99 -Wall -Wpedantic mpi-my-bcast.c -o mpi-my-bcast
 *
 * run with:
 * mpirun -n 7 ./mpi-my-bcast
 *
 ****************************************************************************/

#include <mpi.h>
#include <stdio.h>

/**
 * This is just a stub; you should rewrite this function using
 * MPI_Send() and MPI_Recv(); of course this function can also use
 * MPI_Comm_size() and MPI_Comm_rank() to get the ID of the current
 * process and the communicator size.
 */
void my_Bcast(int *v)
{
    MPI_Bcast( v,               /* buffer       */
               1,               /* count        */
               MPI_INT,         /* datatype     */
               0,               /* root         */
               MPI_COMM_WORLD   /* comm         */
               );
}

int main( int argc, char *argv[] )
{
    int my_rank;
    int v;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if ( 0 == my_rank ) {
        v = 999; /* only process 0 sets the value to be sent */
    } else {
        v = -1;
    }

    printf("Proess %d has %d\n", my_rank, v);

    my_Bcast(&v);

    /* The barrier ensures that all processes have written "Process X
       has Y" before they start to write "Process X got Y" */
    MPI_Barrier(MPI_COMM_WORLD);

    printf("Process %d got %d\n", my_rank, v);

    MPI_Finalize();

    return 0;
}
