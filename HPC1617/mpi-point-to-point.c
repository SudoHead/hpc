/****************************************************************************
 *
 * mpi-point-to-point.c - Simple point-to-point communication demo for MPI
 *
 * Written in 2016 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
 * mpicc mpi-point-to-point.c -o mpi-point-to-point
 *
 * Run with:
 * mpirun -n 2 mpi-point-to-point
 *
 * Process 0 sends an integer value to process 1
 *
 ****************************************************************************/

#include <mpi.h>
#include <stdio.h>

int main( int argc, char *argv[])
{
    int rank, buf;
    MPI_Status status;
    MPI_Init(&argc, &argv);   
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    
    /* Process 0 sends, Process 1 receives */
    if (rank == 0) {
        buf = 123456;
        MPI_Send( &buf,         /* send buffer  */
                  1,            /* count        */
                  MPI_INT,      /* datatype     */
                  1,            /* destination  */
                  0,            /* tag          */
                  MPI_COMM_WORLD /* communicator */
                  );
    }
    else if (rank == 1) {
        MPI_Recv( &buf,         /* receive buffer */
                  1,            /* count        */
                  MPI_INT,      /* datatype     */
                  0,            /* source       */
                  0,            /* tag          */
                  MPI_COMM_WORLD, /* communicator */
                  &status       /* status       */
                  );
        printf( "Received %d\n", buf );
    }
    
    MPI_Finalize();
    return 0;
}
