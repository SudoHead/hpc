/****************************************************************************
 *
 * mpi-get-count.c - Shows how the MPI_Get_Count function can beused
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
 * mpicc mpi-get-count.c -o mpi-get-count
 *
 * Run with:
 * mpirun -n 2 mpi-get-count
 *
 * Process 0 sends a random number of integers to process 1
 *
 ****************************************************************************/

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BUFLEN 16

int main( int argc, char *argv[])
{
    int rank, buf[BUFLEN] = {0};
    int count, i;
    MPI_Status status;
    MPI_Init(&argc, &argv);   
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    
    /* Process 0 sends, Process 1 receives */
    if (rank == 0) {
        /* Initialize the random number generator (otherwise you
           always get the same number of items, since the RNG is
           deterministic) */
        srand(time(NULL));
        /* Fills the buffer with a random number of integers */
        count = 1 + rand()%BUFLEN;
        for (i=0; i<count; i++) {
            buf[i] = i;
        }
        MPI_Send(&buf, count, MPI_INT, 1, 0, MPI_COMM_WORLD);
        printf("Sent %d integers\n", count);
    }
    else if (rank == 1) {
        MPI_Recv(&buf, BUFLEN, MPI_INT, 0, 0, MPI_COMM_WORLD, &status );
        MPI_Get_count(&status, MPI_INT, &count);
        printf( "Received %d integers: ", count );
        for (i=0; i<count; i++) {
            printf("%d ", buf[i]);
        }
        printf("\n");
    }
    
    MPI_Finalize();
    return 0;
}
