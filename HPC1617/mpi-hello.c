/****************************************************************************
 *
 * mpi-hello.c - Hello, world in MPI
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
 * --------------------------------------------------------------------------
 *
 * Compile with:
 * mpicc -Wall mpi-hello.c -o mpi-hello
 *
 * Run with:
 * mpirun -n 4 ./mpi-hello
 *
 ****************************************************************************/

#include "mpi.h"
#include <stdio.h>

int main( int argc, char *argv[] )
{
    int rank, size, len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    MPI_Init( &argc, &argv );	/* no MPI calls before this line */
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Get_processor_name(hostname, &len);
    printf("Greetings from process %d of %d running on %s\n", rank, size, hostname);
    MPI_Finalize();		/* no MPI calls after this line */
    return 0;
}
