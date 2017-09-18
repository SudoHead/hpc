/*****************************************************************************
 * mpi-scan.c - MPI_Scan demo
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
 * mpicc -std=c99 -Wall -Wpedantic mpi-scan.c -o mpi-scan
 *
 * Run with:
 * mpirun -n 4 ./mpi-scan
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int my_rank, comm_sz;
    int *local_x, *scan_x = NULL;
    int local_N = 3, i;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    local_x = (int *) malloc(local_N * sizeof(*local_x));
    for (i = 0; i < local_N; i++) {
	local_x[i] = i + my_rank * local_N;
    }

    scan_x = (int *) malloc(local_N * sizeof(*scan_x));

    MPI_Scan(local_x,		/* sendbuf      */
	     scan_x,		/* recvbuf      */
	     local_N,		/* count        */
	     MPI_INT,		/* datatype     */
	     MPI_SUM,		/* operator     */
	     MPI_COMM_WORLD);

    for (i = 0; i < local_N; i++) {
	printf("rank=%d scan_x[%d]=%d\n", my_rank, i, scan_x[i]);
    }

    MPI_Finalize();
    return 0;
}
