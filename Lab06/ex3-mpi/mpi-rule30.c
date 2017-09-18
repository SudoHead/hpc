/****************************************************************************
 *
 * mpi-rule30.c - Rule 30 Cellular Automaton with MPI
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
 * This program implements the "rule 30 CA" as described in
 * https://en.wikipedia.org/wiki/Rule_30 . Although this program uses
 * MPI, it is essentially serial since the master
 * does everything.
 *
 * Compile with:
 * mpicc -std=c99 -Wall -Wpedantic mpi-rule30.c -o mpi-rule30
 *
 * Run with:
 * mpirun -n 4 ./mpi-rule30 1024 1024
 *
 ****************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/**
 * Given the current state of the CA, compute the next state. This
 * function uses periodic boundary conditions, so that, for example,
 * the neighbors of cur[0] are cur[n-1] and cur[1].
 */
void step( int *cur, int *next, int n )
{
    int i;
    for (i=1; i<n-1; i++) {
        const int east = cur[i-1];
        const int center = cur[i];
        const int west = cur[i+1];
        next[i] = ( (east && !center && !west) ||
                    (!east && !center && west) ||
                    (!east && center && !west) ||
                    (!east && center && west) );
    }
}

/**
 * Initialize the domain; all cells are 0, except the cell
 * in the middle.
 */
void init_domain( int *cur, int n )
{
    int i;
    for (i=0; i<n; i++) {
        cur[i] = 0;
    }
    cur[n/2] = 1;
}

/**
 * Dump the current state of the automaton to PBM file |out|.
 */
void dump_state( FILE *out, int *cur, int n )
{
    int i;
    for (i=0; i<n; i++) {
        fprintf(out, "%d ", cur[i]);
    }
    fprintf(out, "\n");
}

int main( int argc, char* argv[] )
{
    const char *outname = "rule30.pbm";
    FILE *out;
    int *cur, *next, *tmp;
    int width, steps = 1024, s;
    int my_rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if ( (0 == my_rank) && (argc > 3) ) {
        printf("Usage: %s [width [steps]]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    
    if ( argc > 1 ) {
        width = atoi(argv[1]);
    } else {
        width = comm_sz * 100;
    }

    if ( argc > 2 ) {
        steps = atoi(argv[2]);
    } 

    if ( (0 == my_rank) && (width % comm_sz) ) {
        printf("The image width (%d) must be a multiple of comm_sz (%d)\n", width, comm_sz);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    /* This is not a true parallel version: the master does everything */
    if ( 0 == my_rank ) {
        cur = (int*)malloc( width * sizeof(*cur));
        next = (int*)malloc( width * sizeof(*next));

        /* The master creates the output file */
        out = fopen(outname, "w");
        if ( !out ) {
            printf("Cannot create %s\n", outname);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        fprintf(out, "P1\n");
        fprintf(out, "# produced by %s %d %d\n", argv[0], width, steps);
        fprintf(out, "%d %d\n", width, steps);

        /* Initialize the domain */
        init_domain(cur, width);
    }

    int local_width = width / comm_sz;
    //cur + 2 ghost cells
    int *local_cur_ex = (int*)malloc((local_width+2) * sizeof(int));
    int *local_next_ex = (int*)malloc((local_width+2) * sizeof(int));


    MPI_Scatter(cur,
    	local_width,
    	MPI_INT,
		&local_cur_ex[1],
		local_width,
		MPI_INT,
		0,
		MPI_COMM_WORLD    	
    	);

    int ghost_cells[2]; //ghost cells

    int right_proc = (my_rank + 1 + comm_sz) % comm_sz;
    int left_proc = (my_rank - 1 + comm_sz) % comm_sz;
    MPI_Status status;

    for(s = 0; s < steps; s++) {
    	MPI_Sendrecv(
	    	&local_cur_ex[1],
	    	1,
	    	MPI_INT,
	    	left_proc,
	    	0,
	    	&ghost_cells[1],
	    	1,
	    	MPI_INT,
	    	right_proc,
	    	0,
	    	MPI_COMM_WORLD,
	    	&status
	    	);

    	printf("send 1\n");

	    MPI_Sendrecv(
	    	&local_cur_ex[local_width],
	    	1,
	    	MPI_INT,
	    	right_proc,
	    	0,
	    	&ghost_cells[0],
	    	1,
	    	MPI_INT,
	    	left_proc,
	    	0,
	    	MPI_COMM_WORLD,
	    	&status
	    	);

	    printf("send 2\n");

	    local_cur_ex[0] = ghost_cells[0];
	    local_cur_ex[local_width+1] = ghost_cells[1];

	    /* Compute the next state */
        step(local_cur_ex, local_next_ex, local_width+2);

           


	    MPI_Gather(
	    	&local_cur_ex[1],
	    	local_width,
	    	MPI_INT,
	    	cur,
	    	local_width,
	    	MPI_INT,
	    	0,
	    	MPI_COMM_WORLD
	    	);

	     /* swap cur and next */
            tmp = local_cur_ex;
            local_cur_ex = local_next_ex;
            local_next_ex = tmp;

	    if(my_rank == 0) {
	    	dump_state(out, cur, width);
	    }
	    printf("Hi\n");
    }


    if(my_rank == 0) {
    	free(out);
    	free(cur);
		free(next);
    }

    
	free(local_cur_ex);
	free(local_next_ex);



    MPI_Finalize();

    return 0;
}
