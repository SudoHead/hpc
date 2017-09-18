/****************************************************************************
 *
 * cuda-rule30.cu - Rule30 Cellular Automaton with CUDA
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
 * https://en.wikipedia.org/wiki/Rule_30 . This program uses the CPU
 * only.
 *
 * Compile with:
 * nvcc cuda-rule30.cu -o cuda-rule30
 *
 * Run with:
 * /cuda-rule30 1024 1024
 *
 ****************************************************************************/
#include <stdio.h>
#include <stdlib.h>

typedef unsigned char cell_t;

/**
 * Given the current state of the CA, compute the next state.
 */
void rule30( cell_t *cur, cell_t *next, int n )
{
    int i;
    for (i=0; i<n; i++) {
        const cell_t left   = cur[(i-1+n)%n];
        const cell_t center = cur[i        ];
        const cell_t right  = cur[(i+1)%n  ];
        next[i] =                               \
            ( left && !center && !right) ||
            (!left && !center &&  right) ||
            (!left &&  center && !right) ||
            (!left &&  center &&  right);
    }
}

/**
 * Initialize the domain; all cells are 0, with the exception of a
 * single cell in the middle of the domain.
 */
void init_domain( cell_t *cur, int n )
{
    int i;
    for (i=0; i<n; i++) {
        cur[i] = 0;
    }
    cur[n/2] = 1;
}

/**
 * Dump the current state of the CA to PBM file |out|.
 */
void dump_state( FILE *out, cell_t *cur, int n )
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
    cell_t *cur, *next, *tmp;
    int width = 1024, steps = 1024, s;    
    
    if ( argc > 3 ) {
        fprintf(stderr, "Usage: %s [width [steps]]\n", argv[0]);
        return -1;
    }

    if ( argc > 1 ) {
        width = atoi(argv[1]);
    }

    if ( argc > 2 ) {
        steps = atoi(argv[2]);
    }

    const size_t size = width * sizeof(cell_t);

    /* Allocate space for host copy the cur[] and next[] vectors */
    cur = (cell_t*)malloc(size);    
    next = (cell_t*)malloc(size);

    /* Create the output file */
    out = fopen(outname, "w");
    if ( !out ) {
        fprintf(stderr, "Cannot create %s\n", outname);
        return -1;
    }
    fprintf(out, "P1\n");
    fprintf(out, "# produced by %s %d %d\n", argv[0], width, steps);
    fprintf(out, "%d %d\n", width, steps);
    
    /* Initialize the domain */
    init_domain(cur, width);
    
    /* Evolve the CA */
    for (s=0; s<steps; s++) {
        
        /* Dump the current state to the output image */
        dump_state(out, cur, width);

        /* Compute next state */
        rule30(cur, next, width);

        /* swap cur and next on the GPU */
        tmp = cur;
        cur = next;
        next = tmp;
    }
    
    fclose(out);
    free(cur);
    free(next);
    return 0;
}
