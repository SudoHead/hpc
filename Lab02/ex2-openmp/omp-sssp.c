/******************************************************************************
 *
 * omp-sssp.c - Skeleton for computing the single-source
 * shortest path using the Bellman-Ford algorithm
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
 * Skeleton of a program that computes shortest paths on directed
 * graphs using Bellman-Ford algorithm. Given a directed, weighted
 * graph with nonnegative edge weights with n nodes and m edges,
 * Bellman-Ford's algorithm can compute all node distances from a
 * given source node in time O(nm). It can also detect the presence of
 * cycles of negative weights (although no check is done in this
 * program).
 *
 * Compile with:
 * gcc -fopenmp -std=c99 -Wall -Wpedantic omp-sssp.c -o omp-sssp
 *
 * To compute the distances using the graph rome99.gr, using node 0 as
 * the source:
 * ./omp-sssp rome99.gr 0
 *
 * or simply:
 * ./omp-sssp rome99.gr
 *
 ******************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <omp.h>

typedef struct {
    int src, dst;
    double w; 
} edge_t;

typedef struct {
    int n; /* number of nodes */
    int m; /* number of edges */
    edge_t *edges; /* array of m edges */
} graph_t;

/* Load a graph description in DIMACS format from file |fname|; store
   the graph in |g|. For more information on the DIMACS challenge format
   see http://www.diag.uniroma1.it/challenge9/index.shtml */
void load_dimacs(const char* fname, graph_t* g)
{
    FILE* f = fopen(fname, "r");
    const size_t buflen = 1024;
    char buf[buflen], prob[buflen];
    int n, m, src, dst, w;
    int cnt = 0; /* edge counter */
    int nmatch;

    if ( !f ) {
        printf("Cannot open %s for reading\n", fname);
        exit(-1);
    }
    
    while ( fgets(buf, buflen, f) ) {
        switch( buf[0] ) {
        case 'c': 
            break; /* ignore comment lines */
        case 'p':
            /* Parse problem format; assume it is of type "shortest path" */
            sscanf(buf, "%*c %s %d %d", prob, &n, &m);
            if (strcmp(prob, "sp")) {
                printf("Unknown DIMACS problem type %s\n", prob);
                exit(-1);
            }
            printf("DIMACS %s with %d nodes and %d edges\n", prob, n, m);
            g->n = n;
            g->m = m;
            g->edges = malloc(m*sizeof(g->edges[0]));
            cnt = 0;
            break;
        case 'a':
            nmatch = sscanf(buf, "%*c %d %d %d", &src, &dst, &w);
            if (nmatch != 3) {
                printf("malformed line:\n%s\n", buf);
                exit(-1);
            }
            /* the DIMACS graph format labels nodes starting from 1;
               we decrement the ids so that we start from 0 */
            src--;
            dst--;
            g->edges[cnt].src = src;
            g->edges[cnt].dst = dst;
            g->edges[cnt].w = w;
            cnt++;
            break;
        default:
            printf("Unrecognized character %c in input file\n", buf[0]);
            exit(-1);
        }        
    }
    assert( cnt == g->m );
    fclose(f);
}


/* Compute the minimum distances d[] of each node in g from the source
   node s.  s must be an integer in 0..((g->n)-1); the array d[] must
   have g->n elements, and must be pre-allocated by the caller. At the
   end of this function, d[i] is the (minimum) distance of node i from
   node s, for each i = 0, ... (g->n)-1 */
void bellmanford(const graph_t* g, int s, double *d)
{
    const int n = g->n;
    const int m = g->m;
    double *dnew = malloc(n*sizeof(dnew[0])); assert(dnew);

    /* Implementare il resto di questa funzione. Nota: per
       rappresentare il valore "+infinito" è possibile usare il
       simbolo INFINITY (definito in math.h) che rappresenta un
       particolare valore in virgola mobile con le caratteristiche di
       +infinito, tra cui quella di essere maggiore di qualsiasi
       valore finito in virgola mobile. */

    free(dnew);
}

int main( int argc, char* argv[] )
{
    graph_t g;
    int src = 0;
    double *d;
    double tstart, tend;

    if ( argc < 2 || argc > 3 ) {
        fprintf(stderr, "Usage: %s [problem file] [source node]\n", argv[0]);
        return -1;
    }
    
    load_dimacs(argv[1], &g);
    d = (double*)malloc((g.m) * sizeof(d[0]));
    
    if ( argc > 2 ) {
        src = atoi(argv[2]);
        if (src < 0 || src >= g.n) {
            fprintf(stderr, "Invalid source node (should be within the range %d-%d)\n", 0, g.n-1);
            exit(-1);
        }
    }

    tstart = omp_get_wtime();
    bellmanford(&g, src, d);
    tend = omp_get_wtime();

    printf("d[%d]=%f\n", (g.n)-1, d[(g.n)-1]);
    
    printf("Execution time %f\n", tend - tstart);
    return 0;
}
