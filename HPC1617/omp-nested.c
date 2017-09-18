/****************************************************************************
 *
 * omp-nested.c - Nested parallelism with OpenMP
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
 * Compile with:
 * gcc -std=c99 -Wall -pedantic -fopenmp omp-nested.c -o omp-nested
 *
 * Run with:
 * OMP_NESTED=true ./omp-nested
 *
 ****************************************************************************/
#include <stdio.h>
#include <omp.h>

void greet(int level, int parent)
{
    printf("Level %d (parent=%d): greetings from thread %d of %d\n", 
           level, parent, omp_get_thread_num(), omp_get_num_threads());
}

int main( void )
{
    omp_set_num_threads(4);
#pragma omp parallel
    {
        greet(1, -1);
        int parent = omp_get_thread_num();
#pragma omp parallel
        {
            greet(2, parent);
            int parent = omp_get_thread_num();
#pragma omp parallel
            {
                greet(3, parent);
            }
        }
    }    
    return 0;
}
