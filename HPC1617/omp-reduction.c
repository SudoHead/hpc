/****************************************************************************
 *
 * omp-reduction - Demo of reduction operators with OpenMP
 *
 * Written in 2016, 2017 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * To the extent possible under law, the author(s) have dedicated all 
 * copyright and related and neighboring rights to this software to the 
 * public domain worldwide. This software is distributed without any warranty.
 *
 * You should have received a copy of the CC0 Public Domain Dedication
 * along with this software. If not, see 
 * <http://creativecommons.org/publicdomain/zero/1.0/>. 
 *
 * ----------------------------------------------------------------------------
 *
 * Compile with:
 * gcc -fopenmp omp-reduction.c -o omp-reduction
 *
 * Run with:
 * OMP_NUM_THREADS=1 ./omp-reduction
 * OMP_NUM_THREADS=2 ./omp-reduction
 * OMP_NUM_THREADS=4 ./omp-reduction
 *
 ****************************************************************************/

#include <stdio.h>

int main( void )
{
    int a = 2;
#pragma omp parallel reduction(*:a)
    {
	a += 2;
    }
    printf("%d\n",a);
    return 0;
}
