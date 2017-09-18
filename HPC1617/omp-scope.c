/****************************************************************************
 *
 * omp-scoipe.c - Demonstration of the OpenMP "scope" clause.
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
 * --------------------------------------------------------------------------
 *
 * Compile with:
 * gcc -std=c99 -Wall -pedantic -fopenmp omp-scope.c -o omp-scope
 *
 * Run with:
 * ./omp-scope
 *
 ****************************************************************************/
#include <stdio.h>

int main( void )
{
    int a=1, b=1, c=1, d=1;	
#pragma omp parallel num_threads(10) \
    private(a) shared(b) firstprivate(c)
    {	
	printf("Hello World!\n");
	a++;	
	b++;	
	c++;	
	d++;	
    }	
    printf("a=%d\n", a);
    printf("b=%d\n", b);
    printf("c=%d\n", c);
    printf("d=%d\n", d);
    return 0;
}
