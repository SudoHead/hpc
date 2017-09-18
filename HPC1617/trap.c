/****************************************************************************
 *
 * trap.c - Serial implementation of the trapezoid rule;
 * slightly simplified version of the code from
 * http://www.cs.usfca.edu/~peter/ipp/
 *
 * Written by Peter Pacheco
 * Modified in 2016, 2017 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
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
 * gcc -std=c99 -Wall -Wpedantic trap.c -o trap
 *
 * Run with:
 * ./trap -20 20 100
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>

/*
 * Function to be integrated
 */
double f( double x )
{
    return 4.0/(1.0 + x*x);
}

/*
 * Compute the area of function f(x) for x=[a, b] using the trapezoid
 * rule. The integration interval [a,b] is partitioned into n
 * subintervals of equal size.
 */
double trap( double a, double b, int n )
{
#if 1
    /*
     * This code is a direct implementation of the trapezoid rule.
     * The area of the trapezoid on interval [x, x+h] is computed as 
     * h*(f(x) + f(x+h))/2.0. All areas are accumulated in
     * variable |result|.
     */
    double result = 0.0;
    double h = (b-a)/n;
    double x = a;
    int i;
    for ( i = 0; i<n; i++ ) {
	result += h*(f(x) + f(x+h))/2.0;
	x += h;
    }
    return result;
#else
    /* This code is a slightly more efficient implementation of the
     * trapezoid rule, since it evaluates the function f() fewer
     * times. It is based on the observation that the summation
     *
     *   f(x_0) + f(x_1)     f(x_1) + f(x_2)           f(x_{n-1}) + f(x_n)
     * h*--------------- + h*--------------- + ... + h*-------------------
     *         2.0                 2.0                        2.0
     *
     * can be rewritten as:
     *
     *    / f(x_0) + f(x_n)                                     \
     * h*|  --------------- + f(x_1) + f(x_2) + ... + f(x_{n-1}) |
     *    \       2.0                                           /
     */  
    double approx = (f(a) + f(b))/2;
    double h = (b-a)/n;
    double x = a+h;
    int i;
    for ( i = 1; i<n; i++ ) {
	approx += f(x);
	x += h;
    }
    return h*approx;
#endif
}

int main( int argc, char* argv[] )
{
    double a, b, result;
    int n;
    if ( 4 == argc ) {
	a = atof(argv[1]);
	b = atof(argv[2]);
	n = atoi(argv[3]);
    } else {
	a = 0.0;
	b = 1.0;
	n = 1000000;
    }
    result = trap(a, b, n);
    printf("Area: %f\n", result);
    return 0;
}
