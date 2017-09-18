/******************************************************************************
 *
 * levenshtein.c - Compute the Levenshtein edit distance between two strings.
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
 * This program computes the Levenshtein edit distance between twi
 * strings passed on the command line. Note that this program requires
 * a C99 compliant compiler, since it relies on C99-specific features.
 * For gcc this means adding -std=c99 to the command line.
 *
 * Compile with:
 *
 * gcc -std=c99 -Wall -pedantic -fopenmp -O2 levenshtein.c -o levenshtein
 *
 * Run with:
 *
 * ./levenshtein "una mela al giorno leva il medico di torno" "quarantaquattro gatti in fila per 6 col resto di 2"
 *
 ******************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifndef min
inline int min( int a, int b ) 
{
    return ( a < b ? a : b );
}
#endif

/* This function computes the Levenshtein edit distance between
   strings s and t. If we let n = strlen(s) and m = strlen(t), this
   function uses time O(nm) and space O(nm). */
int levenshtein(const char* s, const char* t)
{
    const int n = strlen(s), m = strlen(t);
    int i, j;
    int (*L)[m+1] = malloc((n+1)*(m+1)*sizeof(int)); /* C99 idiom: L is of type int L[n+1][m+1] */
    int result;

    /* degenerate cases first */
    if (n == 0) return m;
    if (m == 0) return n;

    /* Initialize the first column of L */
    for (i = 0; i <= n; i++)
        L[i][0] = i;

    /* Initialize the first row of L */
    for (j = 0; j <= m; j++) 
        L[0][j] = j;

    /* Fills the rest fo the matrix. */
    /* THIS IS THE LOOP YOU NEED TO PARALLELIZE */
    for (i = 1; i <= n; i++) {
        for (j = 1; j <= m; j++) {
            L[i][j] = min(min(L[i-1][j] + 1, L[i][j-1] + 1), L[i-1][j-1] + (s[i-1] != t[j-1]));
        }
    }
    result = L[n][m];
    free(L);
    return result;
}

int main( int argc, char* argv[] )
{
    if ( argc != 3 ) {
	fprintf(stderr, "Usage: %s str1 str2\n", argv[0]);
	return -1;
    }
    
    printf("%d\n", levenshtein(argv[1], argv[2]));
    return 0;
}
