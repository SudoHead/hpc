/****************************************************************************
 *
 * cuda-reverse.cu - Array reversal with CUDA
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
 * ---------------------------------------------------------------------------
 *
 * Compile with:
 * nvcc cuda-reverse.cu -o cuda-reverse
 *
 * Run with:
 * ./cuda-reverse [len]
 *
 * Example:
 * ./cuda-reverse
 *
 ****************************************************************************/
#include <stdio.h>
#include <math.h>

/* Reverse in[] into out[] */
void reverse( int *in, int *out, int n )
{
    int i;
    for (i=0; i<n; i++) {
        out[n - 1 - i] = in[i];
    }
}

/* In-place reversal of x[] into itself */
void inplace_reverse( int *x, int n )
{
    int i = 0, j = n-1;
    while (i < j) {
        int tmp = x[j];
        x[j] = x[i];
        x[i] = tmp;
        j--;
        i++;
    }
}

void fill( int *x, int n )
{
    int i;
    for (i=0; i<n; i++) {
        x[i] = i;
    }
}

int check( int *x, int n )
{
    int i;
    for (i=0; i<n; i++) {
        if (x[i] != n - 1 - i) {
            printf("FAILED: x[%d]=%d, expected %d\n", i, x[i], n-1-i);
            return -1;
        }
    }
    printf("OK\n");
    return 0;
}

int main( int argc, char* argv[] )
{
    int *in, *out;
    int n;
    const int default_len = 1024*1024;

    if ( argc > 2 ) {
        printf("\nUsage: %s [len]\n\nReverse an array of \"len\" elements (default %d)\n\n", argv[0], default_len);
        return -1;
    }

    if ( argc > 1 ) {
        n = atoi(argv[1]);
    } else {
        n = default_len;
    }

    const size_t size = n * sizeof(*in);

    /* Allocate and initialize in[] and out[] */
    in = (int*)malloc(size); 
    fill(in, n);
    out = (int*)malloc(size);
    
    /* Reverse */
    printf("Reverse of %d elements... ", n);
    reverse(in, out, n);    
    check(out, n);

    /* In-place reverse */
    printf("In-place reverse of %d elements... ", n);
    inplace_reverse(in, n);    
    check(in, n);

    /* Cleanup */
    free(in); 
    free(out);
    return 0;
}
