/****************************************************************************
 *
 * omp-merge-sort.c - Merge Sort with OpenMP
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
 * This program generates a random permutation of the first n integers
 * 0, 1, ... n-1 and sorts it using the recursive serial
 * implementation of Merge-Sort. This implementation uses selection
 * sort to sort small subvectors, in order to limit the cost of
 * recursion. The goal of this exercise is to parallelize this program
 * using OpenMP tasks.
 *
 * Compile with:
 *
 * gcc -fopenmp -std=c99 -Wall -Wpedantic omp-mergesort.c -o omp-mergesort
 *
 * Run with:
 *
 * ./omp-mergesort 50000
 *
 ****************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void swap(int* a, int* b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

/**
 * Sort v[low..high] using selection sort
 */
void selectionsort(int* v, int low, int high)
{
    int i, j;
    for (i=low; i<high; i++) {
        for (j=i+1; j<=high; j++) {
            if (v[i] > v[j]) {
                swap(&v[i], &v[j]);
            }
        }
    }
}

/**
 * Merge src[low..mid] with src[mid+1..high], put the result in
 * dst[low..high]
 */
void merge(int* src, int low, int mid, int high, int* dst)
{
    int i=low, j=mid+1, k=low;
    while (i<=mid && j<=high) {
        if (src[i] <= src[j]) {
            dst[k] = src[i++];
        } else {
            dst[k] = src[j++];
        }
        k++;
    }
    /* Handle remaining elements */
    while (i<=mid) {
        dst[k] = src[i++];
        k++;
    }
    while (j<=high) {
        dst[k] = src[j++];
        k++;
    }
}

/**
 * Sort v[i..j] using the recursive version of MergeSort; the array
 * tmp[i..j] is used as a temporary buffer (the caller is responsible
 * for providing a suitably sized array tmp). This implementation uses
 * selection sort for subvectors whose size is less than the cutoff,
 * in order to limit the overhead of recursion.
 */
void mergesort_rec(int* v, int i, int j, int* tmp)
{
    const int cutoff = 16;
    /* If the portion to be sorted is smaller than the cutoff, use
       selectoin sort. This is a well known optimization that avoids
       the overhead of recursion for small vectors. */
    if ( j - i + 1 < cutoff ) 
        selectionsort(v, i, j);
    else {
        const int m = (i+j)/2;
        
        #pragma omp task 
        mergesort_rec(v, i, m, tmp);
        #pragma omp task
        mergesort_rec(v, m+1, j, tmp);

        #pragma omp taskwait
        merge(v, i, m, j, tmp);
        /* copy the sorted data back to array v */
        memcpy(v+i, tmp+i, (j-i+1)*sizeof(v[0]));
    }
}

/**
 * Sort n-element array v[] using mergesort; after allocating a
 * temporary array with the same size of a (used for merging), this
 * function just calls mergesort_rec with the appropriate parameters.
 * After mergesort_rec terminates, the temporary array is deallocated.
 */
void mergesort(int *v, int n)
{
    int* tmp = (int*)malloc(n*sizeof(v[0]));
    #pragma omp parallel 
    #pragma omp master
    mergesort_rec(v, 0, n-1, tmp);
    free(tmp);
}

/* Returns a random integer in the range [a..b], inclusive */
int randab(int a, int b)
{
    return a + rand() % (b-a+1);
}

/**
 * Fills a[] with a random permutation of the intergers 0..n-1; the
 * caller is responsible for allocating a
 */
void fill(int* a, int n)
{
    int i;
    for (i=0; i<n; i++) {
        a[i] = (int)i;
    }
    for (i=0; i<n-1; i++) {
        int j = randab(i, n-1);
        swap(a+i, a+j);
    }
}

/* Return 1 iff a[] contains the values 0, 1, ... n-1, in that order */
int check(int* a, int n)
{
    int i;
    for (i=0; i<n; i++) {
        if ( a[i] != i ) {
            printf("Expected a[%d]=%d, got %d\n", i, i, a[i]);
            return 0;
        }
    }
    return 1;
}

int main( int argc, char* argv[] )
{
    int n = 100000;
    int *a;
    double tstart, tend;
    
    if ( argc == 2 ) {
        n = atoi(argv[1]);
    }

    a = (int*)malloc(n*sizeof(a[0]));

    fill(a, n);
    tstart = omp_get_wtime();
    mergesort(a, n);
    tend = omp_get_wtime();
    printf("Time to sort %d elements: %f\n", n, tend - tstart);
    printf("Check %s\n", (check(a, n) ? "OK" : "failed"));
    return 0;
}
