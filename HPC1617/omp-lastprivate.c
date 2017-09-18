#include <stdio.h>

int main( void )
{
    int tmp = 0, i;
#pragma omp parallel for firstprivate(tmp) lastprivate(tmp)
    for (i = 0; i < 1000; i++) {
        tmp = i;
    }
    printf("%d\n", tmp);
    return 0;
}
