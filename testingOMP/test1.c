#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int args, char **argv) {

  int n = 100000;
  long sum = 0;

  double t_start, t_end;

  if(args > 1) {
    n = atoi(argv[1]);
  }

  long m = 900000000;

  t_start = omp_get_wtime();
  #pragma omp parallel for reduction(+:sum)
  for(long i = 0; i < m; i++) {
      sum += i;
  }
  t_end = omp_get_wtime();

  printf("sum = %ld\n",sum);
  printf("excecution tim e = %f\n", t_end - t_start);
  printf("max threads = %d\n", omp_get_max_threads() );
}
