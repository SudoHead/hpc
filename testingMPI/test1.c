#include <stdio.h>
#include <mpi.h>

int main(int args, char *argv[]) {

  int my_rank, size, len;
  char hostname[MPI_MAX_PROCESSOR_NAME];

  MPI_Init(&args, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Get_processor_name(hostname, &len);

  printf("hostname = %s[len = %d], my_rank = %d, comm_size = %d\n", hostname, len, my_rank, size);

  MPI_Finalize();

  return 0;
}
