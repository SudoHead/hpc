#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char * argv[]) {

	int nSpin = 1;

	int buf = 0;
	int rank, comm_sz = 1;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

	if(argc > 1) {
		nSpin = atoi(argv[1]);
	}

	int spin = 0;

	while(spin < nSpin) {
		if(rank == 0) { //MASTER
			if(spin > 0) { //recv from p-1
				MPI_Recv(&buf, 1, MPI_INT, comm_sz-1, 0, MPI_COMM_WORLD, &status);
			}
			buf++;
			MPI_Send(&buf, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
			printf("Result v = %d after %d spins with p = %d\n", buf, spin, comm_sz);
		} else {
			//printf("recv from %d\n", rank -1);
			MPI_Recv(&buf, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, &status);
			buf++;
			//printf("sending to %d\n",(rank+1) % comm_sz);
			MPI_Send(&buf, 1, MPI_INT, (rank+1) % comm_sz, 0, MPI_COMM_WORLD);
		}
		spin++;
	}

	MPI_Finalize();

	return 0;
}