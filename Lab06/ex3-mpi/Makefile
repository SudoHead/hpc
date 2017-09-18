EXE=mpi-circles mpi-rule30 circles-gen mpi-send-col

ALL: $(EXE)

.PHONY: clean

CFLAGS+=-std=c99 -Wall -Wpedantic

mpi-circles mpi-rule30 mpi-send-col: CC=mpicc

clean:
	\rm -f *~ $(EXE)
