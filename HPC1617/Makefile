EXE_OMP:=$(basename $(wildcard omp-*))
EXE_MPI:=$(basename $(wildcard mpi-*))
EXE_SERIAL:=trap odd-even prefix-sum cache game-of-life reduction
EXE_SIMD:=$(basename $(wildcard simd-*))
EXE_CUDA:=$(basename $(wildcard cuda-*))
EXE:=$(EXE_OMP) $(EXE_MPI) $(EXE_SERIAL) $(EXE_SIMD) $(EXE_CUDA)
DISTFILES:=$(wildcard *.c *.h *.cu *.sh *.ods) Makefile README
DISTNAME:=HPC1617
CFLAGS+=-std=c99 -Wall -Wpedantic
NVCC:=nvcc
NVCFLAGS:=-Wno-deprecated-gpu-targets

ALL: mpi openmp serial simd cuda

dist: $(DISTNAME).tar.gz

$(DISTNAME).tar.gz: $(DISTFILES)
	rm -r -f $(DISTNAME)
	mkdir $(DISTNAME)
	cd $(DISTNAME)
	ln $(DISTFILES) $(DISTNAME)
	tar cfz $(DISTNAME).tar.gz $(DISTNAME)

gfx.o: gfx.c gfx.h

omp-mandelbrot: omp-mandelbrot.o gfx.o
omp-mandelbrot: LDLIBS+=-lX11 -lgomp

cache: LDLIBS+=-lrt	

$(EXE_OMP): CC=gcc
$(EXE_OMP): CFLAGS+=-fopenmp 
$(EXE_OMP): LDLIBS+=-lgomp -lrt
openmp: $(EXE_OMP)

$(EXE_MPI): CC=mpicc
mpi: $(EXE_MPI)

serial: $(EXE_SERIAL)

$(EXE_SIMD): CFLAGS += -march=native
simd: $(EXE_SIMD)

simd-vsum-auto: CFLAGS += -O2 -funsafe-math-optimizations -ftree-vectorize -ftree-vectorizer-verbose=2

simd-vsum-auto.s: simd-vsum-auto.c
	gcc -S -c -fverbose-asm -g -O2 -funsafe-math-optimizations -ftree-vectorize -ftree-vectorizer-verbose=2 $< -o $@

simd-vsum-auto.lst: simd-vsum-auto.s
	as -alhnd $< > $@

cuda: $(EXE_CUDA)

cuda-rotate cuda-matmul: NVCFLAGS+=-lm

% : %.cu
	$(NVCC) $(NVCFLAGS) $< -o $@

clean:
	\rm -f $(EXE) *.o *~ *.pbm *.s

distclean: clean
	\rm -r -f $(DISTNAME) $(DISTNAME).tar.gz
