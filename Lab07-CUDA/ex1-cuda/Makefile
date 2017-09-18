EXE = cuda-dot cuda-odd-even cuda-reverse
NVCC = nvcc
NVCFLAGS =

ALL: $(EXE)

cuda-dot: NVCFLAGS+=-lm

% : %.cu
	$(NVCC) $(NVCFLAGS) $< -o $@

.PHONY: clean

clean:
	\rm -f *~ $(EXE)
