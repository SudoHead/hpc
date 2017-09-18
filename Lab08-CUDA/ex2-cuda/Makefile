EXE=cuda-matsum cuda-rule30 cuda-cat-map
NVCC:=nvcc
NVCFLAGS:=-Wno-deprecated-gpu-targets

ALL: $(EXE)

cuda-matsum: NVCFLAGS+=-lm

% : %.cu
	$(NVCC) $(NVCFLAGS) $< -o $@

.PHONY: clean

clean:
	\rm -f *~ $(EXE) rule30.pbm
