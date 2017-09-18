EXE = c-ray
ALL: $(EXE)

CFLAGS += -O3 -ffast-math -std=c99 -Wpedantic -Wall -fopenmp
LDLIBS += -lm

.PHONY: clean
clean:
	rm -f $(EXE) $(EXE).o *~ *.ppm

