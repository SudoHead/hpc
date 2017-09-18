#!/bin/sh

# Questo script exegue il programma omp-matmul sfruttando OpenMP con
# un numero di core da 1 a 8 (estremi inclusi). Il test con p
# processori viene effettuato su un input che ha dimensione (p * N0 *
# N0 * N0)^(1/3), dove N0 e' la dimensione dell'input nel caso base.
# In altre parole, il test con p processori viene effettuato su un
# input che richiede (in teoria) p volte il tempo richiesto dal caso
# base. Pertanto, questo script puo' essere utilizzato per stimare la
# weak scaling efficiency.

echo "p\tt1\tt2\tt3\tt4\tt5"

N0=500 # base problem size

for p in `seq 1 8`; do
    echo -n "$p\t"
    PROB_SIZE=`echo "e(l($N0 * $N0 * $N0 * $p)/3)" | bc -l`
    for rep in `seq 1 5`; do
#        EXEC_TIME="$( OMP_NUM_THREADS=$p \time -f '%e' ./omp-matmul $PROB_SIZE $n 2>&1 1>/dev/null )"
        EXEC_TIME="$( OMP_NUM_THREADS=$p ./omp-matmul $PROB_SIZE $n | sed 's/Execution time //' )"
        echo -n "${EXEC_TIME}\t"
    done
    echo ""
done
