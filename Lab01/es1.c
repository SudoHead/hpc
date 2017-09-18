/*
1. Somma degli elementi di un array [facile]
Scrivere un programma che calcola la somma dei valori contenuti in un array a[] di double 
(questo corrisponde all'operazione "sum reduce" di cui abbiamo parlato a lezione). 
La dimensione N dell'array può essere passata a riga di comando, oppure può essere definita hardcoded nel codice. 
Il programma deve sfruttare il parallelismo tramite il costrutto omp parallel, partizionando manualmente l'array tra i thread OpenMP. 

Procedere come segue:

1. Iniziare realizzando una versione seriale del programma. 
Includere codice per inizializzare opportunamente l'array a[], possibilmente non con valori casuali; 
conviene inizializzare l'array in modo deterministico per poter verificare la correttezza del risultato 
(es., con un valore costante, oppure con valori interi da 0 a N - 1 in modo che sia noto a priori quale deve essere il valore della somma dei valori);
2. Parallelizzare la versione seriale usando il costrutto omp parallel (inizialmente non usare omp parallel for). 
Detto P il numero di thread OpenMP, il programma deve partizionare l'array in P blocchi di dimensione approssimativamente uniforme. 
Ciascun thread determina gli estremi del sottovettore di cui è responsabile, usando il proprio ID e il valore di P, e calcola la somma della propria porzione di array. 
Le somme vengono memorizzate in un secondo array s[] di P elementi, in modo che ciascun thread scriva un elemento diverso senza causare race condition. 
Fatto questo, uno solo dei processi OpenMP (es., il master) calcola la somma dei valori in s[], determinando così il risultato cercato.
3. Realizzare quindi una nuova versione della funzione di somma usando stavolta il costrutto omp parallel for e le direttive OpenMP per l'aggiornamento atomico della somma.
Si presti attenzione a gestire correttamente nel punto 2 il caso in cui N non sia un multiplo esatto di P; nel punto 3 ciò viene fatto automaticamente dal compilatore. 
Testare il programma anche nel caso in cui N sia minore del numero di thread  P. */

/* REVIEW =  
- double a[size], per size grandi va in crash(alloca su stack) -> usare Malloc
- rand() usa val globale, no thread safe -> usare rand_r()
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define DEF_SIZE 1000

/* generate a random floating point number from min to max */
double randfrom(double min, double max) 
{
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() /div);
}

int main(int argc, char* argv[]) {
	double tstart, tstop;

	int size = argc == 2 ? atoi(argv[1]) : DEF_SIZE;

	printf("size %d\n", size);

	double *a = (double*)malloc(size * sizeof(double));

	double sum = 0.0;

	srand((unsigned int)time(0));

	tstart = omp_get_wtime();

	/*Array init*/
	#pragma omp parallel for
	for(int i = 0; i < size; i++) {
		a[i] = randfrom(0.0, 1.0);
	}

	#pragma omp parallel for reduction(+:sum)
		for (int i = 0; i < size; i++)
		{
			sum += a[i];
		}

	tstop = omp_get_wtime();

	printf("The sum is %f\n", sum );
	printf("Elapsed time %f\n", tstop - tstart);
}