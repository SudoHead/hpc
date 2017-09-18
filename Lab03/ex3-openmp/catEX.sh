#!/bin/bash

num_tests=684

for((i=1; i <= $num_tests;i++))
	do
		./omp-cat-map $i < cat.pgm > cat.$i.pgm
	done