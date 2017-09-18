#!/bin/bash

#at least 1 input
if [ $# -gt 0 ]
then
	
	num_tests=5

	echo "NUM_THREADS = " $1 " # of tests = " $num_tests

	if [ $# -gt 1 ]
		then
		num_tests=$2
	fi

	for((i=1; i <= $num_tests;i++))
	do
		cat sphfract.small.in | OMP_NUM_THREADS=$1 ./c-ray > foo.ppm
	done

	#Wrong inputs
	else
		echo "Wrong input, inset # of threads and eventually # of tests."
		exit
fi