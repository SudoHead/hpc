#!/bin/bash

#usage1: ./myBenchmark "./traffic [nsteps [rho [N]]]" nTest
#options: 1)code to execute; 2) num of tests to carry

#usage2: ./myBenchmark -omp num_threads "./omp-traffic [nsteps [rho [N]]]" nTest
#options: 1)OMP benchmark 2)OMP_NUM_THREADS variable
#					3)code to execute; 4) num of tests to carry

regex='[+-]?[0-9]+([.][0-9]+)?'

num_tests=5
num_tests=${@: -1}

#at least 1 input
if [ $# -gt 0 ]
then

	echo ""

	if [ "$1" == "-omp" ]
	then

		echo "Programma: " $3 " | num_test = " $num_tests
		echo "OMP # threads = " $2
		for((i=1; i <= $num_tests; i++))
		do
			echo `OMP_NUM_THREADS=$2 $3` | grep -Eo $regex
		done

	else

		echo "Programma: " $1 " | num_test = " $num_tests
		for((i=1; i <= $num_tests;i++))
		do
			echo `$1` | grep -Eo $regex
		done

	fi

	echo ""

	#Wrong inputs
	else
		echo "Wrong input, inset the command to execute, # of threads and eventually # of tests."
		exit
fi
