#!/bin/bash

#options: 1)code to execute; 2) num of threads 3) num of tests to carry

#at least 1 input
if [ $# -gt 0 ]
then

	num_tests=5

	if [ $# -gt 2 ]
	then
		num_tests=$3
	fi

	echo "NUM_THREADS = " $2 " # of tests = " $num_tests

	for((i=1; i <= $num_tests;i++))
	do
		OMP_NUM_THREADS=$2 $1
	done

	#Wrong inputs
	else
		echo "Wrong input, inset the command to execute, # of threads and eventually # of tests."
		exit
fi
