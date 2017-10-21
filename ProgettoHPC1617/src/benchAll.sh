#!/bin/bash

#usage: ./benchAll [nsteps [rho [N]]]
#inputs: 1) n. of steps 2) ro number 3) size of the image

#all inputs required

regex='[+-]?[0-9]+([.][0-9]+)?'
div="-----------------------------------------------------"

if [ $# -gt 2 ]
then

  echo ""
  echo -e "\tBenchmark config: " $1 " " $2 " " $3

  echo +$div+

  echo -n -e "\tSerial= \t\t\t"
  tSerial=`echo $(./traffic $1 $2 $3) | grep -Eo $regex`
  echo $tSerial

  echo $div

  echo -n -e "\tOMP= \t\t\t\t"
  tOmp1=`echo $(./omp-traffic $1 $2 $3) | grep -Eo $regex`
  echo $tOmp1 "(`awk "BEGIN {print $tSerial/$tOmp1}"`x speedup)"
  cmp traffic-$1.ppm omp-traffic-$1.ppm

  echo $div

  echo -n -e "\tOMP-V2= \t\t\t"
  tOmp2=`echo $(./omp-trafficV2 $1 $2 $3) | grep -Eo $regex`
  echo $tOmp2 "(`awk "BEGIN {print $tSerial/$tOmp2}"`x speedup)"
  cmp traffic-$1.ppm omp-trafficV2-$1.ppm

  echo $div

  echo -n -e "\tCuda w/o shared memory = \t"
  tcuda=`echo $(./cuda-traffic $1 $2 $3) | grep -Eo $regex`
  echo $tcuda "(`awk "BEGIN {print $tSerial/$tcuda}"`x speedup)"
  cmp traffic-$1.ppm cuda-traffic-$1.ppm

  echo $div

  echo -n -e "\tCuda w/ shared memory = \t"
  tcudaV2=`echo $(./cuda-trafficV2 $1 $2 $3) | grep -Eo $regex`
  echo $tcudaV2 "(`awk "BEGIN {print $tSerial/$tcudaV2}"`x speedup)"
  cmp traffic-$1.ppm cuda-trafficV2-$1.ppm

  echo +$div+
  echo ""

else

  echo usage: ./benchAll [nsteps [rho [N]]]

fi
