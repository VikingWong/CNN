#!/bin/sh

repeats=2
tests=( "bootstrapping" "crossentropy" )
noises=( 0.0 0.1 0.2 0.3 0.4 )
nrTests=${#tests[@]}
nrNoises=${#noises[@]}

total=$((nrTests * repeats  * nrNoises))
a=0
echo $total
while [ $a -lt $total ]
do
   config="[\""${tests[$a % $nrTests]}"\","${noises[$a % $nrNoises]}"]"
   python ./cnn.py "-batch" $a "-config" $config
   a=`expr $a + 1`
done


