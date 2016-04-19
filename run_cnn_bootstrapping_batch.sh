#!/bin/sh

repeats=9
tests=( "bootstrapping" "crossentropy" "bootstrapping_confident")
#tests=( "bootstrapping" "crossentropy" "bootstrapping_confident")
#noises=( 0.0 0.1 0.2 0.3 0.4 )
noises=( 0.0 0.3 )
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


