#!/bin/sh

a=0

while [ $a -lt 5 ]
do
   python ./cnn.py "-batch" $a
   a=`expr $a + 1`
done


