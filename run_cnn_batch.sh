#!/bin/sh

a=0

while [ $a -lt 10 ]
do
   python ./cnn.py "-batch" $a "-curriculum" "/media/olav/Data storage/dataset/E6-mass-curriculum-1600"
   a=`expr $a + 1`
done
