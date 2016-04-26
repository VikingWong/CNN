#!/bin/sh

a=0

while [ $a -lt 10 ]
do
   python ./cnn.py "-batch" $a "-curriculum" "/media/olav/Data storage/dataset/Mass_inexperienced_100"
   a=`expr $a + 1`
done

while [ $a -lt 20 ]
do
   python ./cnn.py "-batch" $a "-curriculum" "/media/olav/Data storage/dataset/Mass_inexperienced_100-baseline"
   a=`expr $a + 1`
done