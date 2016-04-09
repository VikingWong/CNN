#!/bin/sh

a=0

while [ $a -lt 10 ]
do
   python ./cnn.py "-batch" $a "-curriculum" "/media/olav/Data storage/dataset/E1Norwegian_roads_baseline-100-100"
   a=`expr $a + 1`
done

while [ $a -lt 20 ]
do
   python ./cnn.py "-batch" $a "-curriculum" "/media/olav/Data storage/dataset/E1Norwegian_roads_curriculum-100-025"
   a=`expr $a + 1`
done

while [ $a -lt 30 ]
do
   python ./cnn.py "-batch" $a "-curriculum" "/media/olav/Data storage/dataset/E1Norwegian_roads_curriculum-100-015"
   a=`expr $a + 1`
done

while [ $a -lt 40 ]
do
   python ./cnn.py "-batch" $a "-curriculum" "/media/olav/Data storage/dataset/E1Norwegian_roads_curriculum-100-035"
   a=`expr $a + 1`
done