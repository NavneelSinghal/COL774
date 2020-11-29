#!/bin/bash

question=$1
train_data=$2
test_data=$3
output_file=$4

if [[ ${question} == "1" ]]; then
python3 Q1/q1.py $train_data $test_data > $output_file
fi

if [[ ${question} == "2" ]]; then
python3 Q2/q2.py $train_data $test_data > $output_file
fi

if [[ ${question} == "3" ]]; then
python3 Q3/q3.py $train_data $test_data > $output_file
fi
