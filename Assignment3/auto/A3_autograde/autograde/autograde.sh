#!/bin/bash
# RUN: ./autograde.sh users.txt log.txt scores.txt ./data ./sandbox ./submissions qid

# 1 input params
infile=$1
logfile=$2
scorefile=$3
data_dir=$4
sandbox_dir=$5
submissions_dir=$6
qid=$7

echo -e "--\nENTRY_NUM, ACC, TIME" >> $scorefile
echo -e "Autograding...." >> $logfile

while read entry_num; do
    #Check if user submitted the correct file
    echo -e "Evaluating: ${entry_num}"
    user_score=""
    echo -e "Evaluating: ${entry_num}" >> $logfile
    if [ -f  "${submissions_dir}/${entry_num}.zip" ] || [ -f  "${submissions_dir}/${entry_num}.rar" ]; then
        echo "File found" >> $logfile
        #If correct zip file exist
        #Run for allowed time
        time_start=$(date +%s)
        # Change the time as required
        if [ ${qid} -eq 1 ] || [ ${qid} -eq 2 ]; then
            timeout -k 14400s 14400s ./evaluate.sh $data_dir $sandbox_dir $entry_num $submissions_dir $qid >> $logfile
            status=$?
        else    # For neural Networks
            timeout -k 14400s 14400s ./evaluate.sh $data_dir $sandbox_dir $entry_num $submissions_dir $qid >> $logfile
            status=$?
        fi
        time_end=$(date +%s)
        user_time=$(( time_end - time_start ))
        if [ $status == 124 ]; then
            echo -e "Status: Timed out!" >> $logfile
        else
            echo -e "Status: OK, Time taken: ${user_time}" >> $logfile
        fi
        if [ -f  "${sandbox_dir}/${entry_num}/result_q$qid" ]; then
            user_score=`cat "${sandbox_dir}/${entry_num}/result_q$qid"`
        else
            user_score="0.0"
        fi
        echo -e "Scores: ${user_score}" >> $logfile
        user_score+=","
        user_score+=${user_time}
    else
        #If the correct zip file doesn't exist
        echo -e "FILE NOT FOUND!" >> $logfile
        user_score=",NA,NA"
    fi
    echo -e "\n--------------------\n" >> $logfile
    echo -e "${entry_num},${user_score}" >> $scorefile
    #rm -r $5/*.py
    #rm -rf $5/__MACOSX
done < $infile
