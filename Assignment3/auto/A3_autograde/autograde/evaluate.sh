#!/bin/bash
run_dt()
{
    chmod +x $1
    $1 $2 $3 $4 $5 $6
}

run_nn()
{
    chmod +x $1
    $1 $2 $3 $4 $5 $6 $7 $8
}

compute_score()
{
    : '
        Compute score as per predicted values and write to given file
        $1 python_file
        $2 targets
        $3 predicted
        $4 outfile
    '
    python3 $1 $2 $3 $4 $5
}


main()
{
    : '
        $1: data_dir
        $2: sandbox_dir
        $3: entry_number
        $4: submissions_dir
        $5: qid
    '
    main_dir=`pwd`
    qid=$5
    batch=$6
    hidden=$7
    activation=$8
    if [ -f "${4}/${3}.zip" ]; then
        echo -e "Zip file found!"
        if [ -d "${2}/${3}" ]; then
            echo -e "Unzipped dir found!"
        else
            echo -e "Unzipping!"
            mkdir -p "${2}/${3}"
            unzip "${4}/${3}.zip" -d "${2}/${3}"
        fi
    fi

    status="FAIL"
    stud_dir_path="${main_dir}/${2}/${3}/${3}"
    stud_bdir_path="${main_dir}/${2}/${3}"
    stud_bfname="${stud_bdir_path}/${run_script}"
    stud_fname="${stud_dir_path}/${run_script}"
    data_dir_path="${main_dir}/${1}"
    compute_accuracy="${main_dir}/compute_accuracy.py"

    if [ -d "${stud_dir_path}" ]; then
        if [ -f ${stud_fname} ]; then
            sed -i 's/\r$//' ${stud_fname} # Handle windows file endings
            status="OK"
        else
            echo -e "${run_script} not found!"
        fi
    else
        echo -e "Bad directory structure!"
        if [ -f ${stud_bfname} ]; then
            status="OK"
            sed -i 's/\r$//' ${stud_bfname} # Handle windows file endings
            stud_fname="${stud_bfname}"
            stud_dir_path="${stud_bdir_path}"
        else
            echo -e "${run_script} not found!"
        fi
    fi

    if [ $status == "OK" ]; then
        echo -e "Running.."
        cd "$stud_dir_path"

        if [ ${qid} -eq 1 ] || [ ${qid} -eq 2 ]; then
            time run_dt "${stud_fname}" "${qid}" "${data_dir_path}/${train}" "${data_dir_path}/${validation}" "${data_dir_path}/${test}" "${stud_dir_path}/predictions_q${qid}"
            compute_score "${compute_accuracy}" "${data_dir_path}/${test_gt}" "${stud_dir_path}/predictions_q${qid}" "${stud_dir_path}/result_q${qid}"
        else    # For neural Networks
            time run_nn "${stud_fname}" "${data_dir_path}/${X_train}" "${data_dir_path}/${y_train}" "${data_dir_path}/${X_test}" "${stud_dir_path}/predictions_q${qid}" "${batch}" "${hidden}" "${activation}"
            compute_score "${compute_accuracy}" "${data_dir_path}/${y_test_gt}" "${stud_dir_path}/predictions_q${qid}" "${stud_dir_path}/result_q${qid}"
        fi

        if [ -f "${stud_dir_path}/result_q${qid}" ]; then
            cp "${stud_dir_path}/result_q${qid}" "${stud_bdir_path}/result_q${qid}"
        fi
        cd $main_dir
    fi
}


# change these filenames as per your requirement
# label file should be a text file with an ground truth class in each line
# Use dummy labels in dtest; if possible

# Decision Tree
if [ "$5" -eq 1 ] || [ "$5" -eq 2 ]; then
    train="train_test.csv"
    test="test_test.csv"
    validation="val_test.csv"
    test_gt="test_test_gt.csv"

    run_script="run_dt.sh"
else    # Neural Networks
    X_train="X_train.npy"
    y_train="y_train.npy"
    X_test="X_test.npy"
    y_test="y_test.npy"
    y_test_gt="y_test_gt.txt"

    run_script="run_nn.sh"
    batch=100
    hidden="100 10"
    activation="relu"
fi

# $data_dir $sandbox_dir $entry_num $submissions_dir $qid
main $1 $2 $3 $4 $5 ${batch} $hidden $activation
