data_dir=$1
out_dir=$2
question=$3
part=$4
if [[ ${question} == "1" ]]; then
python3 Q1/q1.py $data_dir $out_dir $part
fi
if [[ ${question} == "2" ]]; then
python3 Q2/q2.py $data_dir $out_dir $part
fi
if [[ ${question} == "3" ]]; then
python3 Q3/q3.py $data_dir $out_dir $part
fi
if [[ ${question} == "4" ]]; then
python3 Q4/q4.py $data_dir $out_dir $part
fi
