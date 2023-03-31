mkdir -p "runs_ultra/$1_$2"
\rm -rf "runs_ultra/$1_$2/*"

py=/home/rzh/app/anaconda3/envs/rltr/bin/python
max_train_iteration=10000

if [ "$2" = "yahoo" ]; then
    dataset='Yahoo-LETOR'
elif [ "$2" = "mslr10k" ]; then
    dataset='MSLR-WEB10K'
else
    dataset='MSLR-SMALL'
    max_train_iteration=200
fi

setting="--data_dir ./data/$dataset --model_dir ./runs_ultra/$1_$2 --output_dir ./runs_ultra/$1_$2 --setting_file ./runs_ultra/$1.json"

if [ "$3" = "train" ]; then
    $py main_ultra.py $setting --max_train_iteration $max_train_iteration
else
    $py main_ultra.py $setting --test_only
fi
