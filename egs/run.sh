#!/bin/bash

set -eu

stage=0

# stage 0 - train/test splits and plot token length distribution
# stage 1 - train model
# stage 2 - average model parameters
# stage 3 - evaluate
# stage 4 - compare

# Load deps
. path.sh
. cmd.sh
. parse_options.sh || exit

# datasets
data_dir=../data/train/


# split data
split_data_dir=exp/split_data_dir
if [ $stage -le 0 ]; then
    echo "Processing task performed at $split_data_dir"
    if [ -d $split_data_dir ]; then
        echo "$split_data_dir already exists."
        echo " if you want to retry, delete it."
        exit 1
    fi
    work=$split_data_dir/.work
    mkdir -p $work
    $process_cmd $work/process.log \
        python -u ../models/split_data.py $data_dir \
            $split_data_dir \
            --stage 0 \
        || exit 1
fi
exit
