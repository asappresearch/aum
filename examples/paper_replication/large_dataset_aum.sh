#!/bin/sh

if [ "$#" -ne 3 ]; then
  echo "You must enter exactly 3 command line arguments"
fi

datadir=$1
dataset=$2
seed=$3
NETTYPE="resnet50"

# General arguments for threshold sample trains
args="--data ${datadir}/${dataset} --dataset ${dataset} --net_type ${NETTYPE}"
args="${args} --seed ${seed} --num_valid 0 --use_threshold_samples"
train_args="--num_epochs 60 --lr 0.1 --wd 1e-4 --batch_size 256"
train_args="${train_args} --num_workers 0"

# First threshold sample run
savedir1="results/${dataset}_${NETTYPE}"
savedir1="${savedir1}_threshold1_seed${seed}"
cmd="python runner.py ${args} --save ${savedir1} --threshold_samples_set_idx 1 - train_for_aum_computation ${train_args} - done"
echo $cmd
if [ -z "${TESTRUN}" ]; then
  mkdir -p $savedir1
  echo $cmd > $savedir1/cmd.txt
  eval $cmd
fi

# Second threshold sample run
savedir2="results/${dataset}_${NETTYPE}"
savedir2="${savedir2}_threshold2_seed${seed}"
cmd="python runner.py ${args} --save ${savedir2} --threshold_samples_set_idx 2 - train_for_aum_computation ${train_args} - done"
echo $cmd
if [ -z "${TESTRUN}" ]; then
  mkdir -p $savedir2
  echo $cmd > $savedir2/cmd.txt
  eval $cmd
fi

# Compute AUMs for first threshold sample run
cmd="python runner.py ${args} --save ${savedir1} --threshold_samples_set_idx 1 - generate_aum_details - done"
echo $cmd
if [ -z "${TESTRUN}" ]; then
  mkdir -p ${savedir1}
eval $cmd
fi

# Compute AUMs for the second threshold sample run
cmd="python runner.py ${args} --save ${savedir2} --threshold_samples_set_idx 2 - generate_aum_details - done"
echo $cmd
if [ -z "${TESTRUN}" ]; then
  mkdir -p ${savedir2}
eval $cmd
fi

# Remove the identified mislabeled saples and retrain
savedir="results/${dataset}_${NETTYPE}"
savedir="${savedir}_aumwtr_seed${seed}"
args="--data ${datadir}/${dataset} --save ${savedir} --dataset ${dataset} --net_type ${NETTYPE}"
args="${args} --seed ${seed} --num_valid 0"
train_args="--num_epochs 180 --lr_drops 0.33,0.67 --lr 0.1 --wd 1e-4 --batch_size 256"
train_args="${train_args} --num_workers 0 --aum_wtr ${savedir1},${savedir2}"
cmd="python runner.py ${args} - train ${train_args} - done"
echo $cmd
if [ -z "${TESTRUN}" ]; then
  mkdir -p $savedir
  echo $cmd > $savedir/cmd.txt
  eval $cmd
fi
