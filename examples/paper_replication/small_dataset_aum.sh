#!/bin/sh

if [ "$#" -ne 5 ]; then
  echo "You must enter exactly 5 command line arguments"
fi

datadir=$1
dataset=$2
seed=$3
perc_mislabeled=$4
noise_type=$5
result_dir=$6
NETTYPE="resnet"
depth=32

# General arguments for threshold sample trains
args="--data ${datadir}/${dataset} --dataset ${dataset} --net_type ${NETTYPE} --depth ${depth}"
args="${args} --perc_mislabeled ${perc_mislabeled} --noise_type ${noise_type} --seed ${seed} --use_threshold_samples"
train_args="--num_epochs 150 --lr 0.1 --wd 1e-4 --batch_size 64 --num_workers 0"
train_args="${train_args}"

# First threshold sample run
savedir1="${result_dir}/results/${dataset}_${NETTYPE}${depth}"
savedir1="${savedir1}_percmislabeled${perc_mislabeled}_${noise_type}_threshold1_seed${seed}"
cmd="python runner.py ${args} --save ${savedir1} --threshold_samples_set_idx 1 - train_for_aum_computation ${train_args} - done"
echo $cmd
if [ -z "${TESTRUN}" ]; then
  mkdir -p $savedir1
  echo $cmd > $savedir1/cmd.txt
  eval $cmd
fi

# Second threshold sample run
savedir2="${result_dir}/results/${dataset}_${NETTYPE}${depth}"
savedir2="${savedir2}_percmislabeled${perc_mislabeled}_${noise_type}_threshold2_seed${seed}"
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
savedir="${result_dir}/results/${dataset}_${NETTYPE}${depth}"
savedir="${savedir}_percmislabeled${perc_mislabeled}_${noise_type}_aumwtr_seed${seed}"
args="--data ${datadir}/${dataset} --save ${savedir} --dataset ${dataset} --net_type ${NETTYPE} --depth ${depth}"
args="${args} --perc_mislabeled ${perc_mislabeled} --noise_type ${noise_type} --seed ${seed}"
train_args="--num_epochs 300 --lr 0.1 --wd 1e-4 --batch_size 256"
train_args="${train_args} --aum_wtr ${savedir1},${savedir2}"
cmd="python runner.py ${args} - train ${train_args} - done"
echo $cmd
if [ -z "${TESTRUN}" ]; then
  mkdir -p $savedir
  echo $cmd > $savedir/cmd.txt
  eval $cmd
fi
