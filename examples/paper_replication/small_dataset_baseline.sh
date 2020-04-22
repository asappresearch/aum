#!/bin/sh

if [ "$#" -ne 5 ]; then
  echo "You must enter exactly 5 command line arguments"
fi

datadir=$1
dataset=$2
seed=$3
perc_mislabeled=$4
noise_type=$5
NETTYPE="resnet"
depth=32

savedir="results/${dataset}_${NETTYPE}${depth}"
savedir="${savedir}_percmislabeled${perc_mislabeled}_${noise_type}_baseline_seed${seed}"

args="--data ${datadir}/${dataset} --save ${savedir} --dataset ${dataset} --net_type ${NETTYPE} --depth ${depth}"
args="${args} --perc_mislabeled ${perc_mislabeled} --noise_type ${noise_type} --seed ${seed}"

train_args="--num_epochs 300 --lr 0.1 --wd 1e-4 --batch_size 256"
train_args="${train_args}"

cmd="python runner.py ${args} - train ${train_args} - done"
echo $cmd
if [ -z "${TESTRUN}" ]; then
  mkdir -p $savedir
  echo $cmd > $savedir/cmd.txt
  eval $cmd
fi
