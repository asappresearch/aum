#!/bin/sh

if [ "$#" -ne 3 ]; then
  echo "You must enter exactly 3 command line arguments"
fi

datadir=$1
DATASET=$2
seed=$3
NETTYPE="resnet50"

savedir="results/${DATASET}_${NETTYPE}"
savedir="${savedir}_baseline_seed${seed}"

args="--data /home/ubuntu/${DATASET} --save ${savedir} --dataset ${DATASET} --net_type ${NETTYPE}"
args="${args} --seed ${seed} --num_valid 0"

train_args="--num_epochs 180 --lr_drops 0.33,0.67 --lr 0.1 --wd 1e-4 --batch_size 256"
train_args="${train_args} --num_workers 0"

cmd="python runner.py ${args} - train ${train_args} - done"
echo $cmd
if [ -z "${TESTRUN}" ]; then
  mkdir -p $savedir
  echo $cmd > $savedir/cmd.txt
  eval $cmd
fi
