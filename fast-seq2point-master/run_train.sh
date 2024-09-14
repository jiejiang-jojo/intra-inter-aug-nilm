#!/bin/bash

if [ "$#" -lt 3 ]; then
  echo "USAGE: run_train.sh /home/shijie/nilm/fast-seq2point-master/example_config.json /home/shijie/miniconda3/envs/ /home/shijie/nilm/fast-seq2point-master/"
  exit
fi

CONDA_ENV_NAME=sci

CONFIG="$1"
shift


CONDA_ENV_PATH="$1"
if [ ! -d "$CONDA_ENV_PATH/$CONDA_ENV_NAME" ]; then
  echo "Error: Please make sure the second parameter ($CONDA_ENV_PATH) pointing to a valid conda env folder containing '$CONDA_ENV_NAME'"
  exit
fi
shift


WORKSPACE="$1"
shift

export PATH="$CONDA_ENV_PATH/bin:$PATH"

cd "$WORKSPACE"
source activate $CONDA_ENV_NAME

$CONDA_ENV_PATH/$CONDA_ENV_NAME/bin/python3.6 main_pytorch.py train --config $CONFIG --workspace=$WORKSPACE --cuda $*