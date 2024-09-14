#!/bin/bash


DATASET_DIR="data"
WORKSPACE="."
# # Pack csv files to hdf5
# python3 pack_csv_to_hdf5.py --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# Train
#python3 pretrain_main.py train --config example_config.json --workspace= --cuda 

# python3 main_pytorch.py train --config config/example_config_kettle_fine.json --workspace= --cuda --dataname 'data.h5'
# Inference
#python3 main_pytorch.py inference --config $WORKSPACE/example_config.json --workspace=$WORKSPACE --iteration=21000 --cuda
# python3 main_pytorch.py inference --config config/example_config_washingmachine.json --workspace=$WORKSPACE  --cuda --dataname 'uk_washingmachine.h5'
python3 main_pytorch.py inference --config config/example_config_microwave_fine.json --workspace=$WORKSPACE  --cuda --dataname 'data.h5'

