#!/bin/bash


CUDA_DEVICE_ID=$1
MODEL_NAME=$2


export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_ID


nohup python run.py --model="$MODEL_NAME" > "output_${MODEL_NAME}.log" 2>&1 &

echo "Script is running in the background with PID $! using model $MODEL_NAME on CUDA device $CUDA_DEVICE_ID"