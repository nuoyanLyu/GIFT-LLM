#!/bin/bash

# test new model
MODEL_PATH="all_batch2"
DEVICE="5"
# MODEL_PATH="nt-batch"
# DEVICE="1"

for i in {50..150..50}
do
    MODEL_NAME="ab2_${i}"
    # MODEL_NAME="ntb${i}"
    echo "===== Testing model: $MODEL_NAME ====="
    # game test
    python reason_test/nash-new.py --model_path $MODEL_PATH --model_name $MODEL_NAME --device $DEVICE
    python reason_test/tictactoe.py --model_path $MODEL_PATH --model_name $MODEL_NAME --device $DEVICE
    python reason_test/undercover.py --model_path $MODEL_PATH --model_name $MODEL_NAME --device $DEVICE
    # # mmlu test
    python reason_test/mmlu_dl.py --model_path $MODEL_PATH --model_name $MODEL_NAME --device $DEVICE
    python reason_test/mmlu_pro_dl.py --model_path $MODEL_PATH --model_name $MODEL_NAME --device $DEVICE
    # # math test
    # CUDA_VISIBLE_DEVICES=$DEVICE python reason_test/math_dl.py --model_path $MODEL_PATH --model_name $MODEL_NAME
    python reason_test/lv3to5_dl.py --model_path $MODEL_PATH --model_name $MODEL_NAME --device $DEVICE
    # social test
    python reason_test/social.py --model_path $MODEL_PATH --model_name $MODEL_NAME --device $DEVICE
    # common test
    python reason_test/common.py --model_path $MODEL_PATH --model_name $MODEL_NAME --device $DEVICE
done

echo "✅ All models tested successfully."

