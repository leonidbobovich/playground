#!/bin/bash

#eval "$(conda shell.bash hook)"
#conda activate llama_factory

MODEL_NAME=phi-2
STAGE=sft
EPOCH=.01 #3.0
#DATA=alpaca_gpt4_zh
DATA=alpaca_gpt4_en
SAVE_PATH=./models/$STAGE/$MODEL_NAME-$STAGE-$DATA-$EPOCH
SAVE_PATH_PREDICT=$SAVE_PATH/Predict
MODEL_PATH=./models/$MODEL_NAME
LoRA_TARGET=Wqkv #q_proj,v_proj
TEMPLATE=default
PREDICTION_SAMPLES=20

if [ ! -d $MODEL_PATH ]; then
    echo "Model not found: $MODEL_PATH"
    return 1
fi

if [ ! -d $SAVE_PATH ]; then
    mkdir -p $SAVE_PATH
fi

if [ ! -d $SAVE_PATH_PREDICT ]; then
    mkdir -p $SAVE_PATH_PREDICT
fi

#    --val_max_sample 20 \
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --seed 42 \
    --stage $STAGE \
    --model_name_or_path $MODEL_PATH \
    --dataset $DATA \
    --val_size .1 \
    --finetuning_type lora \
    --do_train \
    --lora_target $LoRA_TARGET \
    --output_dir $SAVE_PATH \
    --overwrite_output_dir \
    --overwrite_cache \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs $EPOCH \
    --do_eval \
    --evaluation_strategy steps \
    --per_device_eval_batch_size 1 \
    --prediction_loss_only \
    --plot_loss \
    --quantization_bit 4 \
    --template $TEMPLATE \
    |& tee $SAVE_PATH/train_eval_log.txt

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage $STAGE \
    --model_name_or_path $MODEL_PATH \
    --do_predict \
    --max_samples $PREDICTION_SAMPLES \
    --predict_with_generate \
    --dataset $DATA \
    --template $TEMPLATE \
    --finetuning_type lora \
    --adapter_name_or_path $SAVE_PATH \
    --output_dir $SAVE_PATH_PREDICT \
    --per_device_eval_batch_size 1 \
    |& tee $SAVE_PATH_PREDICT/predict_log.txt
