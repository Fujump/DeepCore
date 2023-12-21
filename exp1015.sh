#!/bin/bash

> mia_HSJ_1112.txt
# 定义多组 coreset 和 defense 参数
# "Craig" "ContextualDiversity"
# "early_stopping"  "dpsgd" "relaxloss"   
# "vanilla" "advreg" "confidence_penalty" "distillation" "dropout" "label_smoothing"
coreset_options=("HSJ")
defense_options=("dropout")

# 循环运行脚本
for coreset in "${coreset_options[@]}"; do
    for defense in "${defense_options[@]}"; do
        CUDA_VISIBLE_DEVICES=7 python mia_scoring.py --coreset="$coreset" --defense="$defense" >> mia_HSJ_1112.txt
    done
done