#!/bin/bash

> 1120weighted_by_mean_dists.txt
# 定义多组 coreset 和 defense 参数
# "Craig" "ContextualDiversity"
# "early_stopping"  "dpsgd" "relaxloss"   
# "vanilla" "advreg" "confidence_penalty" "distillation" "dropout" "label_smoothing"
coreset_options=("dists_of_knn")
defense_options=("relaxloss")
delta_options=(-0.5 -0.4 -0.3 -0.2 -0.1 0 0.1 0.2 0.3 0.4 0.5)
metric_options=("index" "value")

# 循环运行脚本
for coreset in "${coreset_options[@]}"; do
    for defense in "${defense_options[@]}"; do
        for metric in "${metric_options[@]}"; do
            for delta in "${delta_options[@]}"; do
                CUDA_VISIBLE_DEVICES=7 python mia_scoring.py --coreset="$coreset" --defense="$defense" --model_name weighted_data"$metric"_"$delta" >> 1120weighted_by_mean_dists.txt
            done
        done
    done
done