#!/bin/bash

> mia_Neighbor_others_1111.txt
# 定义多组 coreset 和 defense 参数
# "ContextualDiversity" "Craig" "DeepFool" "Forgetting" "Glister" "Herding"     "Cal" "GraNd" "Uncertainty"     "kCenterGreedy"
# "dpsgd"    "early_stopping"     "vanilla" "advreg" "confidence_penalty" "distillation" "distillation" "label_smoothing" 
coreset_options=("Neighbor")
defense_options=("distillation" "dropout" "label_smoothing")

# 循环运行脚本
for coreset in "${coreset_options[@]}"; do
    for defense in "${defense_options[@]}"; do
        CUDA_VISIBLE_DEVICES=0 python mia_scoring.py --coreset="$coreset" --defense="$defense" >> mia_Neighbor_others_1111.txt
    done
done