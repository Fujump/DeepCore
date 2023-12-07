#!/bin/bash

cd source/cifar/defense/
# 定义多组 coreset 和 defense 参数
# coreset_options=("ContextualDiversity" "Craig" "DeepFool" "Forgetting" "Glister" "Herding" "kCenterGreedy")
defense_options=("vanilla" "relaxloss" "label_smoothing" "early_stopping")

# 循环运行脚本
# for coreset in "${coreset_options[@]}"; do
for defense in "${defense_options[@]}"; do
    CUDA_VISIBLE_DEVICES=6 python "$defense".py -name all_defense_300 --dataset CIFAR10 --model resnet20
done