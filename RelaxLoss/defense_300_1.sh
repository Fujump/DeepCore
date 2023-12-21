#!/bin/bash

# cd source/cifar/defense/
# # 定义多组 coreset 和 defense 参数
# # coreset_options=("ContextualDiversity" "Craig" "DeepFool" "Forgetting" "Glister" "Herding" "kCenterGreedy")
# defense_options=("vanilla" "relaxloss" "label_smoothing" "early_stopping")

# # 循环运行脚本
# # for coreset in "${coreset_options[@]}"; do
# for defense in "${defense_options[@]}"; do
#     CUDA_VISIBLE_DEVICES=6 python "$defense".py -name all_defense_300 --dataset CIFAR10 --model resnet20
# done

cd /data/home/huqiang/DeepCore/
> 1214weighted_by_mean_dists_sigmoid.txt
coreset_options=("dists_of_knn")
defense_options=("relaxloss")
delta_options=(0 -0.1 -0.2 -0.3)
metric_options=("value_s")

# 循环运行脚本
for coreset in "${coreset_options[@]}"; do
    for defense in "${defense_options[@]}"; do
        for metric in "${metric_options[@]}"; do
            for delta in "${delta_options[@]}"; do
                python mia_scoring.py --coreset="$coreset" --defense="$defense" --gpu_idx=3 --model_name weighted_data"$metric"_"$delta" >> 1214weighted_by_mean_dists_sigmoid.txt
            done
        done
    done
done