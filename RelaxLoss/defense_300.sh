#!/bin/bash
# CUDA_VISIBLE_DEVICES=5 python distillation.py -name all_defense_300 --dataset CIFAR10 --model resnet20 --teacher_path "/data/home/huqiang/DeepCore/RelaxLoss/results/CIFAR10/resnet20/vanilla/all_defense_300"

cd source/cifar/defense/
# 定义多组 coreset 和 defense 参数
# coreset_options=("ContextualDiversity" "Craig" "DeepFool" "Forgetting" "Glister" "Herding" "kCenterGreedy")
# "advreg" "confidence_penalty"
# "label_smoothing" "relaxloss" "vanilla"     "dpsgd" "dropout" "early_stopping"
defense_options=("relaxloss")
delta_options=(-0.3 -0.2 -0.1 0 0.1 0.2 0.3)
metric_options=("index_s" "value_s")

# 循环运行脚本
for metric in "${metric_options[@]}"; do
    for delta in "${delta_options[@]}"; do
        for defense in "${defense_options[@]}"; do
            CUDA_VISIBLE_DEVICES=5 python "$defense"_m.py -name weighted_data"$metric"_"$delta" --dataset CIFAR10 --model resnet20 --delta "$delta" --weight_metric "$metric"
        done
    done
done

> 1204weighted_by_mean_dists_sigmoid.txt
# 定义多组 coreset 和 defense 参数
# "Craig" "ContextualDiversity"
# "early_stopping"  "dpsgd" "relaxloss"   
# "vanilla" "advreg" "confidence_penalty" "distillation" "dropout" "label_smoothing"
cd /data/home/huqiang/DeepCore/
coreset_options=("dists_of_knn")
defense_options=("relaxloss")
delta_options=(-0.3 -0.2 -0.1 0 0.1 0.2)
metric_options=("index_s" "value_s")

# 循环运行脚本
for coreset in "${coreset_options[@]}"; do
    for defense in "${defense_options[@]}"; do
        for metric in "${metric_options[@]}"; do
            for delta in "${delta_options[@]}"; do
                CUDA_VISIBLE_DEVICES=7 python mia_scoring.py --coreset="$coreset" --defense="$defense" --model_name weighted_data"$metric"_"$delta" >> 1204weighted_by_mean_dists_sigmoid.txt
            done
        done
    done
done