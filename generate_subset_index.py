import os
import torch
import argparse
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.models import resnet18

import deepcore.methods as methods
import deepcore.datasets as datasets
from RelaxLoss.source.cifar.dataset import CIFAR10, CIFAR100
from RelaxLoss.source.utils.misc import *


def generate_subset_index():
    # pin_memery=False
    # torch.set_num_threads(1)
    transform_train = transform_test = transforms.Compose([transforms.ToTensor(),
                                                               transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                    (0.2023, 0.1994, 0.2010))])
    indices=np.load('/data/home/huqiang/DeepCore/RelaxLoss/results/CIFAR10/resnet20/advreg/full_idx.npy')
    partition = Partition(dataset_size=60000, indices=indices)
    trainset_idx, testset_idx = partition.get_target_indices()##乱序12000
    target_trainset = CIFAR10(root='/data/home/huqiang/DeepCore/RelaxLoss/data', indices=trainset_idx,
                                       download=True, transform=transform_train)##打乱后的前12000个
    target_trainloader = torch.utils.data.DataLoader(target_trainset, batch_size=64, shuffle=False) 
    
    weights_pretrained = torch.load('/data/home/huqiang/DeepCore/weights_resnet18_cifar10.pth', map_location="cuda:0")
    model = resnet18(weights=None, num_classes=10)
    model.load_state_dict(weights_pretrained)
    model.eval()

    args=argparse.Namespace()
    args.model='ResNet18'
    args.dataset='CIFAR10'
    args.data_path='/data/home/huqiang/DeepCore/data'
    args.device = 'cuda'
    args.gpu=[0]
    args.selection_optimizer = "SGD"
    args.selection_lr=0.1
    args.selection_momentum=0.9
    args.selection_weight_decay=5e-4
    args.selection_nesterov=True
    args.selection_batch=32
    args.workers=4
    args.print_freq=20
    args.lr=0.1
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[args.dataset] \
            (args.data_path)
    args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names

# methods.ContextualDiversity,methods.Craig,methods.DeepFool,methods.Forgetting,methods.Glister,methods.Herding,methods.kCenterGreedy,
# methods.Submodular,methods.Uniform
    n=10
    method_list=[methods.ContextualDiversity,methods.Craig,methods.DeepFool,methods.Forgetting,methods.Glister,methods.Herding,methods.kCenterGreedy]
    for method in method_list:
        subset_index=[]
        for i in range(n):
            selection=method(dst_train=target_trainset,
                                train_loader=target_trainloader, 
                                args=args, 
                                fraction=0.1*(i+1),
                                random_seed=42,
                                epochs=20,
                                balance=False)
            selection.model=model
            new_select_index=selection.select()['indices']
            if(i!=0):
                subset_index=[x for x in new_select_index if x not in select_index]
            else:
                subset_index=new_select_index
            select_index=new_select_index

            folder_name=f"/data/home/huqiang/DeepCore/save/{method.__name__}/"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            np.save(f"/data/home/huqiang/DeepCore/save/{method.__name__}/sorted_indices{n-1-i}_10.npy",subset_index)

generate_subset_index()