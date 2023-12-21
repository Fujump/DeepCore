import os
import torch
import argparse
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import pairwise_distances

from RelaxLoss.source.cifar import models
from RelaxLoss.source.cifar.dataset import CIFAR10, CIFAR100

# Define your custom dataset and DataLoader
parser = argparse.ArgumentParser(description='Parameter Processing')
parser.add_argument('--method', type=str, default=None, help='count or distance')
# parser.add_argument('--defense', type=str, default=None, help='defense')
args = parser.parse_args()

# Load the model (e.g., ResNet-20)
model=models.__dict__["resnet20"](num_classes=10)
checkpoint = torch.load(os.path.join("/data/home/huqiang/DeepCore/RelaxLoss/results/CIFAR10/resnet20/relaxloss/all_defense_300/checkpoint.pkl"))
new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}# target_model.load_state_dict(checkpoint['model_state_dict'])
model.load_state_dict(new_state_dict)

# Set the model to evaluation mode
model.eval()

# Define a threshold for neighbor distance
threshold = 0.5  # You can adjust this value
num=5

# Calculate the embeddings for all data samples
embeddings = []
targets = []

transform_train = transform_test = transforms.Compose([transforms.ToTensor(),
                                                               transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                    (0.2023, 0.1994, 0.2010))])
indices=np.load('/data/home/huqiang/DeepCore/RelaxLoss/results/CIFAR10/resnet20/advreg/full_idx.npy')
# partition = Partition(dataset_size=60000, indices=indices)
# trainset_idx, testset_idx = partition.get_target_indices()
target_trainset = CIFAR10(root='/data/home/huqiang/DeepCore/RelaxLoss/data', indices=indices[:12000],
                                       download=True, transform=transform_train)    
target_trainloader = torch.utils.data.DataLoader(target_trainset, batch_size=64, shuffle=False)                           


with torch.no_grad():
    for data, target in target_trainloader:
        output = model(data)
        embeddings.append(output)
        targets.append(target)

embeddings = torch.cat(embeddings)  # Concatenate all embeddings
targets = torch.cat(targets)  # Concatenate all targets

# Calculate pairwise distances between embeddings
pairwise_dist = pairwise_distances(embeddings, embeddings)
print(np.min(pairwise_dist))
if (args.method=="count"):
    # Count the number of neighbors within the threshold for each data point
    num_neighbors = (pairwise_dist <= threshold).sum(axis=1)

    np.save("num_nei.npy", num_neighbors)
    # Sort the indices based on the number of neighbors
    sorted_indices = np.argsort(num_neighbors)[::-1]
    sorted_indices=np.array(sorted_indices)
elif(args.method=="distance"):
    mean_dists=[]
    for i in range(len(pairwise_dist)):
        # print(f"before{np.mean(pairwise_dist[i][:2])}")
        sorted_dist=np.sort(pairwise_dist[i])
        # print(f"end{np.mean(sorted_dist[:2])}")
        mean_dist=np.mean(sorted_dist[:num])
        mean_dists.append(mean_dist)
    np.save("mean_dists_of_nn.npy", mean_dists)

# os.mkdir("/data/home/huqiang/DeepCore/save/Neighbor/")
# n=10
# for i in range(n):
#     np.save(f"/data/home/huqiang/DeepCore/save/Neighbor/sorted_indices{i}_10",sorted_indices[i*1200:(i+1)*1200])
