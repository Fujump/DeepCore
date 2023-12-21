import numpy as np
import os 
n=10
mean_dists=np.load("mean_dists_of_nn.npy")
sorted_indices = np.argsort(mean_dists)[::-1]

dists_indices = np.array(sorted_indices)
os.mkdir("/data/home/huqiang/DeepCore/save/dists_of_knn/")
for i in range(n):
    np.save(f"/data/home/huqiang/DeepCore/save/dists_of_knn/sorted_indices{i}_10",dists_indices[i*1200:(i+1)*1200])
