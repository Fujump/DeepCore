import os
import sys
import abc
import pdb
import pickle
import random
import requests
import argparse
import numpy as np
from sklearn import linear_model, model_selection
import torch
import torch.nn as nn
from sklearn import metrics
from tqdm import tqdm
import torch.optim as optim
import multiprocessing as mp
from scipy.stats import norm, kurtosis, skew
from progress.bar import Bar as Bar
import torch.utils.data as data
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset

import torchvision
from torchvision import transforms
from torchvision import models
from torchvision.models import resnet18


import deepcore.nets as nets
import deepcore.datasets as datasets
import deepcore.methods as methods
from deepcore.methods import *
from utils import *
from datetime import datetime
from time import sleep
from RelaxLoss.source import cifar
from RelaxLoss.source.utils.misc import *
from RelaxLoss.source.utils.base import BaseTrainer
from RelaxLoss.source.utils.logger import AverageMeter, Logger
from RelaxLoss.source.utils.eval import accuracy, accuracy_binary, metrics_binary
from RelaxLoss.source.cifar import models
from RelaxLoss.source import utils
from RelaxLoss.source.cifar import defense
from RelaxLoss.source.cifar.dataset import CIFAR10, CIFAR100
# from RelaxLoss.source.cifar import run_attacks
import metric



parser = argparse.ArgumentParser(description='Parameter Processing')
parser.add_argument('--coreset', type=str, default=None, help='coreset')
parser.add_argument('--defense', type=str, default=None, help='defense')
parser.add_argument('--model_name',type=str, default="all_defense_300",help='name of defensed model' )
parser.add_argument('--gpu_idx',type=str, default="0",help='idx of gpu' )
super_args = parser.parse_args()



DEVICE = f"cuda:{super_args.gpu_idx}" if torch.cuda.is_available() else "cpu"
print("Running on device:", DEVICE.upper())

RNG = torch.Generator().manual_seed(42)
# mp.set_start_method('spawn',force=True)

sys.path.append("./RelaxLoss/source/cifar/models")
# os.environ["CUDA_VISIBLE_DEVICES"]="0"



args=argparse.Namespace()
args.device=DEVICE
# args.selection_batch=None
args.workers=4
args.dataset='CIFAR10'
args.data_path='data'
args.selection_epochs=1
# args.uncertainty="Entropy"
args.balance=False
# args.submodular_greedy="LazyGreedy"
# args.submodular="GraphCut"
# args.selection="Uniform"
# args.fraction=1
# args.seed=int(time.time() * 1000) % 100000
args.seed=1000
args.print_freq=20
args.batch=256
args.selection_batch=256
args.model='ResNet18'
args.gpu=[0]
args.selection_optimizer="SGD"
args.selection_lr=0.1
args.selection_weight_decay=5e-4
args.selection_nesterov=True
args.selection_momentum=0.9

# args.method='advreg'
args.test_batchsize=64
args.random_seed=1000



# download and pre-process CIFAR10
normalize = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

train_set = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=normalize
)
train_loader = DataLoader(train_set, batch_size=128, shuffle=False, num_workers=2)

# we split held out data into test and validation set
held_out = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=normalize
)
test_set, val_set = torch.utils.data.random_split(held_out, [0.5, 0.5], generator=RNG)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=2)

# for the unlearning algorithm we'll also need a split of the train set into
# forget_set and a retain_set
forget_set, retain_set = torch.utils.data.random_split(train_set, [0.1, 0.9], generator=RNG)
forget_loader = torch.utils.data.DataLoader(
    forget_set, batch_size=128, shuffle=True, num_workers=2
)
retain_loader = torch.utils.data.DataLoader(
    retain_set, batch_size=128, shuffle=True, num_workers=2, generator=RNG
)



# download pre-trained weights
local_path = "weights_resnet18_cifar10.pth"
if not os.path.exists(local_path):
    response = requests.get(
        "https://unlearning-challenge.s3.eu-west-1.amazonaws.com/weights_resnet18_cifar10.pth"
    )
    open(local_path, "wb").write(response.content)

weights_pretrained = torch.load(local_path, map_location=DEVICE)

# load model with pre-trained weights
model = resnet18(weights=None, num_classes=10)
model.load_state_dict(weights_pretrained)
model.to(DEVICE)
model.eval()



class Benchmark(object):
    def __init__(self, shadow_train_scores, shadow_test_scores, target_train_scores, target_test_scores):
        self.s_tr_scores = shadow_train_scores
        self.s_te_scores = shadow_test_scores
        self.t_tr_scores = target_train_scores
        self.t_te_scores = target_test_scores
        self.num_methods = len(self.s_tr_scores)

    def load_labels(self, s_tr_labels, s_te_labels, t_tr_labels, t_te_labels, num_classes):
        """Load sample labels"""
        self.num_classes = num_classes
        self.s_tr_labels = s_tr_labels
        self.s_te_labels = s_te_labels
        self.t_tr_labels = t_tr_labels
        self.t_te_labels = t_te_labels

    def _thre_setting(self, tr_values, te_values):
        """Select the best threshold"""
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values >= value) / (len(tr_values) + 0.0)
            te_ratio = np.sum(te_values < value) / (len(te_values) + 0.0)
            acc = 0.5 * (tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre

    def _mem_inf_thre_perclass(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
        """MIA by thresholding per-class feature values """
        t_tr_mem, t_te_non_mem = 0, 0
        for num in range(self.num_classes):
            thre = self._thre_setting(s_tr_values[self.s_tr_labels == num], s_te_values[self.s_te_labels == num])
            t_tr_mem += np.sum(t_tr_values[self.t_tr_labels == num] >= thre)
            t_te_non_mem += np.sum(t_te_values[self.t_te_labels == num] < thre)
        mem_inf_acc = 0.5 * (t_tr_mem / (len(self.t_tr_labels) + 0.0) + t_te_non_mem / (len(self.t_te_labels) + 0.0))
        info = 'MIA via {n} (pre-class threshold): the attack acc is {acc:.3f}'.format(n=v_name, acc=mem_inf_acc)
        print(info)
        return info, mem_inf_acc

    def _mem_inf_thre(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
        """MIA by thresholding overall feature values"""
        t_tr_mem, t_te_non_mem = 0, 0
        thre = self._thre_setting(s_tr_values, s_te_values)
        t_tr_mem += np.sum(t_tr_values >= thre)
        t_te_non_mem += np.sum(t_te_values < thre)
        mem_inf_acc = 0.5 * (t_tr_mem / (len(t_tr_values) + 0.0) + t_te_non_mem / (len(t_te_values) + 0.0))
        info = 'MIA via {n} (general threshold): the attack acc is {acc:.3f}'.format(n=v_name, acc=mem_inf_acc)
        print(info)
        return info, mem_inf_acc

    def _mem_inf_roc(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
        """MIA AUC given the feature values (no need to threshold)"""
        labels = np.concatenate((np.zeros((len(t_te_values),)), np.ones((len(t_tr_values),))))
        results = np.concatenate((t_te_values, t_tr_values))
        auc = metrics.roc_auc_score(labels, results)
        ap = metrics.average_precision_score(labels, results)
        info = 'MIA via {n}: the attack auc is {auc:.3f}, ap is {ap:.3f}'.format(n=v_name, auc=auc, ap=ap)
        print(info)
        return info, auc

    def compute_attack_acc(self, method_names, score_signs, if_per_class_thres=False):
        """Compute Attack accuracy"""
        if if_per_class_thres:
            mem_inf_thre_func = self._mem_inf_thre_perclass
            loginfo = 'per class threshold\n'
        else:
            mem_inf_thre_func = self._mem_inf_thre
            loginfo = 'overall threshold\n'
        results = []
        for i in range(self.num_methods):
            if score_signs[i] == '+':
                info, result = mem_inf_thre_func(method_names[i], self.s_tr_scores[i], self.s_te_scores[i],
                                                 self.t_tr_scores[i], self.t_te_scores[i])
                loginfo += info + '\n'
                results.append(result)

            else:
                info, result = mem_inf_thre_func(method_names[i], -self.s_tr_scores[i], -self.s_te_scores[i],
                                                 -self.t_tr_scores[i], -self.t_te_scores[i])
                loginfo += info + '\n'
                results.append(result)
        return loginfo, method_names, results

    def compute_attack_auc(self, method_names, score_signs):
        """Compute attack AUC (and AP)"""
        loginfo = ''
        results = []
        for i in range(self.num_methods):
            if score_signs[i] == '+':
                info, result = self._mem_inf_roc(method_names[i], self.s_tr_scores[i], self.s_te_scores[i],
                                                 self.t_tr_scores[i], self.t_te_scores[i])
                loginfo += info + '\n'
                results.append(result)
            else:
                info, result = self._mem_inf_roc(method_names[i], -self.s_tr_scores[i], -self.s_te_scores[i],
                                                 -self.t_tr_scores[i], -self.t_te_scores[i])
                loginfo += info + '\n'
                results.append(result)
        return loginfo, method_names, results

class Benchmark_Blackbox(Benchmark):
    def compute_bb_scores(self):
        self.s_tr_outputs, self.s_tr_loss = self.s_tr_scores
        self.s_te_outputs, self.s_te_loss = self.s_te_scores
        self.t_tr_outputs, self.t_tr_loss = self.t_tr_scores
        self.t_te_outputs, self.t_te_loss = self.t_te_scores

        # whether the prediction is correct [num_samples,]
        # pdb.set_trace()
        self.s_tr_corr = (np.argmax(self.s_tr_outputs, axis=1) == self.s_tr_labels).astype(int)
        # print(self.s_tr_corr)
        # pdb.set_trace()
        self.s_te_corr = (np.argmax(self.s_te_outputs, axis=1) == self.s_te_labels).astype(int)
        self.t_tr_corr = (np.argmax(self.t_tr_outputs, axis=1) == self.t_tr_labels).astype(int)
        self.t_te_corr = (np.argmax(self.t_te_outputs, axis=1) == self.t_te_labels).astype(int)

        # confidence prediction of the ground-truth class [num_samples,]
        self.s_tr_conf = np.array([self.s_tr_outputs[i, self.s_tr_labels[i]] for i in range(len(self.s_tr_labels))])
        self.s_te_conf = np.array([self.s_te_outputs[i, self.s_te_labels[i]] for i in range(len(self.s_te_labels))])
        self.t_tr_conf = np.array([self.t_tr_outputs[i, self.t_tr_labels[i]] for i in range(len(self.t_tr_labels))])
        self.t_te_conf = np.array([self.t_te_outputs[i, self.t_te_labels[i]] for i in range(len(self.t_te_labels))])

        # entropy of the prediction [num_samples,]
        self.s_tr_entr = self._entr_comp(self.s_tr_outputs)
        self.s_te_entr = self._entr_comp(self.s_te_outputs)
        self.t_tr_entr = self._entr_comp(self.t_tr_outputs)
        self.t_te_entr = self._entr_comp(self.t_te_outputs)

        # proposed modified entropy [num_samples,]
        self.s_tr_m_entr = self._m_entr_comp(self.s_tr_outputs, self.s_tr_labels)
        self.s_te_m_entr = self._m_entr_comp(self.s_te_outputs, self.s_te_labels)
        self.t_tr_m_entr = self._m_entr_comp(self.t_tr_outputs, self.t_tr_labels)
        self.t_te_m_entr = self._m_entr_comp(self.t_te_outputs, self.t_te_labels)

    def _log_value(self, probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))

    def _entr_comp(self, probs):
        """compute the entropy of the prediction"""
        return np.sum(np.multiply(probs, self._log_value(probs)), axis=1)

    def _m_entr_comp(self, probs, true_labels):
        """-(1-f(x)_y) log(f(x)_y) - \sum_i f(x)_i log(1-f(x)_i)"""

        log_probs = self._log_value(probs)
        reverse_probs = 1 - probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(true_labels.size), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)

    def _mem_inf_via_corr(self):
        """perform membership inference attack based on whether the input is correctly classified or not"""
        t_tr_acc = np.sum(self.t_tr_corr) / (len(self.t_tr_corr) + 0.0)
        # print(np.sum(self.t_tr_corr))
        # print(len(self.t_tr_corr))
        # print(t_tr_acc)
        # pdb.set_trace()
        t_te_acc = np.sum(self.t_te_corr) / (len(self.t_te_corr) + 0.0)
        mem_inf_acc = 0.5 * (t_tr_acc + 1 - t_te_acc)
        info = 'MIA via correctness, the attack acc is {acc1:.3f}, with train acc {acc2:.3f} and test acc {acc3:.3f}'.format(
            acc1=mem_inf_acc, acc2=t_tr_acc, acc3=t_te_acc)
        print(info)
        return info, mem_inf_acc

    def compute_attack_acc(self, method_names=[], all_methods=True, if_per_class_thres=True):
        """Compute Attack accuracy"""
        if if_per_class_thres:
            mem_inf_thre_func = self._mem_inf_thre_perclass
            loginfo = 'per class threshold\n'
        else:
            mem_inf_thre_func = self._mem_inf_thre
            loginfo = 'overall threshold\n'
        results = []
        methods = []
        if (all_methods) or ('correctness' in method_names):
            info, result = self._mem_inf_via_corr()
            loginfo += info + '\n'
        if (all_methods) or ('confidence' in method_names):
            info, result = mem_inf_thre_func('confidence', self.s_tr_conf, self.s_te_conf,
                                             self.t_tr_conf, self.t_te_conf)
            loginfo += info + '\n'
            results.append(result)
            methods.append('confidence ACC')
        if (all_methods) or ('entropy' in method_names):
            info, result = mem_inf_thre_func('entropy', -self.s_tr_entr, -self.s_te_entr,
                                             -self.t_tr_entr, -self.t_te_entr)
            loginfo += info + '\n'
            results.append(result)
            methods.append('entropy ACC')
        if (all_methods) or ('modified entropy' in method_names):
            info, result = mem_inf_thre_func('modified entropy', -self.s_tr_m_entr, -self.s_te_m_entr,
                                             -self.t_tr_m_entr, -self.t_te_m_entr)
            loginfo += info + '\n'
            results.append(result)
            methods.append('modified entropy ACC')
        if (all_methods) or ('loss' in method_names):
            info, result = mem_inf_thre_func('loss', -self.s_tr_loss, -self.s_te_loss,
                                             -self.t_tr_loss, -self.t_te_loss)
            loginfo += info + '\n'
            results.append(result)
            methods.append('loss ACC')
        return loginfo, methods, results

    def compute_attack_auc(self, method_names=[], all_methods=True):
        """Compute all attack AUC"""
        loginfo = ''
        methods = []
        results = []
        if (all_methods) or ('confidence' in method_names):
            info, result = self._mem_inf_roc('confidence', self.s_tr_conf, self.s_te_conf,
                                             self.t_tr_conf, self.t_te_conf)
            loginfo += info + '\n'
            results.append(result)
            methods.append('confidence AUC')
        if (all_methods) or ('entropy' in method_names):
            info, result = self._mem_inf_roc('entropy', -self.s_tr_entr, -self.s_te_entr,
                                             -self.t_tr_entr, -self.t_te_entr)
            loginfo += info + '\n'
            results.append(result)
            methods.append('entropy AUC')
        if (all_methods) or ('modified entropy' in method_names):
            info, result = self._mem_inf_roc('modified entropy', -self.s_tr_m_entr, -self.s_te_m_entr,
                                             -self.t_tr_m_entr, -self.t_te_m_entr)
            loginfo += info + '\n'
            results.append(result)
            methods.append('modified entropy AUC')
        if (all_methods) or ('loss' in method_names):
            info, result = self._mem_inf_roc('loss', -self.s_tr_loss, -self.s_te_loss,
                                             -self.t_tr_loss, -self.t_te_loss)
            loginfo += info + '\n'
            results.append(result)
            methods.append('modified entropy AUC')
        return loginfo, methods, results

class BaseAttacker(object):
    def __init__(self, args, save_dir,member,nonmember):
        self.args = args
        self.save_dir = save_dir
        self.member=member
        self.nonmember=nonmember
        self.set_cuda_device()
        self.set_seed()
        self.set_dataloader()
        self.set_criterion()
        self.load_models()
        

    def set_cuda_device(self):
        """The function to set CUDA device."""
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.device = torch.device(f"cuda:{super_args.gpu_idx}" if self.use_cuda else "cpu")

    def set_criterion(self):
        self.crossentropy = nn.CrossEntropyLoss()
        self.crossentropy_noreduce = nn.CrossEntropyLoss(reduction='none')
        self.softmax = nn.Softmax(dim=1)

    def set_seed(self):
        """Set random seed"""
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        if self.use_cuda:
            torch.cuda.manual_seed_all(self.args.seed)

    @abc.abstractmethod
    def set_dataloader(self):
        """The function to set the dataloader"""
        self.data_root = None
        self.dataset = None
        self.num_classes = None
        self.dataset_size = None
        self.transform_train = None
        self.transform_test = None
        self.target_trainloader = None
        self.target_testloader = None
        self.shadow_trainloader = None
        self.shadow_testloader = None
        self.loader_dict = None

    def load_models(self):
        target_model=models.__dict__[self.args.model](num_classes=self.args.num_classes)
        checkpoint = torch.load(os.path.join(self.args.target_path, 'checkpoint.pkl'))
        new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}# target_model.load_state_dict(checkpoint['model_state_dict'])
        # new_state_dict = {k.replace('_', ''): v for k, v in new_state_dict.items()}# target_model.load_state_dict(checkpoint['model_state_dict'])
        new_state_dict = {k.replace('fc.1.weight', 'fc.weight'): v for k, v in new_state_dict.items()}
        new_state_dict = {k.replace('fc.1.bias', 'fc.bias'): v for k, v in new_state_dict.items()}
        target_model.load_state_dict(new_state_dict)
        self.target_model = target_model
        # self.target_model=torch.load(os.path.join(self.args.target_path, 'model.pt'))
        print('Loading target model from ', self.args.target_path)
        shadow_model=models.__dict__[self.args.model](num_classes=self.args.num_classes)
        checkpoint = torch.load(os.path.join(self.args.shadow_path, 'checkpoint.pkl'))
        print(list(checkpoint.keys()))
        if 'module.' in list(checkpoint['model_state_dict'].keys())[0]:
            new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
            shadow_model.load_state_dict(new_state_dict)
        else:
            shadow_model.load_state_dict(checkpoint['model_state_dict']) # target_path = os.path.join(self.args.target_path, 'checkpoint.pkl')
        # shadow_path = os.path.join(self.args.shadow_path, 'model.pt')
        # shadow_model = torch.load(shadow_path).to(self.device)
        self.shadow_model = shadow_model
        
        print('Loading shadow model from ', self.args.shadow_path)
        self.model_dict = {'t': self.target_model, 's': self.shadow_model}

    def run_blackbox_attacks(self):
        """Run black-box attacks """
        # print(self.args.attack_member_loader)
        # 使用存储的参数重新创建数据集对象
        # subset_dataset = CustomDataset(self.args.subset_data, self.args.subset_labels)
        # sampled_dataset = torch.utils.data.Subset(self.target_testloader.dataset, self.args.sampled_indices)

        # 使用重新创建的数据集对象创建新的数据加载器
        # subset_loader = torch.utils.data.DataLoader(subset_dataset, batch_size=64, shuffle=True)
        # sampled_dataloader = torch.utils.data.DataLoader(sampled_dataset, batch_size=64, shuffle=True, num_workers=4)

        # pdb.set_trace()
        t_logits_pos, t_posteriors_pos, t_losses_pos, t_labels_pos = self.get_blackbox_statistics(
            self.target_trainloader, self.target_model)
        # pdb.set_trace()
        t_logits_neg, t_posteriors_neg, t_losses_neg, t_labels_neg = self.get_blackbox_statistics(
            self.target_testloader, self.target_model)
        s_logits_pos, s_posteriors_pos, s_losses_pos, s_labels_pos = self.get_blackbox_statistics(
            self.shadow_trainloader, self.shadow_model)
        s_logits_neg, s_posteriors_neg, s_losses_neg, s_labels_neg = self.get_blackbox_statistics(
            self.shadow_testloader, self.shadow_model)

        ## metric_based attacks
        bb_benchmark = Benchmark_Blackbox(shadow_train_scores=[s_posteriors_pos, s_losses_pos],
                                          shadow_test_scores=[s_posteriors_neg, s_losses_neg],
                                          target_train_scores=[t_posteriors_pos, t_losses_pos],
                                          target_test_scores=[t_posteriors_neg, t_losses_neg])
        bb_benchmark.load_labels(s_labels_pos, s_labels_neg, t_labels_pos, t_labels_neg, self.num_classes)
        # pdb.set_trace()
        bb_benchmark.compute_bb_scores()

        ## nn attack
        # info, names, results = self.run_nn_attack(s_logits_pos, s_logits_neg, t_logits_pos, t_logits_neg)

        ### Save results
        # log_info = info
        log_info='null'
        # all_names = [names]
        all_names=[['null']]
        # all_results = [results]
        all_results=[['null']]
        info, names, results = bb_benchmark.compute_attack_acc()
        all_names.append(names)
        all_results.append(results)
        log_info += info
        info, names, results = bb_benchmark.compute_attack_auc()
        all_names.append(names)
        all_results.append(results)
        log_info += info
        self.bb_loginfo = log_info
        self.bb_results = np.concatenate(all_results)
        self.bb_names = np.concatenate(all_names)

    def run_whitebox_attacks(self):
        """Run white-box attacks"""

        def run_case(partition, subset, grad_type):
            if partition == 's':
                model_dir = self.args.shadow_path
            else:
                assert partition == 't'
                model_dir = self.args.target_path
            filename = f'{partition}_{subset}_{grad_type}'
            loadername = f'{partition}_{subset}'
            path = os.path.join(model_dir, 'attack', filename + '.pkl')

            # if os.path.exists(path):
            #     stat = unpickle(path)
            # else:
            if grad_type == 'x':
                stat = self.gradient_based_attack_wrt_x(self.loader_dict[loadername], self.model_dict[partition])
            else:
                assert grad_type == 'w'
                stat = self.gradient_based_attack_wrt_w(self.loader_dict[loadername], self.model_dict[partition])
            savepickle(stat, path)
            return stat

        ### Grad w.r.t. x
        s_pos_x = run_case('s', 'pos', 'x')
        s_neg_x = run_case('s', 'neg', 'x')
        t_pos_x = run_case('t', 'pos', 'x')
        t_neg_x = run_case('t', 'neg', 'x')

        ### Grad w.r.t. w
        s_pos_w = run_case('s', 'pos', 'w')
        s_neg_w = run_case('s', 'neg', 'w')
        t_pos_w = run_case('t', 'pos', 'w')
        t_neg_w = run_case('t', 'neg', 'w')

        ### Save results
        all_names = []
        all_results = []
        log_info = ''
        wb_benchmark = Benchmark(shadow_train_scores=[s_pos_x['l1'], s_pos_x['l2'], s_pos_w['l1'], s_pos_w['l2']],
                                 shadow_test_scores=[s_neg_x['l1'], s_neg_x['l2'], s_neg_w['l1'], s_neg_w['l2']],
                                 target_train_scores=[t_pos_x['l1'], t_pos_x['l2'], t_pos_w['l1'], t_pos_w['l2']],
                                 target_test_scores=[t_neg_x['l1'], t_neg_x['l2'], t_neg_w['l1'], t_neg_w['l2']])
        info, names, results = wb_benchmark.compute_attack_acc(
            method_names=['grad_wrt_x_l1 ACC', 'grad_wrt_x_l2 ACC', 'grad_wrt_w_l1 ACC', 'grad_wrt_w_l2 ACC'],
            score_signs=['-', '-', '-', '-'])
        all_names.append(names)
        all_results.append(results)
        log_info += info

        info, names, results = wb_benchmark.compute_attack_auc(
            method_names=['grad_wrt_x_l1 AUC', 'grad_wrt_x_l2 AUC', 'grad_wrt_w_l1 AUC', 'grad_wrt_w_l2 AUC'],
            score_signs=['-', '-', '-', '-'])
        all_names.append(names)
        all_results.append(results)
        log_info += info
        # print(log_info)
        self.wb_loginfo = log_info
        self.wb_results = np.concatenate(all_results)
        self.wb_names = np.concatenate(all_names)

    def save_results(self):
        """Save to attack_log.txt file and .csv file"""
        with open(os.path.join(self.save_dir, 'attack_log.txt'), 'a+') as f:
            log_info = '=' * 100 + '\n' + self.args.target_path + '\n' + self.wb_loginfo + '\n' + self.bb_loginfo
            f.writelines(log_info)
        write_csv(os.path.join(self.save_dir, 'attack_log.csv'),
                  self.args.target_path.split('/')[-1],
                  np.concatenate([self.bb_results, self.wb_results]),
                  np.concatenate([self.bb_names, self.wb_names]))

    def gradient_based_attack_wrt_x(self, dataloader, model):
        """Gradient w.r.t. input"""
        model.eval()

        ## store results
        names = ['l1', 'l2', 'Min', 'Max', 'Mean', 'Skewness', 'Kurtosis']
        all_stats = {}
        for name in names:
            all_stats[name] = []

        ## iterate over batches
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader)):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            ## iterate over samples within a batch
            for input, target in zip(inputs, targets):
                input = torch.unsqueeze(input, 0)
                input.requires_grad = True
                output = model(input)
                loss = self.crossentropy(output, torch.unsqueeze(target, 0))
                model.zero_grad()
                loss.backward()

                ## get gradients
                gradient = input.grad.view(-1).cpu().numpy()

                ## get statistics
                stats = compute_norm_metrics(gradient)
                for i, stat in enumerate(stats):
                    all_stats[names[i]].append(stat)

        for name in names:
            all_stats[name] = np.array(all_stats[name])
        return all_stats

    def gradient_based_attack_wrt_w(self, dataloader, model):
        """Gradient w.r.t. weights"""
        model.eval()

        ## store results
        names = ['l1', 'l2', 'Min', 'Max', 'Mean', 'Skewness', 'Kurtosis']
        all_stats = {}
        for name in names:
            all_stats[name] = []

        ## iterate over batches
        for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader)):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            ## iterate over samples within a batch
            for input, target in zip(inputs, targets):
                input = torch.unsqueeze(input, 0)
                output = model(input)
                loss = self.crossentropy(output, torch.unsqueeze(target, 0))
                model.zero_grad()
                loss.backward()

                ## get gradients
                grads_onesample = []
                for param in model.parameters():
                    grads_onesample.append(param.grad.view(-1))
                gradient = torch.cat(grads_onesample)
                gradient = gradient.cpu().numpy()

                ## get statistics
                stats = compute_norm_metrics(gradient)
                for i, stat in enumerate(stats):
                    all_stats[names[i]].append(stat)

        for name in names:
            all_stats[name] = np.array(all_stats[name])
        return all_stats

    def get_blackbox_statistics(self, dataloader, model):
        """Compute the blackbox statistics (for blackbox attacks)"""
        model.to(self.device)
        model.eval()

        logits = []
        labels = []
        losses = []
        posteriors = []

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                # print(item)
                inputs, targets = inputs.to(torch.float).to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = self.crossentropy_noreduce(outputs, targets)
                posterior = self.softmax(outputs)
                logits.extend(outputs.cpu().numpy())
                posteriors.extend(posterior.cpu().numpy())
                labels.append(targets.cpu().numpy())
                losses.append(loss.cpu().numpy())
        logits = np.vstack(logits)
        posteriors = np.vstack(posteriors)
        labels = np.concatenate(labels)
        losses = np.concatenate(losses)
        return logits, posteriors, losses, labels

    def run_nn_attack(self, s_logits_pos, s_logits_neg, t_logits_pos, t_logits_neg, if_load_checkpoint=True):
        checkpoint_dir = os.path.join(self.args.shadow_path, 'attack', 'nn')
        mkdir(checkpoint_dir)
        trainer = NNAttackTrainer(self.args, checkpoint_dir)
        trainer.set_loader(s_logits_pos, s_logits_neg, t_logits_pos, t_logits_neg)
        if os.path.exists(os.path.join(checkpoint_dir, 'attack_model.pt')) and if_load_checkpoint:
            checkpoint = torch.load(os.path.join(self.args.shadow_path, 'checkpoint.pkl'))
            print(checkpoint.keys())
            if 'module.' in list(checkpoint['attack_model_state_dict'].keys())[0]:
                new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['attack_model_state_dict'].items()}
                attack_model.load_state_dict(new_state_dict)
            else:
                attack_model.load_state_dict(checkpoint['attack_model_state_dict']) 
            # target_path = os.path.join(self.args.target_path, 'checkpoint.pkl')
            # attack_model = torch.load(os.path.join(checkpoint_dir, 'attack_model.pt')).to(self.device)
            print('Load NN attack from checkpoint_dir')
        else:
            max_epoch = 20
            lr = 0.001
            attack_model = NNAttack(self.num_classes)
            optimizer = optim.Adam(attack_model.parameters(), lr=lr)
            logger = trainer.logger
            print('Train NN attack')
            for _ in range(max_epoch):
                train_loss, train_acc = trainer.train(attack_model, optimizer)
                test_loss, test_acc, _ = trainer.test(attack_model)
                logger.append([train_loss, test_loss, train_acc, test_acc])
            torch.save(attack_model, os.path.join(checkpoint_dir, 'attack_model.pt'))
        _, attack_acc, attack_auc = trainer.test(attack_model)
        info = 'MIA via NN : the attack acc is {acc:.3f} \n'.format(acc=attack_acc / 100)
        info += 'MIA via NN : the attack auc is {auc:.3f} \n'.format(auc=attack_auc)
        return info, ['NN ACC', 'NN AUC'], [attack_acc / 100, attack_auc]

def compute_norm_metrics(gradient):
    """Compute the metrics"""
    l1 = np.linalg.norm(gradient, ord=1)
    l2 = np.linalg.norm(gradient)
    Min = np.linalg.norm(gradient, ord=-np.inf)  ## min(abs(x))
    Max = np.linalg.norm(gradient, ord=np.inf)  ## max(abs(x))
    Mean = np.average(gradient)
    Skewness = skew(gradient)
    Kurtosis = kurtosis(gradient)
    return [l1, l2, Min, Max, Mean, Skewness, Kurtosis]

class NNAttack(nn.Module):
    """NN attack model"""

    def __init__(self, input_dim, output_dim=1, hiddens=[100]):
        super(NNAttack, self).__init__()
        self.layers = []
        for i in range(len(hiddens)):
            if i == 0:
                layer = nn.Linear(input_dim, hiddens[i])
            else:
                layer = nn.Linear(hiddens[i - 1], hiddens[i])
            self.layers.append(layer)
        self.last_layer = nn.Linear(hiddens[-1], output_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = self.relu(layer(output))
        output = self.last_layer(output)
        return output

class NNAttackTrainer(BaseTrainer):
    """Trainer for the NN attack"""

    @staticmethod
    def construct_dataloader(stat_pos, stat_neg):
        """Construct dataloader from statistics"""
        attack_data = np.concatenate([stat_neg, stat_pos], axis=0)
        attack_data = np.sort(attack_data, axis=1)
        attack_targets = np.concatenate([np.zeros(len(stat_neg)), np.ones(len(stat_pos))])
        attack_targets = attack_targets.astype(np.int)
        attack_indices = np.arange(len(attack_data))
        np.random.shuffle(attack_indices)
        attack_data = attack_data[attack_indices]
        attack_targets = attack_targets[attack_indices]
        tensor_x = torch.from_numpy(attack_data)
        tensor_y = torch.from_numpy(attack_targets)
        tensor_y = tensor_y.unsqueeze(-1).type(torch.FloatTensor)
        attack_dataset = data.TensorDataset(tensor_x, tensor_y)
        attack_loader = data.DataLoader(attack_dataset, batch_size=256, shuffle=True,generator=torch.Generator(device = f"cuda:{super_args.gpu_idx}"))
        return attack_loader

    def set_loader(self, s_logits_pos, s_logits_neg, t_logits_pos, t_logits_neg):
        """Set the training and testing dataloader"""
        self.trainloader = self.construct_dataloader(s_logits_pos, s_logits_neg)
        self.testloader = self.construct_dataloader(t_logits_pos, t_logits_neg)

    def set_criterion(self):
        """Set the training criterion (BCE by default)"""
        self.criterion = nn.BCELoss()

    def train(self, model, optimizer):
        """Train"""
        model.train()
        criterion = self.criterion
        losses = AverageMeter()
        top1 = AverageMeter()
        batch_time = AverageMeter()
        dataload_time = AverageMeter()
        time_stamp = time.time()

        bar = Bar('Processing', max=len(self.trainloader))
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            ### Record the data loading time
            dataload_time.update(time.time() - time_stamp)

            ### Output
            outputs = model(inputs)
            if outputs.shape[-1] == 1:
                outputs = outputs.view(-1)
                targets = targets.view(-1)
                outputs = nn.Sigmoid()(outputs)
                prec1 = accuracy_binary(outputs.data, targets.data)
            else:
                prec1 = accuracy(outputs.data, targets.data)[0]
            loss = criterion(outputs, targets)

            ### Record accuracy and loss
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            ### Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### Record the total time for processing the batch
            batch_time.update(time.time() - time_stamp)
            time_stamp = time.time()

            ### Progress bar
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                batch=batch_idx + 1,
                size=len(self.trainloader),
                data=dataload_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg
            )
            bar.next()

        bar.finish()
        return (losses.avg, top1.avg)

    def test(self, model):
        """Test"""
        model.eval()
        criterion = self.criterion
        losses = AverageMeter()
        top1 = AverageMeter()
        batch_time = AverageMeter()
        dataload_time = AverageMeter()
        time_stamp = time.time()
        ytest = []
        ypred_score = []

        bar = Bar('Processing', max=len(self.testloader))
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                ### Record the data loading time
                dataload_time.update(time.time() - time_stamp)

                ### Forward
                outputs = model(inputs)
                if outputs.shape[-1] == 1:
                    outputs = outputs.view(-1)
                    targets = targets.view(-1)
                    outputs = nn.Sigmoid()(outputs)
                    prec1 = accuracy_binary(outputs.data, targets.data)
                    ytest.append(targets.cpu().numpy())
                    ypred_score.append(outputs.cpu().numpy())
                else:
                    prec1 = accuracy(outputs.data, targets.data)[0]
                    ytest.append(targets.cpu().numpy())
                    outputs = nn.Softmax(dim=1)(outputs)
                    ypred_score.append(outputs.cpu().numpy()[:, 1])

                ### Evaluate
                loss = criterion(outputs, targets)
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))

                ### Record the total time for processing the batch
                batch_time.update(time.time() - time_stamp)
                time_stamp = time.time()

                ### Progress bar
                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(self.testloader),
                    data=dataload_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                )
                bar.next()
            bar.finish()
        ytest = np.concatenate(ytest)
        ypred_score = np.concatenate(ypred_score)
        auc, ap, f1, pos_num, frac = metrics_binary(ytest, ypred_score)
        return (losses.avg, top1.avg, auc)

    def set_logger(self):
        """Set up logger"""
        title = self.args.dataset
        self.start_epoch = 0
        logger = Logger(os.path.join(self.save_dir, 'log.txt'), title=title)
        logger.set_names(['Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'])
        self.logger = logger

class Attacker(BaseAttacker):
    def __init__(self, args, save_dir, member, nonmember):
        super().__init__(args, save_dir, member, nonmember)

    def set_dataloader(self):
        """The function to set the dataset parameters"""
        self.data_root = '/data/home/huqiang/DeepCore/RelaxLoss/data'
        if self.args.dataset == 'CIFAR10':
            self.dataset = CIFAR10
            self.num_classes = 10
            self.dataset_size = 60000
        elif self.args.dataset == 'CIFAR100':
            self.dataset = CIFAR100
            self.num_classes = 100
            self.dataset_size = 60000
        transform_train = transform_test = transforms.Compose([transforms.ToTensor(),
                                                               transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                    (0.2023, 0.1994, 0.2010))])
        self.transform_train = transform_train
        self.transform_test = transform_test

        ## Set the partition and datloader
        # indices = np.load(os.path.join(self.args.target_path, 'full_idx.npy'))
        indices = np.load('/data/home/huqiang/DeepCore/RelaxLoss/results/CIFAR10/resnet20/advreg/full_idx.npy')
        if os.path.exists(os.path.join(self.args.shadow_path, 'full_idx.npy')):
            shadow_indices = np.load(os.path.join(self.args.shadow_path, 'full_idx.npy'))
            # print("indices")
            # print(indices)
            # print(shadow_indices)
            assert np.array_equiv(indices, shadow_indices)
        self.partition = Partition(dataset_size=self.dataset_size, indices=indices)
        target_train_idx, target_test_idx = self.partition.get_target_indices()
        shadow_train_idx, shadow_test_idx = self.partition.get_shadow_indices()

        indices=np.load('/data/home/huqiang/DeepCore/RelaxLoss/results/CIFAR10/resnet20/advreg/full_idx.npy')
    
        target_trainset = self.dataset(root=self.data_root, indices=indices[member_indices],
                                       download=True, transform=self.transform_train)##前12000中的0.1
        target_testset = self.dataset(root=self.data_root, indices=indices[nonmember_indices],
                                      download=True, transform=self.transform_test)
        shadow_trainset = self.dataset(root=self.data_root, indices=shadow_train_idx,
                                       download=True, transform=self.transform_train)
        shadow_testset = self.dataset(root=self.data_root, indices=shadow_test_idx,
                                      download=True, transform=self.transform_test)
        self.target_trainloader = torch.utils.data.DataLoader(target_trainset, batch_size=self.args.test_batchsize, shuffle=False)
        self.target_testloader = torch.utils.data.DataLoader(target_testset, batch_size=self.args.test_batchsize, shuffle=False)
        self.shadow_trainloader = torch.utils.data.DataLoader(shadow_trainset, batch_size=self.args.test_batchsize, shuffle=False)
        self.shadow_testloader = torch.utils.data.DataLoader(shadow_testset, batch_size=self.args.test_batchsize, shuffle=False)
        self.loader_dict = {'s_pos': self.shadow_trainloader, 's_neg': self.shadow_testloader,
                            't_pos': self.target_trainloader, 't_neg': self.target_testloader}

def check_args(parser):
    '''check and store the arguments as well as set up the save_dir'''
    ## set up save_dir
    args = parser.parse_args()
    save_dir = os.path.join(args.target_path, 'attack')
    mkdir(save_dir)
    mkdir(os.path.join(args.shadow_path, 'attack'))

    ## load configs and store the parameters
    preload_configs = load_yaml(os.path.join(args.target_path, 'params.yml'))
    parser.set_defaults(**preload_configs)
    args = parser.parse_args()
    write_yaml(vars(args), os.path.join(save_dir, 'params.yml'))
    return args, save_dir



# # Function to divide the training set into n subsets based on importance scores
# def divide_into_subsets(importance_scores, n, type='size'):
#     if type=='len':
#         min_score = np.min(importance_scores)
#         max_score = np.max(importance_scores)
#         intervals = np.linspace(min_score, max_score, n + 1)
#         subset_indices = np.digitize(importance_scores, intervals) - 1
#     else:
#         # Sort the importance scores and find the indices that would sort the array
#         sorted_indices = np.argsort(importance_scores)

#         # Calculate the number of data points in each subset
#         subset_size = len(importance_scores) // n

#         # Create an array to hold the subset indices for each data point
#         subset_indices = np.zeros(len(importance_scores), dtype=np.int64)

#         np.save(f"/data/home/huqiang/DeepCore/save/{super_args.coreset}/sorted_indices.npy",sorted_indices)

#         # Assign data points to subsets based on sorted indices
#         for i in range(n):
#             start_idx = i * subset_size
#             end_idx = (i + 1) * subset_size
#             np.save(f"/data/home/huqiang/DeepCore/save/{super_args.coreset}/sorted_indices{i}_10.npy",sorted_indices[start_idx:end_idx])
#             subset_indices[sorted_indices[start_idx:end_idx]] = i

#         # For any remaining data points, assign them to the last subset
#         print(subset_indices)
#         subset_indices[sorted_indices[end_idx:]] = n - 1

#     return subset_indices

def get_subset_midpoints(importance_scores, subset_indices, n_subsets,type='len'):
    if type=='size':
        subset_midpoints = []
        for i in range(n_subsets):
            subset_mask = subset_indices == i
            subset_scores = importance_scores[subset_mask]
            if len(subset_scores) > 0:
                midpoint = np.mean(subset_scores)
            else:
                midpoint = 0.0  # Handle empty subsets if necessary
            subset_midpoints.append(midpoint)
    else:
        # Calculate the range of importance scores
        min_score = np.min(importance_scores)
        max_score = np.max(importance_scores)

        # Calculate the step size for equally spaced intervals
        step_size = (max_score - min_score) / n_subsets

        # Calculate the midpoints of the importance score intervals
        subset_midpoints = np.linspace(min_score + step_size / 2, max_score - step_size / 2, n_subsets)

    return np.array(subset_midpoints)

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_item = self.data[index]
        label_item = self.labels[index]
        return data_item, label_item



def run_defense(dataset_prefix, save_root, args, method):
    model_flag = (dataset_prefix == 'cifar')
    if method == 'distillation':
        ## train teacher model
        teacher_path = os.path.join(save_root, 'vanilla', f'seed{args.seed}')
        if not os.path.exists(os.path.join(teacher_path, 'model.pt')):
            command = f'python {dataset_prefix}/defense/vanilla.py -name seed{args.seed} -s {args.seed} --dataset {args.dataset}'
            command += f' --model {args.model}' if model_flag else ''
            os.system(command)

        ## train student model
        command = f'python {dataset_prefix}/defense/{method}.py -name seed{args.seed} -s {args.seed} --dataset {args.dataset} -teacher {teacher_path}'
        command += f' --model {args.model}' if model_flag else ''
        os.system(command)

    else:
        command = f'python {dataset_prefix}/defense/{method}.py -name seed{args.seed} -s {args.seed} --dataset {args.dataset}'
        command += f' --model {args.model}' if model_flag else ''
        os.system(command)

def run_shadow(dataset_prefix, save_root, args, method):
    model_flag = (dataset_prefix == 'cifar')
    if method == 'distillation':
        ## train teacher model
        teacher_path = os.path.join(save_root, 'vanilla', f'seed{args.seed}', 'shadow')
        if not os.path.exists(os.path.join(teacher_path, 'model.pt')):
            command = f'python RelaxLoss/source/{dataset_prefix}/defense/vanilla.py -name seed{args.seed} -s {args.seed} --dataset {args.dataset} --partition shadow'
            command += f' --model {args.model}' if model_flag else ''
            os.system(command)

        ## train student model
        command = f'python RelaxLoss/source/{dataset_prefix}/defense/{method}.py -name seed{args.seed} -s {args.seed} ' \
                  f'--dataset {args.dataset} -teacher {teacher_path} --partition shadow'
        command += f' --model {args.model}' if model_flag else ''
        os.system(command)

    else:
        command = f'python RelaxLoss/source/{dataset_prefix}/defense/{method}.py -name seed{args.seed} -s {args.seed} --dataset {args.dataset} --partition shadow'
        command += f' --model {args.model}' if model_flag else ''
        os.system(command)

def run_attack(dataset_prefix, target, shadow, member,nonmember):
    save_dir = os.path.join(target, 'attack')
    mkdir(save_dir)
    mkdir(os.path.join(shadow, 'attack'))

    ## load configs and store the parameters
    preload_configs = load_yaml(os.path.join(target, 'params.yml'))
    # parser.set_defaults(**preload_configs)
    # args = parser.parse_args()
    write_yaml(vars(args), os.path.join(save_dir, 'params.yml'))

    args.target_path=target
    args.shadow_path=shadow

    attacker = Attacker(args, save_dir ,member,nonmember)
    # attacker.target_trainloader
    # pdb.set_trace()
    attacker.run_blackbox_attacks()
    attacker.run_whitebox_attacks()
    attacker.save_results()
    # command = (
    # f'python RelaxLoss/source/{dataset_prefix}/run_attacks.py '
    # f'-target "{target}" -shadow "{shadow}" --subset_data "{subset_data}" --subset_labels "{subset_labels}" --sampled_indices "{sampled_indices}"'
    # )

    # os.system(command)
    # os.system(f'python RelaxLoss/source/{dataset_prefix}/run_attacks.py -target {target} -shadow {shadow} -member {member} -nonmember {nonmember}')

def attack(args,member=None,nonmember=None):
    FILE_DIR = os.path.dirname(os.path.abspath("/data/home/huqiang/DeepCore/mia_scoring.ipynb"))
    SAVE_ROOT_IMAGE = os.path.join(FILE_DIR, 'RelaxLoss/results/%s/%s/')
    SAVE_ROOT_GENERAL = os.path.join(FILE_DIR, 'RelaxLoss/results/%s/')

    # args = parse_arguments()
    if args.dataset in ['CIFAR10', 'CIFAR100']:
        dataset_prefix = 'cifar'
        save_root = SAVE_ROOT_IMAGE % (args.dataset, args.model)
    elif args.dataset in ['Texas', 'Purchase']:
        dataset_prefix = 'nonimage'
        save_root = SAVE_ROOT_GENERAL % args.dataset
    else:
        raise NotImplementedError

    ## train shadow model (for attack)
    base_shadow_path = os.path.join(save_root, 'vanilla', f'seed{args.seed}', 'shadow')
    if not os.path.exists(os.path.join(base_shadow_path, 'model.pt')):
        run_shadow(dataset_prefix, save_root, args, 'vanilla')

    ## run attack
    # target_path = os.path.join(save_root, args.method, f'seed{args.seed}')
    target_path=f"/data/home/huqiang/DeepCore/RelaxLoss/results/CIFAR10/resnet20/{super_args.defense}/{super_args.model_name}"
    if super_args.defense == 'early_stopping':
        all_targets = [p for p in os.listdir(target_path) if os.path.isdir(os.path.join(target_path,p)) and 'ep' in p]
        all_targets = [os.path.join(target_path,p) for p in all_targets]
        for target_path in all_targets:
            run_attack(dataset_prefix, target_path, base_shadow_path, member,nonmember)
    else:
        run_attack(dataset_prefix, target_path, base_shadow_path, member,nonmember)



# def perform_membership_inference_attack(model, train_loader, test_loader, n, method='uncertainty', type='size'):
#     # Step 1: Importance scoring on the training set
#     # if method=='uncertainty':
#     #     uncertainty = metrics.Uncertainty(model, selection_method="Margin")
#     #     importance_scores = uncertainty.rank_uncertainty(train_loader)
#     # elif method=='forgetting':
#     #     forgetting_model = metrics.Forgetting(dst_train=train_loader.dataset, args=args, balance=True)
#     #     importance_scores=forgetting_model.calculate_importance_scores(model, train_loader)
#     # elif method=='grand':
#     #     grand_model = metrics.GraNd(dst_train=train_loader.dataset, args=args, balance=True)
#     #     importance_scores = grand_model.calculate_importance_scores_gradients(model, train_loader)
#     # else:
#     args.model="ResNet18"
#     # importance_scores = metric.calculate_importance_scores(model,method)

#     # dst_subset = torch.utils.data.Subset(train_loader.dataset, subset["indices"])
#     # print(len(importance_scores))
#     # print("+-+-+-")
#     # print(importance_scores)

#     # Step 2: Divide the training set into n subsets based on importance scores
#     # subset_indices = divide_into_subsets(importance_scores, n, type)

#     # Step 3 and Step 4: Perform membership inference attack and compute attack scores for each subset
#     mia_scores = []
#     subset_loaders=[]
#     args.model="resnet20"
#     for subset_idx in range(n):
#         print(train_loader.dataset.data.shape)
#         print(subset_indices.shape)
#         subset_data = torch.tensor(train_loader.dataset.data)[(subset_indices == subset_idx)]
#         # print(len(train_loader.dataset.targets))
#         subset_labels = torch.tensor(train_loader.dataset.targets)[(subset_indices == subset_idx)]
#         print(subset_data.shape,subset_labels.shape)
#         subset_dataset = CustomDataset(subset_data, subset_labels)
#         subset_loader = DataLoader(subset_dataset, batch_size=64, shuffle=False)
#         subset_loader.dataset.data = subset_loader.dataset.data.permute(0, 3, 1, 2)
        
#         total_samples = len(test_loader.dataset)
#         num_samples_to_sample =subset_loader.dataset.data.shape[0]
#         sampled_indices = random.choices(range(total_samples), k=num_samples_to_sample)
#         sampled_dataset = Subset(test_loader.dataset, sampled_indices)
#         sampled_dataloader = DataLoader(sampled_dataset, batch_size=64, shuffle=True, num_workers=4,generator=torch.Generator(device=f"cuda:{super_args.gpu_idx}"))

#         # mia_scores_subset=mia_by_loss(model,subset_loader,test_loader)
#         subset_loaders.append(subset_loader)
#         # attack(args,subset_loader,sampled_dataloader)

#         # mia_scores.append(mia_scores_subset.mean())
#     ###
#     # with open('/data/home/huqiang/DeepCore/save/subset_loaders.pkl', 'wb') as f:
#     #     pickle.dump(subset_loaders, f)
#     ###
    
#     args.model="ResNet18"
#     # Step 5: Output the midpoints of the importance score intervals and the corresponding attack scores
#     # importance_intervals = np.linspace(0, 1, n + 1)[:-1] + 0.5 / n
#     subset_midpoints = get_subset_midpoints(importance_scores, subset_indices, n,type)

#     return subset_midpoints, np.array(mia_scores)



# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[args.dataset] \
            (args.data_path)
args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names
print(args.num_classes)

# selection_args = dict(epochs=args.selection_epochs,
#                                   selection_method=args.uncertainty,
#                                   balance=args.balance,
#                                   greedy=args.submodular_greedy,
#                                   function=args.submodular
#                                   )
###
transform_train = transform_test = transforms.Compose([transforms.ToTensor(),
                                                               transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                    (0.2023, 0.1994, 0.2010))])
indices=np.load('/data/home/huqiang/DeepCore/RelaxLoss/results/CIFAR10/resnet20/advreg/full_idx.npy')
partition = Partition(dataset_size=60000, indices=indices)
trainset_idx, testset_idx = partition.get_target_indices()
target_trainset = CIFAR10(root='/data/home/huqiang/DeepCore/RelaxLoss/data', indices=trainset_idx,
                                       download=True, transform=transform_train)    
target_trainloader = torch.utils.data.DataLoader(target_trainset, batch_size=64, shuffle=False)                           
###

# method = methods.__dict__[args.selection](target_trainloader, args, args.fraction, args.seed, **selection_args)
# subset = method.select()

n_subsets = 10  # You can change the number of subsets as desired
# importance_intervals, mia_scores = perform_membership_inference_attack(model, target_trainloader, test_loader, n_subsets, method=globals().get(super_args.coreset)(dst_train=target_trainset,train_loader=target_trainloader, args=args, fraction=1,random_seed=42,epochs=20,balance=False) )
# importance_intervals, mia_scores = perform_membership_inference_attack(model, train_loader, test_loader, n_subsets, method="uncertainty")
# torch.save(model.state_dict(), './save/model.pt')
# Print the results
# print("Importance Score Intervals (Midpoints):", importance_intervals)
# print("Membership Inference Attack Scores:", mia_scores)



args.model="resnet20"
# ran=list(range(12000))
# random.shuffle(ran)

for i in range(10):
    # member_indices=np.load(f"/data/home/huqiang/DeepCore/save/{super_args.coreset}/sorted_indices{i}_10.npy")
    # member_indices=np.load(f"/data/home/huqiang/DeepCore/save/{super_args.coreset}/sorted_indices{0}_10.npy")
    # member_indices=ran[i*1200:(i+1)*1200]
    # print(member_indices)
    member_indices=[k for k in range(12000) if ((indices[k]>=5000*i)&(indices[k]<5000*(i+1)))]
    # nonmember_indices=[k for k in range(12000,24000) if ((indices[k]>=5000*i)&(indices[k]<5000*(i+1)))]
    nonmember_indices= np.random.choice(range(12000, 24000), size=1200, replace=False)
    print(super_args.coreset+"-------"+super_args.defense+"--------"+super_args.model_name)
    attack(args,member_indices,nonmember_indices)