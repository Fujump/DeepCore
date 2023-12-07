import os
import sys
import argparse
import random
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(FILE_DIR, '../'))
sys.path.append(os.path.join(FILE_DIR, '../../'))
SAVE_ROOT = os.path.join(FILE_DIR, '../../../results/%s/%s/relaxloss')
import models as models
from base import CIFARTrainer
from utils import mkdir, str2bool, write_yaml, load_yaml, adjust_learning_rate, \
    AverageMeter, Bar, plot_hist, accuracy, one_hot_embedding, CrossEntropy_soft


#############################################################################################################
# get and save the arguments
#############################################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-name', type=str, help='experiment name, used for set up save_dir')
    parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'CIFAR100'], help='dataset name')
    parser.add_argument('--random_seed', '-s', type=int, default=1000, help='random seed')
    parser.add_argument('--model', type=str, help='model architecture')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--schedule_milestone', type=int, nargs='+', help='when to decrease the learning rate')
    parser.add_argument('--gamma', type=float, help='learning rate step gamma')
    parser.add_argument('--weight_decay', type=float, help='weight decay')
    parser.add_argument('--momentum', type=float, help='momentum')
    parser.add_argument('--train_batchsize', type=int, help='training batch size')
    parser.add_argument('--test_batchsize', type=int, help='testing batch size')
    parser.add_argument('--num_workers', type=int, help='number of workers')
    parser.add_argument('--num_epochs', '-ep', type=int, help='number of epochs')
    parser.add_argument('--alpha', type=float, help='the desired loss level')
    parser.add_argument('--upper', type=float, help='upper confidence level')
    parser.add_argument('--partition', type=str, choices=['target', 'shadow'], help='training partition')
    parser.add_argument('--if_resume', type=str2bool, help='If resume from checkpoint')
    parser.add_argument('--if_data_augmentation', '-aug', type=str2bool, help='If use data augmentation')
    parser.add_argument('--if_onlyeval', type=str2bool, help='If only evaluate the pre-trained model')
    parser.add_argument('--delta', type=float, help='range of the weights')
    parser.add_argument('--weight_metric', type=str,choices=['index', 'value','index_s','value_s'], help='metric of the weights')
    return parser


def check_args(parser):
    '''
    check and store the arguments as well as set up the save_dir
    :param args: arguments
    :return:
    '''
    ## set up save_dir
    args = parser.parse_args()
    save_dir = os.path.join(SAVE_ROOT % (args.dataset, args.model), args.exp_name)
    if args.partition == 'shadow':
        save_dir = os.path.join(save_dir, 'shadow')
    mkdir(save_dir)
    if args.random_seed is None:
        args.random_seed = random.randint(1, 10 ^ 5)

    ## load configs and store the parameters
    if args.if_onlyeval:
        preload_configs = load_yaml(os.path.join(save_dir, 'params.yml'))
        parser.set_defaults(**preload_configs)
        args = parser.parse_args()
    else:
        default_configs = load_yaml(FILE_DIR + '/configs/default.yml')
        specific_configs = load_yaml(FILE_DIR + '/configs/%s_%s.yml' % (args.dataset, args.model))
        parser.set_defaults(**default_configs)
        parser.set_defaults(**specific_configs)
        args = parser.parse_args()
        write_yaml(vars(args), os.path.join(save_dir, 'params.yml'))

    ## store this script
    shutil.copy(os.path.realpath(__file__), save_dir)
    return args, save_dir


#############################################################################################################
# helper functions
#############################################################################################################
# Define a custom loss function that penalizes the number of neighbors within a threshold delta
class CustomLoss(nn.Module):
    def __init__(self, delta):
        super(CustomLoss, self).__init__()
        self.delta = delta

    def forward(self, outputs,pairwise_distances, target_distance):
        # Count the number of neighbors within the threshold delta for each example
        # neighbor_count = torch.sum(pairwise_distances < self.delta, dim=1)
        batch_distances = torch.cdist(outputs, outputs, p=2)
        distance_var = torch.var(batch_distances)
        # distance_penalty = torch.abs(distance_var - target_distance)
        # Calculate the penalty term to encourage uniform neighbor count
        # neighbor_penalty = torch.abs((neighbor_count - mean_neighbors).float()).mean()/25

        # Add a penalty term based on neighbor_count
        # penalty_term = -torch.mean(neighbor_count.float())/100

        # Combine the standard loss and the penalty term
        total_loss = distance_var

        return total_loss

class Trainer(CIFARTrainer):
    def set_criterion(self):
        """Set up the relaxloss training criterion"""
        self.crossentropy_noreduce = nn.CrossEntropyLoss(reduction='none')
        self.crossentropy_soft = partial(CrossEntropy_soft, reduction='none')
        self.crossentropy = nn.CrossEntropyLoss()
        self.custom_loss = CustomLoss(delta=0.5)
        self.softmax = nn.Softmax(dim=1)
        self.pairwise_distances=torch.zeros(12000, 12000)
        self.target_distance=0
        self.alpha = self.args.alpha
        self.upper = self.args.upper
        self.delta=self.args.delta
        self.weight_metric=self.args.weight_metric
        torch.autograd.set_detect_anomaly(True)
        self.dists_of_knn_indices=[]
        for i in range(10):
            sub_index=np.load(f"/data/home/huqiang/DeepCore/save/dists_of_knn/sorted_indices{i}_10.npy")
            self.dists_of_knn_indices.extend(sub_index)

        num_examples = len(self.dists_of_knn_indices)
        min_weight = 1-self.delta
        max_weight = 1+self.delta

        # 创建权重张量
        self.weights = torch.zeros(num_examples)
        mean_dists=np.load("/data/home/huqiang/DeepCore/mean_dists_of_nn.npy")
        for i, index in enumerate(self.dists_of_knn_indices):
            if (self.weight_metric=="index"):
                weight = self.delta * ((2*i-num_examples) / (num_examples)) + 1
            elif(self.weight_metric=="value"):
                weight = self.delta * ((2*mean_dists[index]-np.max(mean_dists)-np.min(mean_dists)) / (np.max(mean_dists)-np.min(mean_dists))) + 1
            elif(self.weight_metric=="index_s"):
                weight = 2*torch.sigmoid(torch.tensor(self.delta*((2*i-num_examples) / (num_examples))))
            elif(self.weight_metric=="value_s"):
                weight = 2*torch.sigmoid(torch.tensor(self.delta*((mean_dists[index]-np.mean(mean_dists)) / np.var(mean_dists))))
            elif(self.weight_metric=="value_t"):
                weight = (1+torch.tanh((mean_dists[index]-np.mean(mean_dists)) / np.var(mean_dists)))/2
            self.weights[index] = weight

    def train(self, model, optimizer, epoch):
        model.train()

        losses = AverageMeter()
        losses_ce = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        batch_time = AverageMeter()
        dataload_time = AverageMeter()
        time_stamp = time.time()

        sum_neighbors=0
        bar = Bar('Processing', max=len(self.trainloader))
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            dataload_time.update(time.time() - time_stamp)
            outputs = model(inputs)
            loss_ce_full = self.crossentropy_noreduce(outputs, targets)
            # loss_ce = torch.mean(loss_ce_full)
            loss_ce = torch.mean(loss_ce_full * self.weights[batch_idx*self.args.train_batchsize:(batch_idx+1)*self.args.train_batchsize])

            if epoch % 2 == 0:  # gradient ascent/ normal gradient descent
                loss = (loss_ce - self.alpha).abs()
            else:
                if loss_ce > self.alpha:  # normal gradient descent
                    loss = loss_ce
                else:  # posterior flattening
                    pred = torch.argmax(outputs, dim=1)
                    correct = torch.eq(pred, targets).float()
                    confidence_target = self.softmax(outputs)[torch.arange(targets.size(0)), targets]
                    confidence_target = torch.clamp(confidence_target, min=0., max=self.upper)
                    confidence_else = (1.0 - confidence_target) / (self.num_classes - 1)
                    onehot = one_hot_embedding(targets, num_classes=self.num_classes)
                    soft_targets = onehot * confidence_target.unsqueeze(-1).repeat(1, self.num_classes) \
                                   + (1 - onehot) * confidence_else.unsqueeze(-1).repeat(1, self.num_classes)
                    loss = (1 - correct) * self.crossentropy_soft(outputs, soft_targets) - 1. * loss_ce_full
                    loss = torch.mean(loss)
            # neighbor_loss = self.custom_loss(outputs,self.pairwise_distances,self.target_distance)
            # loss=loss+neighbor_loss
            # Calculate pairwise distances for this batch and update the overall matrix
            # batch_distances = torch.cdist(outputs, outputs, p=2)
            # start_idx = batch_idx * len(inputs)
            # end_idx = start_idx + len(inputs)
            # self.pairwise_distances[start_idx:end_idx, start_idx:end_idx] = batch_distances.clone()


            ### Record accuracy and loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            losses_ce.update(loss_ce.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            ### Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### Record the total time for processing the batch
            batch_time.update(time.time() - time_stamp)
            time_stamp = time.time()

            ### Progress bar
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx + 1,
                size=len(self.trainloader),
                data=dataload_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
        self.mean_neighbors=sum_neighbors/12000

        bar.finish()
        return (losses_ce.avg, top1.avg, top5.avg)


#############################################################################################################
# main function
#############################################################################################################
def main():
    ### config
    args, save_dir = check_args(parse_arguments())

    ### Set up trainer
    trainer = Trainer(args, save_dir)
    model = models.__dict__[args.model](num_classes=trainer.num_classes)
    model = torch.nn.DataParallel(model)
    model = model.to(trainer.device)
    torch.backends.cudnn.benchmark = True
    print('Total params: %.2f' % (sum(p.numel() for p in model.parameters())))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    ### Load checkpoint
    start_epoch = 0
    logger = trainer.logger
    if args.if_resume or args.if_onlyeval:
        try:
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(os.path.join(save_dir, 'checkpoint.pkl'))
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['opt_state_dict'])
            logger = Logger(os.path.join(save_dir, 'log.txt'), title=title, resume=True)
        except:
            pass

    if args.if_onlyeval:
        print('\nEvaluation only')
        test_loss, test_acc, test_acc5 = trainer.test(model)
        print(' Test Loss: %.8f, Test Acc(top1): %.2f, Test Acc(top5): %.2f' % (test_loss, test_acc, test_acc5))
        return

    ### Training
    for epoch in range(start_epoch, args.num_epochs):
        adjust_learning_rate(optimizer, epoch, args.gamma, args.schedule_milestone)
        train_ce, train_acc, train_acc5 = trainer.train(model, optimizer, epoch)
        test_ce, test_acc, test_acc5 = trainer.test(model)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        logger.append([lr, train_ce, test_ce, train_acc, test_acc, train_acc5, test_acc5])
        print('Epoch %d, Train acc: %f, Test acc: %f, lr: %f' % (epoch, train_acc, test_acc, lr))

        ### Save checkpoint
        save_dict = {'epoch': epoch,
                     'model_state_dict': model.state_dict(),
                     'opt_state_dict': optimizer.state_dict()}
        torch.save(save_dict, os.path.join(save_dir, 'checkpoint.pkl'))
        torch.save(model, os.path.join(save_dir, 'model.pt'))

    ### Visualize
    trainer.logger_plot()
    train_losses, test_losses = trainer.get_loss_distributions(model)
    plot_hist([train_losses, test_losses], ['train', 'val'], os.path.join(save_dir, 'hist_ep%d.png' % epoch))


if __name__ == '__main__':
    main()
