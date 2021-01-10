import os
import torch
import os.path as osp
import shutil
import numpy as np


def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    os.makedirs(osp.dirname(fpath),exist_ok=True)
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth.tar'))

def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath)
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))
    
def get_iou(item1,item2):
    if item1.ndim != 1:
        item1 = item1.squeeze()
    if item2.ndim != 1:
        item2 = item2.squeeze()
    intersection = 0
    for item in item1:
        if item in item2:
            intersection+=1
    union = len(item1)+len(item2)-intersection
    return intersection*1.0 / union

def get_precision(pred_bs,ground_truth_bs):
    ground_truth_bs = [eval(i) for i in ground_truth_bs]
    precision_list = []
    for n,v in enumerate(pred_bs):
        each_bs_precision = []
        v = v.cpu().numpy().tolist()
        for i in ground_truth_bs[n]:
            intersection = set(v).intersection(set(i))
            # p = len(intersection) / len(set(v))
            p = len(intersection) / len(v)
            each_bs_precision.append(p)
        precision_list.append(max(each_bs_precision))
    precision = sum(precision_list)
    return precision

def get_recall(pred_bs,ground_truth_bs):
    ground_truth_bs = [eval(i) for i in ground_truth_bs]
    recall_list = []
    
    for n,v in enumerate(pred_bs):
        each_bs_recall = []    
        v = v.cpu().numpy().tolist()
        for i in ground_truth_bs[n]:
            intersection = set(v).intersection(set(i))
            # compute recall score
            recall = len(intersection) / len(set(i))
            each_bs_recall.append(recall)
        recall_list.append(max(each_bs_recall))
    batch_recall_score = sum(recall_list)
    return batch_recall_score
    
def get_iou(pred_bs,ground_truth_bs):
    ground_truth_bs = [eval(i) for i in ground_truth_bs]
    iou_list = []
    
    for n,v in enumerate(pred_bs):
        each_bs_iou = []
        v = v.cpu().numpy().tolist()
        for i in ground_truth_bs[n]:
            intersection = set(v).intersection(set(i))
            union = set(v).union(set(i))
            
            # compute recall score
            recall = len(intersection) / len(union)
            each_bs_iou.append(recall)
        iou_list.append(max(each_bs_iou))
    batch_iou = sum(iou_list)
    return batch_iou

def get_f1(pred_bs,ground_truth_bs):
    ground_truth_bs = [eval(i) for i in ground_truth_bs]
    f1_list = []
    
    for n,v in enumerate(pred_bs):
        each_bs_f1 = []
        v = v.cpu().numpy().tolist()
        for i in ground_truth_bs[n]:
            intersection = set(v).intersection(set(i))
            union = set(v).union(set(i))
            precision = len(intersection) / len(v)
            # compute recall score
            recall = len(intersection) / len(set(i))
            if precision + recall != 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
            each_bs_f1.append(f1)
        f1_list.append(max(each_bs_f1))
    batch_f1 = sum(f1_list)
    return batch_f1

if __name__ == "__main__":
    v = [2,3,10,10]
    i =  [2,3,5,8,10]
    a = set(v)
    b = set(i)
    intersection = set(v).intersection(set(i))
    print(intersection)
    