import argparse
import torch as th
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from mmcv import Config
import os
import os.path as osp
import numpy as np
import json
from data_process.data_preprocess import get_data
from SDTT import sdtt
from SDTT.utils import save_checkpoint,load_checkpoint,get_precision,get_recall,get_iou,get_f1
import time

def train(model, train_loader, optimizer, scheduler, epoch, criterion,cfg):
    model.train()
    for each_batch,(video_name,seg,sentence,sentence_feat,sentence_label,video_data,video_label,choose_label,mean_choose_label,anno_ground_truth) in enumerate(train_loader):
        
        video_clip_feat = video_data.float().cuda()
        clip_mask = video_label.cuda()
        sentence_mask = sentence_label.cuda()
        sentence_feat = sentence_feat.cuda()
        target = choose_label.view(-1,cfg.max_len).cuda() # (bs*M)
        
        optimizer.zero_grad()
        probs = model(video_clip_feat,sentence_feat,clip_mask,sentence_mask)
        outputs = probs.view(-1, cfg.max_len) # (bs*M, L)
        # y = th.randint(0,12,(12, cfg.annoed_seq_len)).cuda()
        loss = outputs.mul(target).sum()
        loss.backward()
        optimizer.step()
        # loss = F.nll_loss(outputs, y)
        if each_batch % cfg.logger.log_interval == 0:
        
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage Loss: {:.6f}'.format(
                epoch, int(each_batch * len(video_clip_feat)), len(train_loader.dataset),
                100. * each_batch / len(train_loader), loss.item()/cfg.batch_size))
    scheduler.step()

        
def test(model, data_loader, optimizer, scheduler, criterion,cfg):
    model.eval()
    test_loss = 0
    batch_num = 0
    precision = 0.0
    recall_score = 0.0
    iou = 0.0
    f1 = 0.0
    with th.no_grad():
        for each_batch,(video_name,seg,sentence,sentence_feat,sentence_label,video_data,video_label,choose_label,mean_choose_label,anno_ground_truth) in enumerate(data_loader):
            batch_num += 1
            video_clip_feat = video_data.float().cuda()
            clip_mask = video_label.cuda()
            sentence_mask = sentence_label.cuda()
            sentence_feat = sentence_feat.cuda()
            target = choose_label.view(-1,cfg.max_len).cuda() # (bs*M)
            
            probs = model(video_clip_feat,sentence_feat,clip_mask,sentence_mask)
            outputs = probs.view(-1, cfg.max_len) # (bs*M, L)
            pred_indices = probs.argmin(dim=2)  # (bs,M)
            
            # get recall precision
            average_precision = get_precision(pred_indices,anno_ground_truth)
            precision += average_precision
            
            # get recall score
            batch_recall_score = get_recall(pred_indices,anno_ground_truth)
            recall_score += batch_recall_score
            
            # get iou score
            batch_iou_score = get_iou(pred_indices,anno_ground_truth)
            iou += batch_iou_score
            
            # get f1 score
            batch_f1_score = get_f1(pred_indices,anno_ground_truth)
            f1 += batch_f1_score
            
            loss = outputs.mul(target).sum()
            test_loss += loss.item()
            
    test_loss = test_loss / batch_num / cfg.batch_size
    precision = precision / batch_num / cfg.batch_size
    recall_score = recall_score / batch_num / cfg.batch_size
    iou = iou / batch_num / cfg.batch_size
    f1 = f1 / batch_num / cfg.batch_size
    
    return test_loss,precision,recall_score,iou,f1




def parse_args():
    parser = argparse.ArgumentParser(description='Runner')
    parser.add_argument('--config', default='config.py', help='config file path')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
    print('Reading data ...')
    train_set, test_set, val_set = get_data(cfg)
    print('DataLoader is processing ...')
    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size,
        shuffle=True, **cfg.data_loader_kwargs)
    test_loader = DataLoader(
        test_set, batch_size=cfg.batch_size,
        shuffle=False, **cfg.data_loader_kwargs)
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size,
        shuffle=True, **cfg.data_loader_kwargs)
    
    model = sdtt.__dict__[cfg.model_name](cfg).cuda()
    model = nn.DataParallel(model)
    
    if cfg.resume is not None:
        checkpoint = load_checkpoint(cfg.resume)
        model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.__dict__[cfg.optim.name](
        model.parameters(), **cfg.optim.setting
    )
    scheduler = optim.lr_scheduler.__dict__[cfg.stepper.name](
        optimizer, **cfg.stepper.setting
    )
    criterion = nn.NLLLoss()
    
    if cfg.train_flag:
        print("Training...")
        max_evp = -1
        for epoch in range(1, cfg.epochs + 1):
            train(model, train_loader, optimizer, scheduler, epoch, criterion, cfg)
            val_loss,precision,recall_score,iou,f1 = test(model, val_loader, optimizer, scheduler,criterion, cfg)
            print('----------------ValDataset Metric-----------------')
            print("Average Loss: {:.3f}".format(val_loss))
            print("Average Precision: {:.3f}".format(precision))
            print("Recall: {:.3f}".format(recall_score))
            print("IoU: {:.3f}".format(iou))
            print("F1 Score: {:.3f}".format(f1))
            print('---------------------------------------------------')
            test_loss,test_precision,test_recall_score,test_iou,test_f1 = test(model, test_loader, optimizer, scheduler,criterion, cfg)
            print('----------------TestDataset Metric-----------------')
            print("Average Loss: {:.3f}".format(test_loss))
            print("Average Precision: {:.3f}".format(test_precision))
            print("Recall: {:.3f}".format(test_recall_score))
            print("IoU: {:.3f}".format(test_iou))
            print("F1 Score: {:.3f}".format(test_f1))
            print('---------------------------------------------------')
            if precision >= max_evp:
                is_best = True
                max_ap = precision
            else:
                is_best = False
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
            }, is_best=is_best, fpath=osp.join(cfg.logger.logs_dir, 'checkpoint.pth.tar'))
            
    if cfg.test_flag:
        print('Testing with saved model ...')
        checkpoint = load_checkpoint(
            osp.join(cfg.logger.logs_dir, 'model_best.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])
        t1 = time.time()
        test_loss,precision,recall_score,iou,f1 = test(model, test_loader, optimizer, scheduler, criterion, cfg)
        t2 = time.time() - t1
        print('----------------Metric-----------------')
        print("Average Loss: {:.3f}".format(test_loss))
        print("Average Precision: {:.3f}".format(precision))
        print("Recall: {:.3f}".format(recall_score))
        print("IoU: {:.3f}".format(iou))
        print("F1 Score: {:.3f}".format(f1))
        print('---------------------------------------')
        print('Spend time {}s'.format(t2))
        final_dict = {
            'precision':precision,
            'loss':test_loss,
            'recall': recall_score,
            'iou':iou,
            'f1':f1
        }
        log_dict = {'cfg': cfg.__dict__['_cfg_dict'], 'final': final_dict}
        with open(osp.join(cfg.logger.logs_dir, "log.json"),"w") as f:
            json.dump(str(log_dict),f,indent=4)
        
        # print(log_dict)
    