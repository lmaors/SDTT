# import torch as th
import os.path as osp
import torch.utils.data as data
import torch
import torch as th
import torch.nn.functional as F
import os
import numpy as np
import json
import pickle as pkl
import h5py
# from multiprocessing import Manager, Pool, Process
from mmcv import Config
# from ..SDTT.utils import get_iou

class ThumbnailDataSet(data.Dataset):
    def __init__(self,data_dict):
        super(ThumbnailDataSet, self).__init__()
        self.data_dict = data_dict
        # self.max_length = cfg.max_length
        # self.data_root = cfg.data_root
        
    
    def __len__(self):
        return len(self.data_dict['video_name'])
    
    def __getitem__(self,index):
        data_dict = self.data_dict
        video_name = data_dict['video_name'][index]
        seg = data_dict['seg'][index]
        sentence = data_dict['sentence'][index]
        sentence_feat = data_dict['sentence_feat'][index]
        video_data = data_dict['video_data'][index]
        anno_ground_truth = data_dict['anno_ground_truth'][index]
        video_label = data_dict['video_label'][index]
        sentence_label = data_dict['sentence_label'][index]
        choose_label = data_dict['choose_label'][index]
        mean_choose_label = data_dict['mean_choose_label'][index]
        
        return video_name,seg,sentence,sentence_feat,sentence_label,video_data,video_label,choose_label,mean_choose_label,anno_ground_truth
    

def get_annoSegFts(video_name,seg,train_json,test_json,video_info,all_c3d_fts,cfg):
    max_len = cfg.max_len
    c3d_fts_dim = cfg.c3d_fts_dim
    c3d_per_f = cfg.c3d_per_f
    
    video_data = np.zeros([max_len, c3d_fts_dim]) + 0.0
    video_label = np.zeros(max_len) + 0.0
    video_full_fts = np.array(all_c3d_fts[video_name]['c3d_features'])
    [video_fps, video_total_frames, _, _] = video_info[video_name]
    if video_name in train_json:
        video_duration = train_json[video_name]['duration']
    else:
        video_duration = test_json[video_name]['duration']
    if video_duration <= 0:
        return None, None
    if len(video_full_fts) <= 0:
        return None, None
    left_frame = int(seg[0] * 1.0 / video_duration * video_total_frames)
    right_frame = int(seg[1] * 1.0 / video_duration * video_total_frames)
    start = left_frame
    jj = 0
    while int(start+video_fps) < right_frame and int(start+video_fps) < int(video_total_frames) and jj < max_len:
        current_left = start
        current_right = min(start + int(2*video_fps), video_total_frames)
        left_idx = int(current_left*1.0 / c3d_per_f)
        right_idx = int(current_right*1.0 / c3d_per_f)
        if right_idx <= video_full_fts.shape[0]:
            seg_fts = np.mean(video_full_fts[left_idx:right_idx, :], axis=0)
        else:
            seg_fts = np.zeros([c3d_fts_dim])+0.0
        video_data[jj, :] = seg_fts
        video_label[jj] = 1.0
        jj += 1
        start = current_right
    video_data = np.array(video_data)
    return video_data, video_label

def sentence_feat_padding(feat,cfg):
    feat = torch.from_numpy(feat)
    sent_max_len = cfg.sentence_max_len
    off_set = sent_max_len - feat.size(0)
    padding_feat = F.pad(feat.T, pad=(0, off_set), mode='constant', value=0)
    sentence_label = np.zeros(cfg.sentence_max_len)+0.0
    label_len = feat.shape[0]
    if label_len < cfg.sentence_max_len:
        sentence_label[0:label_len] = 1.0
    else:
        sentence_label[:] = 1.0
    # print(padding_feat.T)
    return padding_feat.T , sentence_label

def get_iou(item1,item2):
    
    item1 = item1.tolist() if item1.shape else [0] 
    item2 = item2.tolist() if item2.shape else [0] 
    intersection = 0
    
    for item in item1:
        if item in item2:
            intersection+=1
            
    union = len(item1)+len(item2)-intersection
    
    return intersection*1.0 / union

def choose_commmon_person(person_choose_label_list,cfg):
    
    
    max_len = cfg.max_len
    annoed_seq_len = cfg.annoed_seq_len
    final_choose_indice = []
    [person_num, max_video_len] = np.shape(
        person_choose_label_list)
    
    person_choose = [[] for k in range(person_num)]
    for person_id in range(person_num):
        person_choose[person_id] = np.argwhere(
            person_choose_label_list[person_id] > 0).squeeze()
    common_id = -1
    max_iou = 0.0
    # print(person_choose)
    for person_id in range(person_num):
        current_result = person_choose[person_id]
        iou_list = []
        for gap in range(person_num-1):
            other_result = person_choose[np.mod(
                person_id+gap+1, person_num)]
            iou_list.append(get_iou(current_result, other_result))
        if max_iou < np.mean(iou_list):
            max_iou = np.mean(iou_list)
            common_id = person_id
    final_choose_indice=person_choose_label_list[common_id]
    tag_list = [ i for i,v in enumerate(final_choose_indice) if v ==1]
    label_indice = torch.zeros([annoed_seq_len,max_len])
    for i,v in enumerate(tag_list):
        if i < annoed_seq_len:
            label_indice[i][v]= 1.0
    
    return label_indice

def data_preprocess(data,train_json,test_json,video_info,c3d_feats,all_sentence_feat,cfg):
    
    max_len = cfg.max_len
    annoed_seq_len = cfg.annoed_seq_len
    video_name_list = []
    seg_list = []
    sentence_list = []
    sentence_label_list = []
    sentence_feat_list = []
    video_data_list = []
    video_label_list = []
    anno_ground_truth_list = []
    choose_label_list = []
    
    mean_choose_label_list = []

    current_line_num = 0
    for e in data:
        content = e.split('\t')
        video_name = content[0]

        seg = eval(content[1])
        if seg[1] - seg[0] < 20.0:
            current_line_num += 1
            continue

        sentence = content[2]
        sentence_len = len(sentence.split(' '))
        if sentence_len>cfg.sentence_max_len:
            current_line_num += 1
            continue
        
        
        

        video_data, video_label = get_annoSegFts(video_name,seg,train_json,test_json,video_info,c3d_feats,cfg)
        if video_data is None and video_label is None:
            continue
        
        tmp = eval(content[3])

        mean_choose_label = np.zeros(max_len)+0.0
        choose_label = np.zeros([len(tmp), max_len], dtype=np.float)+0.0
        
        max_item = max(max(row) for row in tmp)
        if max_item <= max_len:
            person_id = 0
            for item in tmp:
                for ii in item:
                    mean_choose_label[ii-1] += 1
                    choose_label[person_id][ii-1] = 1.0
                person_id += 1
        else:
            continue
        if len(tmp) == 0:
            continue
        mean_choose_label = mean_choose_label * 1.0 / len(tmp)

        video_name_list.append(video_name)
        seg_list.append(seg)
        sentence_list.append(sentence)
        
        video_data_list.append(video_data)
        video_label_list.append(video_label)
        
        anno_ground_truth_list.append(content[3])
        choose_label = choose_commmon_person(choose_label,cfg)
        choose_label_list.append(choose_label)
        mean_choose_label_list.append(mean_choose_label)
        
        sentence_feat_name = video_name + '_'+ str(int(seg[0]))
        if all_sentence_feat[sentence_feat_name] is None:
            continue
        sentence_feat, sentence_label = sentence_feat_padding(all_sentence_feat[sentence_feat_name],cfg)
        sentence_feat_list.append(sentence_feat)
        sentence_label_list.append(sentence_label)
        
    data_dict = {
        'video_data':video_data_list,
        'video_name':video_name_list,
        'sentence':sentence_list,
        'sentence_feat':sentence_feat_list,
        'sentence_label':sentence_label_list,
        'seg':seg_list,
        'video_label':video_label_list,
        'anno_ground_truth':anno_ground_truth_list,
        'choose_label':choose_label_list,
        'mean_choose_label':mean_choose_label_list
        
    }
    
    return data_dict

def get_data(cfg):
    root_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
    with open(osp.join(root_path, 'dataset/annotated_thumbnail/anno_train.txt'), 'r') as fr:
        train_data = fr.readlines()
    with open(osp.join(root_path, 'dataset/annotated_thumbnail/anno_val.txt'), 'r') as fv:
        val_data = fv.readlines()
    with open(osp.join(root_path, 'dataset/annotated_thumbnail/anno_test.txt'), 'r') as ft:
        test_data = ft.readlines()
    
    train_json_path = osp.join(root_path, 'dataset/activitynet/train.json')
    test_json_path = osp.join(root_path, 'dataset/activitynet/val_merge.json')
    train_json = json.load(open(train_json_path))
    test_json = json.load(open(test_json_path))
    
    video_info = pkl.load(
    open(osp.join(root_path, 'dataset/activitynet/video_info.pkl'), 'rb'))
    
    feature_path_c3d = osp.join(
    root_path, 'dataset/activitynet_c3d_feat/sub_activitynet_v1-3.c3d.hdf5')
    all_c3d_fts = h5py.File(feature_path_c3d)
    
    feature_path_sentence = osp.join(
    root_path, 'dataset/albert_sentence_feat/sentence_feat.pkl')
    with open(feature_path_sentence,'rb') as fp:
        all_sentence_feat = pkl.load(fp)
    
    data_dict_train = data_preprocess(train_data,train_json,test_json,video_info,all_c3d_fts,all_sentence_feat,cfg)
    data_dict_val = data_preprocess(val_data,train_json,test_json,video_info,all_c3d_fts,all_sentence_feat,cfg)
    data_dict_test = data_preprocess(test_data,train_json,test_json,video_info,all_c3d_fts,all_sentence_feat,cfg)
    train_set = ThumbnailDataSet(data_dict_train)
    print('TrainSet has beed processed.')
    val_set = ThumbnailDataSet(data_dict_val)
    print('ValSet has beed processed.')
    test_set = ThumbnailDataSet(data_dict_test)
    print('TestSet has beed processed.')
    
    return train_set, val_set, test_set

if __name__ == "__main__":

    
    cfg = Config.fromfile('config.py')
    
    train_set, val_set, test_set = get_data(cfg)
    print(val_set)
    # sentence_feat_padding(torch.Tensor([[0,9,7,7],[1,2,2,1]]),'cfg')


