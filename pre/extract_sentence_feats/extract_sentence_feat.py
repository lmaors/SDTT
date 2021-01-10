import nltk
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import to_array
import numpy as np
import os
import os.path as osp
import pickle
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

base_dir = os.path.dirname(os.path.dirname(__file__))
config_path = os.path.join(
    base_dir, 'pred_models/albert_base/albert_config.json')
checkpoint_path = os.path.join(
    base_dir, 'pred_models/albert_base/model.ckpt-best')
dict_path = os.path.join(
    base_dir, 'pred_models/albert_base/vocab.txt')
#加载punkt句子分割器

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
model = build_transformer_model(
    config_path, checkpoint_path, model='albert')  # 建立模型，加载权重
sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')



def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def extract_feat(video_id,sentence):
    global video_sentence_feat
    if sentence:
        # sentences = sen_tokenizer.tokenize(sentence)
        # 编码
        token_ids, segment_ids = tokenizer.encode(sentence)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        outs = model.predict([token_ids, segment_ids])
        # print(outs)
        feat = outs.squeeze(0)
        # feat = normalization(np.sum(outs,axis=0))
    else:
        feat =  np.zeros((1,768),dtype=np.float32)
    video_sentence_feat[video_id] = feat


if __name__ == '__main__':
    video_sentence_feat = {}
    # token_ids, segment_ids = tokenizer.encode(['i', 'like', 'china', 'teacher'])
    # token_ids, segment_ids = to_array([token_ids], [segment_ids])
    # print(model.predict([token_ids, segment_ids]))
    
    root_path = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
    with open(osp.join(root_path, 'dataset/annotated_thumbnail/anno_train.txt'), 'r') as fr:
        train_data = fr.readlines()
    with open(osp.join(root_path, 'dataset/annotated_thumbnail/anno_val.txt'), 'r') as fv:
        val_data = fv.readlines()
    with open(osp.join(root_path, 'dataset/annotated_thumbnail/anno_test.txt'), 'r') as ft:
        test_data = ft.readlines()
    all_data_list = train_data+val_data+test_data
    
    
    for e in all_data_list:
        video_id = e.split('\t')[0]
        video_anno_begin = eval(e.split('\t')[1])[0]
        video_sentence = e.split('\t')[2]
        video_id = video_id+'_'+str(int(video_anno_begin))
        print(video_id)
        extract_feat(video_id,video_sentence)
        print('Have processed the video of {}.'.format(video_id))
    
    if not osp.exists(osp.join(root_path,'dataset/albert_sentence_feat')):
        os.mkdir(osp.join(root_path,'dataset/albert_sentence_feat'))
    with open(osp.join(root_path,'dataset/albert_sentence_feat/sentence_feat.pkl'), 'wb') as fw:
        pickle.dump(video_sentence_feat,fw)

