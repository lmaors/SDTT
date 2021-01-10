import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self,seq_len,embed_size):
        super(PositionalEmbedding, self).__init__()
        self.seq_len = seq_len
        # self.seq_len = 210
        self.embed_size = embed_size
        # self.embed_size = 500
        self.position_embedding = nn.Embedding(self.seq_len, self.embed_size)
        self.layernorm = nn.LayerNorm(self.embed_size)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        position_ids = th.arange(self.seq_len, dtype=th.long)
        position_ids = position_ids.unsqueeze(0).expand(x.size(0),x.size(1)).cuda()
        position_embeddings = self.position_embedding(position_ids)
        embeddings = x + position_embeddings
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SentenceEncoder(nn.Module):
    def __init__(self,cfg):
        super(SentenceEncoder, self).__init__()
    
        """Container module with an encoder, a recurrent or transformer module, and a decoder."""
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
       
        self.model_type = 'Transformer'
        self.ninp=cfg.encoder.embed_size
        self.nhead = cfg.encoder.attn_heads
        self.dropout=cfg.encoder.dropout
        self.nhid=cfg.encoder.dim_feedforward  
        self.activation = cfg.encoder.activation
        self.nlayers = cfg.encoder.layers
        self.bs = cfg.batch_size
        self.seq_len = cfg.encoder.sentence_seq_len
        self.pos_embedding = PositionalEmbedding(self.seq_len,self.ninp).cuda()
        encoder_layers = TransformerEncoderLayer(self.ninp, self.nhead, self.nhid, self.dropout, self.activation)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers)


    def forward(self, src, src_mask=None):
        src = self.pos_embedding(src) 
        src = src.permute(1,0,2)   # shape(seq_len, batch_size, embed_size)
        out = self.transformer_encoder(src, src_mask)
        # print(self.src_mask)
        out = src.permute(1,0,2)   # shape(batch_size, seq_len, embed_size)
        return out

class VideoClipDecoder(nn.Module):
    def __init__(self,cfg):
        super(VideoClipDecoder, self).__init__()
        try:
            from torch.nn import TransformerDecoder, TransformerDecoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.ninp=cfg.decoder.embed_size
        self.nhead = cfg.decoder.attn_heads
        self.dropout=cfg.decoder.dropout
        self.nhid=cfg.decoder.dim_feedforward  
        self.activation = cfg.decoder.activation
        self.nlayers = cfg.decoder.layers
        self.bs = cfg.batch_size
        self.seq_len = cfg.decoder.clip_seq_len
        self.pos_embedding = PositionalEmbedding(self.seq_len,self.ninp).cuda()
        decoder_layers = TransformerDecoderLayer(self.ninp, self.nhead, self.nhid, self.dropout, self.activation)
        self.transformer_decoder = TransformerDecoder(decoder_layers, self.nlayers)
        
    def forward(self,tgt,memory,tgt_mask=None,memory_mask=None):
        tgt = self.pos_embedding(tgt) 
        tgt = tgt.permute(1,0,2)   # shape(seq_len, batch_size, embed_size)
        out = self.transformer_decoder(tgt, memory, tgt_mask)
        # print(self.src_mask)
        out = tgt.permute(1,0,2)   # shape(batch_size, seq_len, embed_size)
        return out
        
class ThumbnailTransformer(nn.Module):
    def __init__(self,cfg):
        super(ThumbnailTransformer, self).__init__()
        self.enc = SentenceEncoder(cfg)
        self.dec = VideoClipDecoder(cfg)
        self.ptr = PointerNetwork(cfg)
        
    
    def forward(self,src,tgt,src_mask=None,tgt_mask=None,memory_mask=None):
        memory = self.enc(src,src_mask)
        output = self.dec(tgt,memory,tgt_mask,memory_mask)
        probs =  self.ptr(memory,output)
        return probs

class PointerNetwork(nn.Module):
    def __init__(self,cfg):
        super(PointerNetwork, self).__init__()
        self.bs = cfg.batch_size
        self.hidden_size = cfg.pointernet.embed_size
        self.annoed_seq_len = cfg.annoed_seq_len
        self.weight_size = cfg.pointernet.weight_size
        self.dec = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.maxpool = nn.MaxPool2d((cfg.decoder.clip_seq_len,1))
        self.averagepool = nn.AvgPool2d((cfg.decoder.clip_seq_len,1))
        self.W1 = nn.Linear(self.hidden_size, self.weight_size, bias=False) # blending encoder
        self.W2 = nn.Linear(self.hidden_size, self.weight_size, bias=False) # blending decoder
        self.vt = nn.Linear(self.weight_size, 1, bias=False) # scaling sum of enc and dec by v.T

    
    def temporal_atten(self,probs,bg_position=0):
        # probs (bs,L)
        mask = th.zeros_like(probs,dtype=th.bool)
        for k,bs in enumerate(probs):

            if bg_position < len(bs):
                idx = bs[bg_position:].argmin(-1).item()
                mask[k][:bg_position+idx] = 1
                bg_position = idx + 1
            else:
                mask[k][:] = 1
        return mask,bg_position
    
    def forward(self, trs_enc_out, trs_dec_out):

        probs = []
        # Decoding
        cell_state = self.maxpool(trs_dec_out).squeeze().cuda() # (bs,1,  W)
        hidden = trs_enc_out[:,0:1,:].squeeze().cuda() # (bs,1,  W)
        # TODO
        # decoder_input = th.zeros_like(hidden).cuda()
        decoder_input = self.averagepool(trs_dec_out).squeeze().cuda()
        # decoder_input = th.zeros(self.bs, self.hidden_size).cuda()
        
        # hidden =  th.zeros_like(hidden).cuda()
        bg_position = 0
        mask = th.zeros((trs_dec_out.size(0),trs_dec_out.size(1)),dtype=th.bool).cuda()
        for i in range(self.annoed_seq_len): # range(M)
            # TODO 时序注意力需要修改 hidden
            hidden, cell_state = self.dec(decoder_input, (hidden, cell_state)) # (bs, h), (bs, h)
            # Compute blended representation at each decoder time step
            blend1 = self.W1(trs_dec_out)          # (bs,L,  W)
            blend2 = self.W2(hidden.unsqueeze(1))                  # (bs, W)
            blend_sum = th.tanh(blend1 + blend2)    # (bs,L,  W)
            out = self.vt(blend_sum).squeeze()        # (bs,L)
            out = -F.log_softmax(out + 1e-7, -1) # (bs, L)
            
            out = out.masked_fill(mask,value=7.0)
            mask, bg_position = self.temporal_atten(out,bg_position)
            hidden = trs_dec_out.masked_fill(mask.unsqueeze(2).repeat(1,1,trs_dec_out.size(2)),value=0.0)
            
            hidden = self.averagepool(hidden).squeeze()
            probs.append(out)
        probs = th.stack(probs, dim=1)           # (bs, M, L)
        # masked_probs = self.temporal_atten(probs)
        return probs
    
class SDTT(nn.Module):
    def __init__(self,cfg):
        super(SDTT, self).__init__()
        self.bs = cfg.batch_size
        self.embed_size = cfg.embed_size
        self.annoed_seq_len = cfg.annoed_seq_len
        self.src_seq_len = cfg.encoder.sentence_seq_len
        self.tgt_seq_len = cfg.decoder.clip_seq_len
        self.nhead = cfg.encoder.attn_heads
        self.linear = nn.Linear(cfg.c3d_fts_dim, cfg.embed_size)
        self.thumbnail_trs = ThumbnailTransformer(cfg)
    
    def _make_mask(self,video_clip_label,sentence_label):
        src_mask_list = []
        tgt_mask_list = []
        memory_mask_list = []
        for e in range(video_clip_label.shape[0]):
            e_sentence_label = sentence_label[e]
            
            len_sentence_label = len(e_sentence_label.nonzero())
            src_mask = th.zeros(e_sentence_label.size(0),e_sentence_label.size(0)) + 0.0
            src_mask[:len_sentence_label,:len_sentence_label] = 1.0
            
            e_clip_label = video_clip_label[e]
            len_clip_label = len(e_clip_label.nonzero())
            tgt_mask = th.zeros(e_clip_label.size(0),e_clip_label.size(0)) + 0.0
            tgt_mask[:len_clip_label,:len_clip_label] = 1.0
            
            memory_mask = th.zeros(e_sentence_label.size(0),e_clip_label.size(0)) + 0.0
            memory_mask[:len_sentence_label,:len_clip_label] = 1.0
            
            src_mask_list.append(src_mask)
            tgt_mask_list.append(tgt_mask)
            memory_mask_list.append(memory_mask)
            
        src_mask = th.stack(src_mask_list, dim=0).repeat(self.nhead,1,1)   
        tgt_mask = th.stack(tgt_mask_list, dim=0).repeat(self.nhead,1,1) 
        memory_mask = th.stack(memory_mask_list, dim=0).repeat(self.nhead,1,1)
        src_mask = src_mask.float().masked_fill(src_mask == 0, float('-inf')).masked_fill(src_mask == 1, float(0.0))
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        memory_mask = memory_mask.float().masked_fill(memory_mask == 0, float('-inf')).masked_fill(memory_mask == 1, float(0.0))
        
        return src_mask.cuda(), tgt_mask.cuda(), memory_mask.cuda()  # shape(bs*nhead,sent_len,sent_len) shape(bs*nhead,clip_len,clip_len) shape(bs*nhead,sent_len,clip_len)
    
    def forward(self,video_clip_feat,sentence_feat,video_clip_label,sentence_label):
        video_clip_feat = self.linear(video_clip_feat)
        src_mask, tgt_mask, memory_mask = self._make_mask(video_clip_label,sentence_label)
        output = self.thumbnail_trs(sentence_feat,video_clip_feat,src_mask,tgt_mask,memory_mask)
        return output
        

if __name__ == '__main__':

    from mmcv import Config
    cfg = Config.fromfile("./config.py")
    video_feat = th.randn(cfg.batch_size, cfg.max_len, 500).cuda()
    clip_mask = th.zeros((cfg.batch_size,cfg.max_len))
    sentence_feat = th.randn(cfg.batch_size, cfg.sentence_max_len, 768).cuda()
    sentense_mask = th.ones((cfg.batch_size,cfg.sentence_max_len))
    sdtt_model = SDTT(cfg).cuda()
    probs = sdtt_model(video_feat,sentence_feat,clip_mask,sentense_mask)
    outputs = probs.view(-1, 32) # (bs*M, L)
    y = th.randint(0,12,(12, 5)).cuda()
    y = y.view(-1) # (bs*M)
    loss = F.nll_loss(outputs, y)