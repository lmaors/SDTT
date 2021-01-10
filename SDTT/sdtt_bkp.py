import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self,cfg):
        super(PositionalEmbedding, self).__init__()
        self.seq_len = cfg.model.transformer_seq_len
        # self.seq_len = 210
        self.embed_size = cfg.model.embed_size
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


class VideoClipTransformer(nn.Module):
    def __init__(self,cfg):
        super(VideoClipTransformer, self).__init__()
    
        """Container module with an encoder, a recurrent or transformer module, and a decoder."""
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
       
        self.model_type = 'Transformer'
        self.src_mask = None
        self.ninp=cfg.model.embed_size
        self.nhead = cfg.model.transformer_attn_heads
        self.dropout=cfg.model.transformer_dropout
        self.nhid=cfg.model.transformer_dim_feedforward  
        self.activation = cfg.model.transformer_activation
        self.nlayers = cfg.model.transformer_n_layers
        self.bs = cfg.batch_size
        self.seq_len = cfg.model.transformer_seq_len
        self.pos_embedding = PositionalEmbedding(cfg).cuda()
        encoder_layers = TransformerEncoderLayer(self.ninp, self.nhead, self.nhid, self.dropout, self.activation)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers)
        

    def _generate_square_subsequent_mask(self, mask):
        # mask = (th.triu(th.ones(mask.size(0),mask.size(0))) == 1).transpose(0, 1)
        mask = mask.unsqueeze(2)
        # mask = mask.permute(1,0,2)   # shape(batch_size, seq_len, embed_size)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.expand(mask.size(0),self.seq_len,self.seq_len).repeat(self.nhead,1,1)
        # print(mask)
        return mask


    def forward(self, src, mask=None):
        if mask is not None:
            device = src.device
            # if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_subsequent_mask(mask).to(device)
            self.src_mask = mask
        else:
            self.src_mask = None
        src = self.pos_embedding(src) 
        src = src.permute(1,0,2)   # shape(seq_len, batch_size, embed_size)
        out = self.transformer_encoder(src, self.src_mask)
        # print(self.src_mask)
        out = src.permute(1,0,2)   # shape(batch_size, seq_len, embed_size)
        return out

class PointerNetwork(nn.Module):
    def __init__(self, input_size, emb_size, weight_size, answer_seq_len, hidden_size=512, is_GRU=False):
        super(PointerNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.answer_seq_len = answer_seq_len
        self.weight_size = weight_size
        self.emb_size = emb_size
        self.is_GRU = is_GRU

        # self.emb = nn.Embedding(input_size, emb_size)  # embed inputs
        if is_GRU:
            self.enc = nn.GRU(emb_size, hidden_size, batch_first=True)
            self.dec = nn.GRUCell(emb_size, hidden_size) # GRUCell's input is always batch first
        else:
            self.enc = nn.LSTM(emb_size, hidden_size, batch_first=True)
            self.dec = nn.LSTMCell(emb_size, hidden_size) # LSTMCell's input is always batch first

        self.W1 = nn.Linear(hidden_size, weight_size, bias=False) # blending encoder
        self.W2 = nn.Linear(hidden_size, weight_size, bias=False) # blending decoder
        self.vt = nn.Linear(weight_size, 1, bias=False) # scaling sum of enc and dec by v.T

    def forward(self, input, encoder_init_from_sentence):
        batch_size = input.size(0)
        # input = self.emb(input) # (bs, L, embd_size)

        # Encoding
        self.enc.flatten_parameters()
        # self.dec.flatten_parameters()
        encoder_states, hc = self.enc(input) # encoder_state: (bs, L, H)
        encoder_states = encoder_states.transpose(1, 0) # (L, bs, H)

        # Decoding states initialization
        # TODO
        # decoder_input = th.zeros(batch_size, self.emb_size).cuda() # (bs, embd_size)
        decoder_input = encoder_init_from_sentence.squeeze().cuda()
        hidden = th.zeros([batch_size, self.hidden_size]).cuda()  # (bs, h)
        cell_state = encoder_states[-1]                                # (bs, h)

        probs = []
        # Decoding
        for i in range(self.answer_seq_len): # range(M)
            if self.is_GRU:
                hidden = self.dec(decoder_input, hidden) # (bs, h), (bs, h)
            else:
                hidden, cell_state = self.dec(decoder_input, (hidden, cell_state)) # (bs, h), (bs, h)

            # Compute blended representation at each decoder time step
            blend1 = self.W1(encoder_states)          # (L, bs, W)
            blend2 = self.W2(hidden)                  # (bs, W)
            blend_sum = th.tanh(blend1 + blend2)    # (L, bs, W)
            out = self.vt(blend_sum).squeeze()        # (L, bs)
            out = F.log_softmax(out.transpose(0, 1).contiguous() + 1e-8, -1) # (bs, L)
            probs.append(-out)

        probs = th.stack(probs, dim=1)           # (bs, M, L)

        return probs
    
class SDTT(nn.Module):
    def __init__(self,cfg):
        super(SDTT, self).__init__()
        self.bs = cfg.batch_size
        self.embed_size = cfg.embed_size
        self.hidden_size = cfg.model.ptr_net_hidden_size
        self.annoed_seq_len = cfg.annoed_seq_len
        self.linear = nn.Linear(cfg.c3d_fts_dim, cfg.embed_size)
        self.align_sentence_video_clip = nn.Linear(cfg.sentence_max_len-1,cfg.max_len)
        self.video_clip_transformer = VideoClipTransformer(cfg)
        self.ptr_net = PointerNetwork(self.bs,self.embed_size,self.hidden_size,self.annoed_seq_len)
        
    def forward(self,video_clip_feat,sentence_feat,mask=None):
        video_clip_feat = self.linear(video_clip_feat)
        token_feat = sentence_feat[::,1:]
        token_feat = self.align_sentence_video_clip(token_feat.permute(0,2,1))
        token_feat = token_feat.permute(0,2,1)
        multi_feat = video_clip_feat + token_feat
        hid_out = self.video_clip_transformer(multi_feat,mask)
        out = self.ptr_net(hid_out,sentence_feat[::,0:1])
        return out
        

if __name__ == '__main__':
    # x = th.Tensor([[1, 1, 0],[1,0,0]])
    # x = x.unsqueeze(2)
    # y = x.expand(2, 3,3)
    # y = y.repeat(8,1,1)
    # # print(y)
    # probs = th.randn(10, 5, 2)
    # print(probs)
    # pred_indices = probs.argmin(dim=2)
    # print(pred_indices)
    from mmcv import Config

    cfg = Config.fromfile("./config.py")
    video_feat = th.randn(cfg.batch_size, cfg.max_len, 500).cuda()
    mask_ = th.ones((cfg.batch_size,cfg.max_len))
    sentence_feat = th.randn(cfg.batch_size, cfg.sentence_max_len, 500).cuda()
    sdtt_model = SDTT(cfg).cuda()
    probs = sdtt_model(video_feat,sentence_feat,mask_)
    outputs = probs.view(-1, 32) # (bs*M, L)
    y = th.randint(0,12,(12, 5)).cuda()
    y = y.view(-1) # (bs*M)
    loss = F.nll_loss(outputs, y)