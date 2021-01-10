experiment_name = "DynamicThumbnail"
experiment_description = "dynamic thumbnail generation"

max_len = 128 # video clip max length
c3d_fts_dim = 500
c3d_per_f = 8
sentence_max_len = 40
annoed_seq_len = 5
batch_size = 16
gpus='0,1'
data_loader_kwargs = dict(num_workers=16, pin_memory=True, drop_last=True)

resume = None
train_flag = 0
test_flag = 1
embed_size = 768
model_name = 'SDTT'

decoder = dict(
    clip_seq_len=max_len,
    embed_size=768,
    attn_heads=8,
    dropout=0.4,
    dim_feedforward = 768*4, # 4 * embed_size
    activation="relu",
    layers=6,
)

encoder = dict(
    sentence_seq_len=sentence_max_len,
    embed_size=768,
    attn_heads=8,
    dropout=0.4,
    dim_feedforward = 768*4, # 4 * embed_size
    activation="relu",
    layers=0,
)

pointernet = dict(
    embed_size=768,
    weight_size=192,
)
epochs = 40

# optimizer
optim = dict(name='Adam',
             setting=dict(lr=8e-5, weight_decay=5e-4))

# optim = dict(name='SGD',
#              setting=dict(lr=5e-5, weight_decay=5e-4, momentum=0.9))

stepper = dict(name='MultiStepLR',
               setting=dict(milestones=[20]))

stepper_cos_lr = dict(name='CosineAnnealingLR',
               setting=dict(T_max=epochs+20, eta_min=0,last_epoch=-1))

logger = dict(log_interval=50, logs_dir="logs/{}".format(experiment_name))