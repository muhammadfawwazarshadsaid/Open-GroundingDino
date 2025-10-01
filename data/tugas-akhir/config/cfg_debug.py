# --- Dataset config ---
data_aug_scales = [480, 512, 544]   # kecil aja
data_aug_max_size = 640             # hemat VRAM
data_aug_scales2_resize = [400, 500]
data_aug_scales2_crop = [384, 600]
batch_size = 2                      # kecil banget biar cepat

# --- Model config ---
modelname = 'groundingdino'
backbone = 'swin_T_224_1k'
position_embedding = 'sine'
enc_layers = 3                      # dikurangi
dec_layers = 3
dim_feedforward = 512
hidden_dim = 128
nheads = 4
num_queries = 100                   # super kecil
num_feature_levels = 2

# --- Optimizer & LR ---
lr = 1e-4
lr_backbone = 1e-5
weight_decay = 1e-4
epochs = 2                          # debug cepet
lr_drop = 1
save_checkpoint_interval = 1
clip_max_norm = 0.1

# --- Loss coefficients ---
set_cost_class = 1.0
set_cost_bbox = 5.0
set_cost_giou = 2.0
cls_loss_coef = 2.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0

# --- Extra ---
num_select = 50
batch_norm_type = 'FrozenBatchNorm2d'
masks = False
aux_loss = True

# --- Labels ---
label_list = [
    "Auxiliary", "Base Plate", "Box", "Connection Power Supply", 
    "Door Handle Drawer", "Drawer Stopper", "Front Plate", "Handle Drawer",
    "Index Mechanism", "Locking Mechanism", "Mounting Component", 
    "Push Button Index Mechanism", "Roda Drawer", "Support Outgoing", "Top Plate"
]
