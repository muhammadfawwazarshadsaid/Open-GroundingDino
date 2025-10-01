# --- Dataset config ---
data_aug_scales = [480, 512, 544, 576, 608, 640]   # lebih kecil dari default
data_aug_max_size = 800                            # hemat VRAM
data_aug_scales2_resize = [400, 500, 600]
data_aug_scales2_crop = [384, 600]
batch_size = 2                                     # aman di Colab GPU

# --- Model config ---
modelname = 'groundingdino'
backbone = 'swin_T_224_1k'
position_embedding = 'sine'
enc_layers = 4
dec_layers = 4
dim_feedforward = 1024
hidden_dim = 256
nheads = 8
num_queries = 300
num_feature_levels = 3

# --- Optimizer & LR ---
lr = 1e-4
lr_backbone = 1e-5
weight_decay = 1e-4
epochs = 20
lr_drop = 10
save_checkpoint_interval = 5
clip_max_norm = 0.1

# --- Loss coefficients ---
set_cost_class = 1.0
set_cost_bbox = 5.0
set_cost_giou = 2.0
cls_loss_coef = 2.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0

# --- Extra ---
num_select = 100
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

# --- Eval ---
use_coco_eval = True   # karena dataset_mode = "coco"
