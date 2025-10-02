# =======================
# Config: cfg_odvg_lora.py
# Dataset kecil (500 img), Colab T4
# =======================

# Augmentasi
data_aug_scales = [480, 512, 544, 576, 608, 640]
data_aug_max_size = 1024
data_aug_scales2_resize = [400, 500, 600]
data_aug_scales2_crop = [384, 600]
data_aug_scale_overlap = None

# Training
batch_size = 2     # T4 aman di batch kecil
epochs = 35        # lebih panjang dari full fine-tune
lr = 1e-4          # global LR
lr_drop_list = [20, 30]

# Model
modelname = 'groundingdino'
backbone = 'swin_T_224_1k'
position_embedding = 'sine'
hidden_dim = 256
enc_layers = 6
dec_layers = 6
nheads = 8
num_queries = 900

# LoRA
use_lora = True
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05

# Freeze backbone + BERT (biar hemat RAM)
freeze_keywords = ['backbone', 'bert']
lr_backbone = 1e-6
lr_backbone_names = ['backbone.0', 'bert']
lr_linear_proj_mult = 1e-6
lr_linear_proj_names = ['ref_point_head', 'sampling_offsets']

# Loss
set_cost_class = 1.0
set_cost_bbox = 5.0
set_cost_giou = 2.0
cls_loss_coef = 2.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0

# Text encoder
text_encoder_type = "bert-base-uncased"
max_text_len = 256
use_text_enhancer = True
use_fusion_layer = True
use_text_cross_attention = True

# Optimizer
weight_decay = 1e-4
clip_max_norm = 0.1

# Eval
use_coco_eval = False
label_list = [
    "Auxiliary", "Base Plate", "Box", "Connection Power Supply", 
    "Door Handle Drawer", "Drawer Stopper", "Front Plate", "Handle Drawer",
    "Index Mechanism", "Locking Mechanism", "Mounting Component", 
    "Push Button Index Mechanism", "Roda Drawer", "Support Outgoing", "Top Plate"
]
