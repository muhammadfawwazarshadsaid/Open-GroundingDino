# ============================================
# Config for LoRA fine-tuning GroundingDINO
# ============================================

# Dataset
train_dataset = dict(
    root="data/tugas-akhir/train/",
    anno="data/tugas-akhir/annotations/train_odvg_final.jsonl",
    label_map="data/tugas-akhir/config/label_map_final.json",
    dataset_mode="odvg",
)

val_dataset = dict(
    root="data/tugas-akhir/valid/",
    anno="data/tugas-akhir/valid/_annotations.coco_final.json",
    label_map="data/tugas-akhir/config/label_map_final.json",
    dataset_mode="coco",
)

# Augmentasi
data_aug_scales = [480, 512, 544, 576, 608, 640]
data_aug_max_size = 1024
data_aug_scales2_resize = [400, 500, 600]
data_aug_scales2_crop = [384, 600]
data_aug_scale_overlap = None

# Training
batch_size = 2                 # kecil biar aman di T4
epochs = 35                    # lebih panjang buat LoRA converge
lr = 1e-4                      # global LR
lr_drop_list = [20, 30]        # schedule turunin LR

# Model
modelname = 'groundingdino'
backbone = 'swin_T_224_1k'
position_embedding = 'sine'
hidden_dim = 256
enc_layers = 6
dec_layers = 6
nheads = 8
num_queries = 900

# Loss weights
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
