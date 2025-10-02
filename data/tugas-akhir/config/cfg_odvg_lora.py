# Save this file as data/tugas-akhir/config/cfg_odvg.py

# --- DATA ---
data_aug_scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
data_aug_max_size = 1333
data_aug_scales2_resize = [400, 500, 600]
data_aug_scales2_crop = [384, 600]
data_aug_scale_overlap = None
batch_size = 4

# --- MODEL ---
modelname = 'groundingdino'
backbone = 'swin_T_224_1k'
position_embedding = 'sine'
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
enc_layers = 6
dec_layers = 6
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
two_stage_type = 'standard'
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
transformer_activation = 'relu'
dec_pred_bbox_embed_share = True
dn_box_noise_scale = 1.0
dn_label_noise_ratio = 0.5
dn_label_coef = 1.0
dn_bbox_coef = 1.0
embed_init_tgt = True
dn_labelbook_size = 15
max_text_len = 256
text_encoder_type = "bert-base-uncased"
use_text_enhancer = True
use_fusion_layer = True
use_checkpoint = True
use_transformer_ckpt = True
use_text_cross_attention = True
text_dropout = 0.0
fusion_dropout = 0.0
fusion_droppath = 0.1
sub_sentence_present = True
max_labels = 50

# --- OPTIMIZER ---
lr = 0.0001
backbone_freeze_keywords = None
freeze_keywords = ['bert']
lr_backbone = 1e-05
lr_backbone_names = ['backbone.0', 'bert']
lr_linear_proj_mult = 1e-05
lr_linear_proj_names = ['ref_point_head', 'sampling_offsets']
weight_decay = 0.0001
param_dict_type = 'ddetr_in_mmdet'
ddetr_lr_param = False

# --- TRAINING ---
epochs = 15
lr_drop = 4
save_checkpoint_interval = 1
clip_max_norm = 0.1
onecyclelr = False
multi_step_lr = False
lr_drop_list = [4, 8]
frozen_weights = None

# --- LOSS ---
aux_loss = True
set_cost_class = 1.0
set_cost_bbox = 5.0
set_cost_giou = 2.0
cls_loss_coef = 2.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
enc_loss_coef = 1.0
interm_loss_coef = 1.0
no_interm_box_loss = False
mask_loss_coef = 1.0
dice_loss_coef = 1.0
focal_alpha = 0.25
focal_gamma = 2.0

# --- OTHER ---
batch_norm_type = 'FrozenBatchNorm2d'
label_list = [
    "Auxiliary", "Base Plate", "Box", "Connection Power Supply",
    "Door Handle Drawer", "Drawer Stopper", "Front Plate", "Handle Drawer",
    "Index Mechanism", "Locking Mechanism", "Mounting Component",
    "Push Button Index Mechanism", "Roda Drawer", "Support Outgoing", "Top Plate"
]
dn_scalar = 100
matcher_type = 'HungarianMatcher'
decoder_module_seq = ['sa', 'ca', 'ffn']
use_coco_eval = False