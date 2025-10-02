# ================================================================================= #
#        Konfigurasi GroundingDINO (Freeze Backbone & LR lebih rendah)              #
# ================================================================================= #

# --- Pengaturan Augmentasi & Dataset ---
data_aug_scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
data_aug_max_size = 1333
data_aug_scales2_resize = [400, 500, 600]
data_aug_scales2_crop = [384, 600]
data_aug_scale_overlap = None
batch_size = 4
masks = False
use_coco_eval = False

# --- Pengaturan Arsitektur Model ---
modelname = 'groundingdino'
backbone = 'swin_T_224_1k'
position_embedding = 'sine'
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
num_feature_levels = 4
batch_norm_type = 'FrozenBatchNorm2d'

# --- Pengaturan Transformer ---
hidden_dim = 256
enc_layers = 6
dec_layers = 6
dim_feedforward = 2048
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
enc_n_points = 4
dec_n_points = 4
pre_norm = False
transformer_activation = 'relu'
decoder_sa_type = 'sa'
decoder_module_seq = ['sa', 'ca', 'ffn']

# --- Pengaturan Text Encoder & Fusion ---
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

# --- Pengaturan DeNoising (DN) ---
dn_box_noise_scale = 1.0
dn_label_noise_ratio = 0.5
dn_label_coef = 1.0
dn_bbox_coef = 1.0
embed_init_tgt = True
dn_labelbook_size = 15
dn_scalar = 100

# --- Pengaturan Loss ---
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
matcher_type = 'HungarianMatcher'

# --- Pengaturan Training & Optimizer ---
epochs = 15
lr = 2e-5
lr_backbone = 1e-6
lr_backbone_names = ['backbone.0', 'bert']
lr_linear_proj_mult = 1e-5
lr_linear_proj_names = ['ref_point_head', 'sampling_offsets']
weight_decay = 0.0001
lr_drop = 4
lr_drop_list = [8, 12]
save_checkpoint_interval = 1
clip_max_norm = 0.1
param_dict_type = 'ddetr_in_mmdet'
ddetr_lr_param = False
onecyclelr = False
multi_step_lr = False
frozen_weights = None
backbone_freeze_keywords = None  # <--- Parameter yang hilang sudah ditambahkan
freeze_keywords = ['backbone.0', 'bert'] 

# --- Pengaturan Lainnya ---
max_labels = 50
dilation = False
pdetr3_bbox_embed_diff_each_layer = False
pdetr3_refHW = -1
random_refpoints_xy = False
fix_refpoints_hw = -1
dabdetr_yolo_like_anchor_update = False
dabdetr_deformable_encoder = False
dabdetr_deformable_decoder = False
use_deformable_box_attn = False
box_attn_type = 'roi_align'
dec_layer_number = None
decoder_layer_noise = False
dln_xy_noise = 0.2
dln_hw_noise = 0.2
add_channel_attention = False
add_pos_value = False
two_stage_type = 'standard'
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
dec_pred_bbox_embed_share = True
two_stage_pat_embed = 0
two_stage_add_query_num = 0
two_stage_learn_wh = False
two_stage_default_hw = 0.05
two_stage_keep_all_tokens = False
num_select = 300
nms_iou_threshold = -1
dec_pred_class_embed_share = True
match_unstable_error = True
use_ema = False
ema_decay = 0.9997
ema_epoch = 0
use_detached_boxes_dec_out = False

# --- Daftar Label/Kelas ---
label_list = [
    "Auxiliary", "Base Plate", "Box", "Connection Power Supply",
    "Door Handle Drawer", "Drawer Stopper", "Front Plate", "Handle Drawer",
    "Index Mechanism", "Locking Mechanism", "Mounting Component",
    "Push Button Index Mechanism", "Roda Drawer", "Support Outgoing", "Top Plate"
]