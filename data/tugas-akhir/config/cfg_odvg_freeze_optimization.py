# --------------------
# Dataset & Augmentasi
# --------------------
data_aug_scales = [480, 512, 544, 576, 608, 640]
data_aug_max_size = 1024                     # turunin dari 1333 biar stabil
data_aug_scales2_resize = [400, 500, 600]
data_aug_scales2_crop = [384, 600]
data_aug_scale_overlap = None

# --------------------
# Batch & Training
# --------------------
batch_size = 4
grad_accum_steps = 2                         # accumulate grad → batch efektif = 8
epochs = 60                                  # dari 15 → 60 biar cukup waktu
lr = 1e-4                                    # dari 2e-4 → lebih stabil
lr_backbone = 1e-5                           # backbone learning rate
lr_linear_proj_mult = 1e-5
weight_decay = 0.0001
lr_drop_list = [30, 50]                      # lr drop 2x selama training
clip_max_norm = 0.1

# --------------------
# Model
# --------------------
modelname = 'groundingdino'
backbone = 'swin_T_224_1k'                   # bisa dinaikkan ke swin_S kalau GPU kuat
position_embedding = 'sine'
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
enc_layers = 6
dec_layers = 6
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.1                                # kasih sedikit dropout
nheads = 8

# --------------------
# Queries
# --------------------
num_queries = 300                            # dari 900 → lebih kecil, lebih stabil
num_select = 100
query_dim = 4
num_patterns = 0
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4

# --------------------
# Two-Stage & Transformer
# --------------------
two_stage_type = 'standard'
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
transformer_activation = 'relu'
dec_pred_bbox_embed_share = True

# --------------------
# DN (denoising)
# --------------------
dn_box_noise_scale = 1.0
dn_label_noise_ratio = 0.5
dn_label_coef = 1.0
dn_bbox_coef = 1.0
dn_labelbook_size = 15
dn_scalar = 100
embed_init_tgt = True

# --------------------
# Text encoder
# --------------------
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

# --------------------
# Freeze strategy
# --------------------
freeze_keywords = ['backbone.0','bert']      # awalnya freeze backbone & BERT
# setelah 15 epoch, unfreeze backbone biar adaptif

lr_backbone_names = ['backbone.0', 'bert']
lr_linear_proj_names = ['ref_point_head', 'sampling_offsets']

# --------------------
# Loss & Matcher
# --------------------
aux_loss = True
set_cost_class = 1.0
set_cost_bbox = 5.0
set_cost_giou = 2.0
cls_loss_coef = 4.0                          # dari 2.0 → 4.0, bantu klasifikasi
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
enc_loss_coef = 1.0
interm_loss_coef = 1.0
mask_loss_coef = 1.0
dice_loss_coef = 1.0
focal_alpha = 0.25
focal_gamma = 2.0

matcher_type = 'HungarianMatcher'
decoder_module_seq = ['sa', 'ca', 'ffn']
decoder_sa_type = 'sa'
nms_iou_threshold = -1
dec_pred_class_embed_share = True

# --------------------
# Misc
# --------------------
save_checkpoint_interval = 1
match_unstable_error = True
use_ema = False
ema_decay = 0.9997
ema_epoch = 0
use_detached_boxes_dec_out = False
use_coco_eval = False
batch_norm_type = 'FrozenBatchNorm2d'
masks = False

# --------------------
# Label List (15 kelas)
# --------------------
label_list = [
    "Auxiliary", "Base Plate", "Box", "Connection Power Supply", 
    "Door Handle Drawer", "Drawer Stopper", "Front Plate", "Handle Drawer",
    "Index Mechanism", "Locking Mechanism", "Mounting Component", 
    "Push Button Index Mechanism", "Roda Drawer", "Support Outgoing", "Top Plate"
]
