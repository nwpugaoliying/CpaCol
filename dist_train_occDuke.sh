# Train_Stage_One
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port 60130 train.py \
--config_file configs/OCC_Duke/vit_transreid_stride.yml MODEL.DIST_TRAIN True \
INPUT.OCC_PTH '../occluded_patch/Duke/' SOLVER.EVAL_PERIOD 10 SOLVER.CHECKPOINT_PERIOD 30 \
OUTPUT_DIR './logs/occ_Duke/stage_1/' SOLVER.BASE_LR 0.008 SOLVER.MAX_EPOCHS 140 \
MODEL.USE_EMBED True MODEL.USE_FB_PART True MODEL.USE_4D False 


# Train_Stage_Two
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch --nproc_per_node=2 --master_port 60137 train.py \
--config_file configs/OCC_Duke/vit_transreid_stride.yml MODEL.DIST_TRAIN True \
MODEL.STAGE_1_MODEL './logs/occ_Duke/stage_1/transformer_140.pth' \
MODEL.USE_4D True SOLVER.CHECKPOINT_PERIOD 30 SOLVER.EVAL_PERIOD 120 MODEL.IF_DIST_CORR True \
SOLVER.BASE_LR 0.001 SOLVER.IMS_PER_BATCH 128 OUTPUT_DIR './logs/occ_Duke/stage_2/' 

