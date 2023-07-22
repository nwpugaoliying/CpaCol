# Train_Stage_One
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 60045 train.py \
--config_file configs/OCC_ReID/vit_transreid_stride.yml MODEL.DIST_TRAIN True \
INPUT.OCC_PTH '../occluded_patch/Market/' SOLVER.EVAL_PERIOD 10 INPUT.CJ_PROB 0.0 SOLVER.CHECKPOINT_PERIOD 30 \
SOLVER.IMS_PER_BATCH 64 SOLVER.MAX_EPOCHS 120 SOLVER.SEED 123 \
MODEL.USE_EMBED True MODEL.USE_FB_PART True MODEL.USE_4D False OUTPUT_DIR './logs/occ_ReID/stage_1/' 


# Train_Stage_Two
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 60057 train.py \
--config_file configs/OCC_ReID/vit_transreid_stride.yml MODEL.DIST_TRAIN True \
INPUT.OCC_PTH '../occluded_patch/Market/' MODEL.STAGE_1_MODEL './logs/occ_ReID/stage_1/transformer_best.pth' \
MODEL.USE_4D True SOLVER.EVAL_PERIOD 30 INPUT.CJ_PROB 0.0 SOLVER.CHECKPOINT_PERIOD 30 \
SOLVER.BASE_LR 0.001 SOLVER.MAX_EPOCHS 120 SOLVER.IMS_PER_BATCH 128 \
MODEL.IF_DIST_CORR True OUTPUT_DIR './logs/occ_ReID/stage_2/' 
