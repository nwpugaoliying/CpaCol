
## test
python test.py --config_file configs/OCC_Duke/vit_transreid_stride.yml MODEL.DIST_TRAIN False MODEL.DEVICE_ID "('0')" \
MODEL.USE_4D True INPUT.OCC_PTH '../occluded_patch/Duke/' TEST.WEIGHT './logs/occ_Duke/stage_2/transformer_120.pth' \
TEST.NECK_FEAT 'after' MODEL.STAGE_1_MODEL './logs/occ_Duke/stage_1/transformer_best.pth'

