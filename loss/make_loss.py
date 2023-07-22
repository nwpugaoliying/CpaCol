# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .contrastive_loss import SimMinLoss, SimMaxLoss
import torch
def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    sim_min = SimMinLoss(metric='cos').cuda()
    sim_max= SimMaxLoss(metric='cos', alpha=0.05).cuda()
    sim_max_fg= SimMaxLoss(metric='cos', alpha=cfg.SOLVER.FG_SIM_MAX_LOSS_ALPHA).cuda()
    sim_max_local= SimMaxLoss(metric='cos', alpha=cfg.SOLVER.FB_PART_SIM_MAX_LOSS_ALPHA).cuda()

    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, fb_feats, emb_score_list, Fine_Verify_list, target, target_cam, epochs):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':  # off
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    if isinstance(fb_feats, list): 
                        FB_LOSS = sim_max(fb_feats[0]) + sim_max(fb_feats[1]) + sim_min(fb_feats[0], fb_feats[1]) 

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS + \
                               FB_LOSS
                else:

                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    FB_LOSS = 0
                    if isinstance(fb_feats, list) and cfg.MODEL.USE_FG: 
                        ma = target.unsqueeze(0).repeat(target.size(0),1)
                        mb = target.unsqueeze(1).repeat(1,target.size(0))
                        consistent = (ma-mb) == 0

                        loss_weight = 0.0 if epochs<2 else 1.0

                        num_full_feature = cfg.MODEL.EMBED_LAYER

                        for i in range(num_full_feature):
                            FG_loss = sim_max_fg(fb_feats[0][i], consistent) #* consistent.size(0) * 0.25
                            FB_LOSS += (FG_loss + sim_max(fb_feats[1][i]) + sim_min(fb_feats[0][i], fb_feats[1][i])) * loss_weight
                        
                        if cfg.MODEL.USE_FB_PART:
                            FB_PART_LOSS = 0

                            for i in range(num_full_feature, len(fb_feats[0])): ## embedding layer
                                for j in range(len(fb_feats[0][i])): ## batch size 
                                    FB_p_loss = sim_max_local(fb_feats[0][i][j]) + sim_max_local(fb_feats[1][i][j]) + sim_min(fb_feats[0][i][j], fb_feats[1][i][j])
                                    FB_PART_LOSS += FB_p_loss * loss_weight 
                                    
                            FB_LOSS += FB_PART_LOSS / len(fb_feats[0][i]) * cfg.SOLVER.FB_PART_SIM_MAX_LOSS_WEIGHT


                    EMB_ID_LOSS = 0 
                    if cfg.MODEL.USE_EMBED and isinstance(emb_score_list, list):
                        for i in range(len(emb_score_list)):
                            EMB_ID = F.cross_entropy(emb_score_list[i], target)
                            EMB_ID_LOSS += EMB_ID
                        

                    VERIFY_LOSS = 0
                    if isinstance(Fine_Verify_list, list):
                        VERIFY_LOSS = verify_loss(Fine_Verify_list)
                        return VERIFY_LOSS + 0.0 * (ID_LOSS + TRI_LOSS+FB_LOSS+EMB_ID_LOSS)
                    # print("ID_LOSS:", ID_LOSS, 'TRI_LOSS:', TRI_LOSS, 'EMB_ID:', EMB_ID_LOSS, 'FB_LOSS:', FB_LOSS)
                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS + FB_LOSS * 1.0 + EMB_ID_LOSS * 1.0 + VERIFY_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion


def verify_loss(score_list):
    score_pos = score_list[0]
    score_negs = score_list[1]
    sim = score_pos - score_negs
    
    _, indices = sim.sort(descending=False, dim=0)
    
    loss_pos = F.binary_cross_entropy_with_logits(score_pos, torch.zeros(score_pos.shape[0]).cuda(), reduction='none')
    loss_pos = loss_pos.sum(-1)
    loss_neg = F.binary_cross_entropy_with_logits(score_negs, torch.ones(score_negs.shape[0]).cuda(), reduction='none')
    loss_neg = loss_neg.sum(-1)
    loss = loss_pos + loss_neg
    match_loss = loss
    y = score_negs.new().resize_as_(score_negs).fill_(1)        
    loss = F.margin_ranking_loss(score_negs, score_pos, y, margin=0.3)
    
    loss_triplet = loss
    
    return match_loss+loss_triplet