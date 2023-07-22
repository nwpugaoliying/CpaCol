import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from .backbones.cvt_pytorch import cvt_small_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from .functions import visualization
import os
from PIL import Image, ImageFile
import torchvision.transforms.functional as TF
import random
import copy
import cv2
import torch.nn.functional as F
from .base.matcher import Cor_Learner
from .base.correlation import Correlation


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        
        

        
    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE  # softmax
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label= None, view_label=None):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)

        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.use_4d = cfg.MODEL.USE_4D
        self.use_embed = cfg.MODEL.USE_EMBED
        self.occ_reid = True if cfg.DATASETS.NAMES=='occ_reid' else False
        self.embed_layer = cfg.MODEL.EMBED_LAYER
        self.embed_layer_list = cfg.MODEL.EMBED_LAYER_LIST
        print('self.embed_layer:', self.embed_layer, self.embed_layer_list)
        
        
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, 
            local_feature=cfg.MODEL.USE_EMBED, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, 
            drop_path_rate=cfg.MODEL.DROP_PATH, embed_layer=self.embed_layer, embed_layer_list=self.embed_layer_list,
            if_fb_part=cfg.MODEL.USE_FB_PART, num_part=cfg.MODEL.FB_PART_NUM)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm

        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE  # softmax
        print('ID loss:',self.ID_LOSS_TYPE)
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)
        
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        if self.use_embed:
            self.classifier_emb_layers = nn.ModuleList([nn.Linear(self.in_planes, self.num_classes, bias=False) for _ in range(self.embed_layer)])
            
            self.bottleneck_emb_layers = nn.ModuleList([nn.BatchNorm1d(self.in_planes) for _ in range(self.embed_layer)])

        if self.use_4d:
            self.conv2ds = nn.ModuleList([nn.Conv2d(768, 256, kernel_size=3, padding=1, bias=False) for _ in range(self.embed_layer)])
            
            self.norm2ds = nn.ModuleList([nn.BatchNorm2d(256) for _ in range(self.embed_layer)])


        for i in range(self.embed_layer):
            if self.use_embed:
                self.classifier_emb_layers[i].apply(weights_init_classifier)  

                self.bottleneck_emb_layers[i].bias.requires_grad_(False)
                self.bottleneck_emb_layers[i].apply(weights_init_kaiming)

            if self.use_4d:
                self.conv2ds[i].apply(weights_init_kaiming)
                self.norm2ds[i].apply(weights_init_kaiming)


        self.occlusion_path = cfg.INPUT.OCC_PTH
        self.occlusion_bank = []
        filelist = os.listdir(self.occlusion_path)
        for file in filelist:
            img_path = os.path.join(self.occlusion_path, file) 
            img = Image.open(img_path).convert('RGB')
            img = TF.to_tensor(img)
            self.occlusion_bank.append(img)
        
        if self.use_4d:
            self.matcher = Cor_Learner(inch = [0, self.embed_layer])
            self.scales = [1.0 for _ in range(self.embed_layer)]
            print('self.scales =', self.scales)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange
        
        self.register_buffer('pixel_mean', torch.Tensor(cfg.INPUT.PIXEL_MEAN).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(cfg.INPUT.PIXEL_STD).view(1, -1, 1, 1), False)
        
        if self.use_4d:
            assert os.path.exists(cfg.MODEL.STAGE_1_MODEL)
            self.load_param_finetune(cfg.MODEL.STAGE_1_MODEL)
        self.if_dist = cfg.MODEL.IF_DIST_CORR
        print('MODEL.IF_DIST_CORR:', self.if_dist)
        
        
        
        
    def forward(self, x, label=None, cam_label= None, view_label=None, epoch=0):  # label is unused if self.cos_layer == 'no'
        inputs = x
        if self.training:
            x = self.insert_occlusion(x)
        else:
            N,C,H,W = x.shape
        x = self.preprocess_image(x)
        
        if self.use_4d:
            with torch.no_grad():
                self.base.eval()
                global_feat, features, fg_feats_list, bg_feats_list, emb_feats_list, feats_4D = self.base(x, cam_label=cam_label, view_label=view_label, epoch=epoch, occ_reid = self.occ_reid)
        else:
            global_feat, features, fg_feats_list, bg_feats_list, emb_feats_list, feats_4D = self.base(x, cam_label=cam_label, view_label=view_label, epoch=epoch, occ_reid = self.occ_reid)
        

        if self.use_4d:
            # obtain feature maps for 4D sim. calculate
            feats_4D = feats_4D[:,:,1:]
            N, H, L, C = feats_4D.shape  # H=3
            feats_4D = feats_4D.permute(0,1,3,2).view(N,H,C,22,11)
        

            feats_4D_ = []

            for i in range(H):
                feats_idx = self.conv2ds[i](feats_4D[:,i])
                feats_idx = self.norm2ds[i](feats_idx)
                feats_4D_.append(feats_idx.unsqueeze(1))
            feats_4D = torch.cat(feats_4D_, dim=1)

            if self.training:
                if epoch > 20:
                    pos_idx, neg_idx = self.hard_sample(global_feat, label)
                else:
                    pos_idx, neg_idx = self.random_sample(label)

                correlation_pos = self.compute_correlation(feats_4D, feats_4D[pos_idx])
                correlation_neg = self.compute_correlation(feats_4D, feats_4D[neg_idx])

                score = self.matcher(torch.cat([correlation_pos, correlation_neg], dim=0), if_dist=self.if_dist)
                N = feats_4D.shape[0]
                Fine_Verify_list = [score[:N], score[N:]]
            else:
                Fine_Verify_list = None
        else:
            Fine_Verify_list = None


        if self.use_embed:
            emb_score_list = []
            for i in range(self.embed_layer):
                emb_feats_bn = self.bottleneck_emb_layers[i](emb_feats_list[i])
                emb_score = self.classifier_emb_layers[i](emb_feats_bn)
                emb_score_list.append(emb_score)
        else:
            emb_score_list = None


        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        # token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        
    
        # lf_1
        b2_feats = self.b2(features)[:,1:]
        
        N, L, C = b2_feats.shape
        
        local_feat_1 = b2_feats[:, :patch_length].mean(1)
        local_feat_2 = b2_feats[:, patch_length:patch_length*2].mean(1)
        local_feat_3 = b2_feats[:, patch_length*2:patch_length*3].mean(1)
        local_feat_4 = b2_feats[:, patch_length*3:patch_length*4].mean(1)
        
        
        
        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            

            return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                        cls_score_4
                        ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
                            local_feat_4], [fg_feats_list, bg_feats_list], emb_score_list, Fine_Verify_list  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1), feats_4D
            else:
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1), feats_4D
                    
                    
    def compute_correlation(self, query, gallery, norm=True, with_matcher=False):
        
        correlation = Correlation.build_crossscale_correlation(query, gallery, self.scales) # , self.conv2ds, self.norm2ds
        if with_matcher:
            with torch.no_grad():
                self.matcher.eval()
                score = self.matcher(correlation)
                self.matcher.train()
            return score
        return correlation
        
    def random_sample(self, targets):
        N = targets.shape[0]
        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).int().data.cpu().numpy()
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).int().data.cpu().numpy()

        import numpy as np
        import random
        pos_idx = []
        for i in range(N):
            cand = np.where(is_pos[i]==1)[0]
            random.shuffle(cand)
            idx = cand[0]
            pos_idx.append(idx)
        pos_idx = torch.tensor(pos_idx).cuda()    
        neg_idx = []
        for i in range(N):
            cand = np.where(is_neg[i]==1)[0]
            random.shuffle(cand)
            # random.shuffle(list(cand))
            idx = cand[0]
            neg_idx.append(idx)
        neg_idx = torch.tensor(neg_idx).cuda()       
        return pos_idx, neg_idx
        
    def hard_sample(self, features, targets):
        N,C = features.shape
        feats = features
        dist = euclidean_dist(feats, feats)
        is_pos = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()
        
        pos_idx = torch.max(dist * is_pos, dim=1)[1]
        neg_idx = torch.min((dist * is_neg) + is_pos * 9999, dim=1)[1]
        return pos_idx, neg_idx
        
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path, map_location=torch.device('cpu'))
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))
        
    def preprocess_image(self, batched_inputs):
        images = batched_inputs
        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images
        
        
    def insert_occlusion(self, x):
        N, _,H, W = x.shape
        
        occlusion_bank_ = copy.deepcopy(self.occlusion_bank)
        random.shuffle(occlusion_bank_)
        for i in range(x.shape[0]):
            # idx = random.randint(0, len(self.occlusion_bank)-1)
            # occlusion = self.occlusion_bank[idx]
            if torch.rand(1)>0.3:
                continue
            elif len(occlusion_bank_)==0:
                continue
                
            occlusion = occlusion_bank_.pop()  
            # print(x.shape, occlusion.shape)
            min_h = int(occlusion.shape[1]/16)
            min_w = int(occlusion.shape[2]/16)
            # randn = random.randint(3, min_) if min_>3 else min_
            _, H_, W_ = occlusion.shape
            # print(H_, H_-16*min_, W_, W_-16*min_)
            stride_h = 256 if int(16*min_h) > 256 else int(16*min_h)
            stride_w = 128 if int(16*min_w) > 128 else int(16*min_w)
            
            
            H_start = 0 if int(H_-stride_h)==0 else random.randint(0, int(H_-stride_h))
            W_start = 0 if int(W_-stride_w)==0 else random.randint(0, int(W_-stride_w))
            
            occlusion = occlusion[:,H_start:stride_h+H_start, W_start:stride_w+W_start]    
            
            
            # print(occlusion.shape)
            _, H_, W_ = x[i].shape
            H_start = 0 if int(H_-stride_h)==0 else random.randint(0, int(H_-stride_h))
            W_start = 0 if int(W_-stride_w)==0 else random.randint(0, int(W_-stride_w))
            x[i,:,H_start:stride_h+H_start, W_start:stride_w+W_start] = occlusion
        
        
        
        
        
        return x
    
    
    
    
    

__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID,
    'cvt_small_patch16_224_TransReID': cvt_small_patch16_224_TransReID
}

def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model
