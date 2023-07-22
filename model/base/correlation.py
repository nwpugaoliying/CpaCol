r""" Provides functions that builds/manipulates correlation tensors """
# Original code: HSNet (https://github.com/juhongm999/hsnet)

import torch
import numpy as np
import torch.nn.functional as F
import math
from torch.nn.functional import interpolate as resize
from .geometry import Geometry

import pdb

class Correlation:
    @classmethod
    def compute_crossscale_correlation(cls, _src_feats, _trg_feats, origin_resolution):
        """ Build 6-dimensional correlation tensor """
        eps = 1e-8

        bsz, ha, wa, hb, wb = origin_resolution

        # Build multiple 4-dimensional correlation tensor
        corr6d = []
        '''
        for i, src_feat in enumerate(_src_feats):
            ch = src_feat.size(1)
            sha, swa = src_feat.size(-2), src_feat.size(-1)
            src_feat = src_feat.view(bsz, ch, -1).transpose(1, 2)
            src_norm = src_feat.norm(p=2, dim=2, keepdim=True)

            for j, trg_feat in enumerate(_trg_feats):
                shb, swb = trg_feat.size(-2), trg_feat.size(-1)
                trg_feat = trg_feat.view(bsz, ch, -1)
                trg_norm = trg_feat.norm(p=2, dim=1, keepdim=True)
                
                # print(corr.max(), corr.min(), i,j)
                
                corr = torch.bmm(src_feat, trg_feat)
                # print(corr.max(), corr.min(), i,j)
                corr_norm = torch.bmm(src_norm, trg_norm) + eps
                corr = corr / corr_norm
                
                correlation = corr.view(bsz, sha, swa, shb, swb).contiguous()
                corr6d.append(correlation)
        '''
        
        for i, (src_feat, trg_feat) in enumerate(zip(_src_feats, _trg_feats)):
        
            ch = src_feat.size(1)
            sha, swa = src_feat.size(-2), src_feat.size(-1)
            src_feat = src_feat.view(bsz, ch, -1).transpose(1, 2)
            src_norm = src_feat.norm(p=2, dim=2, keepdim=True)
            shb, swb = trg_feat.size(-2), trg_feat.size(-1)
            trg_feat = trg_feat.view(bsz, ch, -1)
            trg_norm = trg_feat.norm(p=2, dim=1, keepdim=True)
            corr = torch.bmm(src_feat, trg_feat)

            corr_norm = torch.bmm(src_norm, trg_norm) + eps
            corr = corr / corr_norm
            correlation = corr.view(bsz, sha, swa, shb, swb).contiguous() ## BS * 22 * 11 * 22 * 11

            corr6d.append(correlation)
        
        
        # Resize the spatial sizes of the 4D tensors to the same size
        for idx, correlation in enumerate(corr6d):
            corr6d[idx] = Geometry.interpolate4d(correlation, [ha, wa, hb, wb])
            
        # Build 6-dimensional correlation tensor
        corr6d = torch.stack(corr6d).view(len(_src_feats), bsz, ha, wa, hb, wb).transpose(0,1)
        # print('correlation, corr6d:', corr6d.shape)  # ([64, 3, 22, 11, 22, 11])
        return corr6d.clamp(min=0)

    @classmethod
    def build_crossscale_correlation(cls, query_feats, key_feats, scales, conv2ds=None, norm2ds=None):
        eps = 1e-8

        bsz, _, _, hq, wq = query_feats.size()
        bsz, _, _, hk, wk = key_feats.size()
        # Construct feature pairs with multiple scales
        _query_feats_scalewise = []
        _key_feats_scalewise = []
        # for i, (scale, conv, norm) in enumerate(zip(scales, conv2ds, norm2ds)):
        for i, (scale) in enumerate(zip(scales)):
            # shq = round(hq * math.sqrt(scale))
            # swq = round(wq * math.sqrt(scale))
            # shk = round(hk * math.sqrt(scale))
            # swk = round(wk * math.sqrt(scale))

            # _query_feats = resize(query_feats, (shq, swq), mode='nearest')
            # _key_feats = resize(key_feats, (shk, swk), mode='nearest')
            _query_feats = query_feats[:, i]
            _key_feats = key_feats[:, i]
            
            '''
            _query_feats = conv(_query_feats)
            _query_feats = norm(_query_feats)
            _key_feats = conv(_key_feats)
            _key_feats = norm(_key_feats)
            '''
            
            _query_feats_scalewise.append(_query_feats)
            _key_feats_scalewise.append(_key_feats)

        corrs = cls.compute_crossscale_correlation(_query_feats_scalewise, _key_feats_scalewise, (bsz, hq, wq, hk, wk))
        return corrs.contiguous()
