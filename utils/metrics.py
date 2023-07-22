import torch
import numpy as np
import os
from utils.reranking import re_ranking
import time
import pickle

import pickle as pickle
import os.path as osp

def load_pickle(path):
  """Check and load pickle object.
  According to this post: https://stackoverflow.com/a/41733927, cPickle and 
  disabling garbage collector helps with loading speed."""
  assert osp.exists(path)
  # gc.disable()
  with open(path, 'rb') as f:
    ret = pickle.load(f)
  # gc.enable()
  return ret
  




def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def pair_wise_distance_compute(query_feats, gallery_feats, func, batch_size):
    with torch.no_grad():
        num_q = query_feats.shape[0]
        num_g = gallery_feats.shape[0]
        dist = torch.zeros((num_q, num_g)).cuda()
        # query_feats = torch.tensor(query_feats)
        # gallery_feats = torch.tensor(gallery_feats)
        # query_feats = query_feats.clone().detach()
        # gallery_feats = gallery_feats.clone().detach()
        for i in range(num_q):
            feat_q_ = query_feats[i].cuda()
            for k in range(0, num_g, batch_size):
                k2 = min(k + batch_size, num_g)
                feats_g = gallery_feats[k:k2].cuda()
                feat_q = feat_q_.unsqueeze(0).expand(feats_g.shape)
                distance = func(feat_q, feats_g, with_matcher=True)
                dist[i, k:k2] = distance   
        dist = dist.cpu().data.numpy()
    return dist
    
    
    
    
class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.feats_4d = []
        
        
        

    def update(self, output):  # called once for each batch
        feat, feat_4d, pid, camid = output
        self.feats.append(feat.cpu())
        self.feats_4d.append(feat_4d.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self, use_4d=False, func_=None):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
           
        
            
        feats_4d = torch.cat(self.feats_4d, dim=0)    
        if use_4d:
            partition = {}
            distmat_g = distmat
            distmat_4d = pair_wise_distance_compute(feats_4d[:self.num_query], feats_4d[self.num_query:], func_, 256)
            partition['distmat_g'] = distmat_g
            partition['distmat_4d'] = distmat_4d

            partition['q_pids'] = q_pids
            partition['g_pids'] = g_pids
            partition['q_camids'] =   q_camids
            partition['g_camids'] =   g_camids

            
            q_num, num_g = distmat_g.shape 
            mean_val = distmat_g.mean(1).reshape(q_num,1) 
            std_val = distmat_g.std(1).reshape(q_num,1) + 1e-7

            distmat_g = (distmat_g - mean_val) / std_val


            mean_val = distmat_4d.mean(1).reshape(q_num,1) 
            std_val = distmat_4d.std(1).reshape(q_num,1) + 1e-7

            distmat_4d = (distmat_4d - mean_val) / std_val
            ###
            for fuse in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                distmat = distmat_g + distmat_4d * fuse

                cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
                print("Validation Results, fuse parameter: ", fuse)
                print("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    print("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))



            distmat = distmat_g + distmat_4d * 0.3

            if self.num_query == 1000:
                file_name = './logs/occ_ReID/distance.pkl'
            else:
                file_name = './logs/occ_Duke/distance.pkl'
            
            with open(file_name, 'wb') as f:
                pickle.dump(partition, f)
            
        # distmat = distmat_g + distmat_4d * 0.3
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf



