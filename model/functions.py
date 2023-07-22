import torch
from scipy.stats import beta
import numpy as np
from torch import nn
import torch.nn.functional as F
from openTSNE import TSNE
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import cv2
global im_count
im_count = 0
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    beta = np.random.beta(0.1, 0.1, 1)[0]  
    #print(beta)
    #fused_std = style_std * beta + content_std * (1-beta)
    #fused_mean = style_mean * beta + content_mean * (1-beta)  
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)
    #return normalized_feat * fused_std.expand(size) + fused_mean.expand(size)
    
    
    
    
def adaptive_instance_normalization_(content_feat, style_mean, style_std):
    #assert (content_feat.size()[:2] == style_feat.size()[:2])
    N, C, H, W = content_feat.shape
    size = content_feat.size()
    content_mean, content_std = calc_mean_std(content_feat)
    #print(content_mean)
    normalized_feat = (content_feat - content_mean.expand(
        size))/content_std.expand(size)
    #print(beta)
    #fused_std = style_std * beta + content_std * (1-beta)
    #fused_mean = style_mean * beta + content_mean * (1-beta)  
    
    style_std = style_std * 0.5 + content_std * 0.5
    style_mean = style_mean * 0.5 + content_mean * 0.5
    
    
    #trans_feats = normalized_feat * style_std.expand(size) + style_mean.expand(size)
    
    
    #print((normalized_feat * style_std.expand(size)).view(N,C,-1).mean(2))
    #print(style_mean)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)
    #return normalized_feat * fused_std.expand(size) + fused_mean.expand(size)
    
    
    


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())

    
def kd_loss(scores, target_scores, T=0.07):
    """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].

    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
    'Hyperparameter': temperature"""
    #print(torch.max(scores, dim=1)[1], torch.max(target_scores, dim=1)[1])
    
    
    device = scores.device
    log_scores_norm = F.log_softmax(scores / T, dim=1)
    targets_norm = F.softmax(target_scores / T, dim=1)

    # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
    #n = scores.size(1)
    #if n > target_scores.size(1):
    #    n_batch = scores.size(0)
    #    zeros_to_add = torch.zeros(n_batch, n - target_scores.size(1))
    #    zeros_to_add = zeros_to_add.to(device)
    #    targets_norm = torch.cat([targets_norm.detach(), zeros_to_add], dim=1)

    # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
    
    KD_loss_unnorm = -(targets_norm * log_scores_norm)
    #print(KD_loss_unnorm.max(1))
    KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)  # --> sum over classes
    KD_loss_unnorm = KD_loss_unnorm.mean()  # --> average over batch
    

    
    #print(KD_loss_unnorm)
    # normalize
    KD_loss = KD_loss_unnorm

    return KD_loss
    
    
def visualization(feat, im_):
    global im_count
    if im_count >= 50:
        im_count = 0
    background = np.zeros((256,128,3))
    N, H, W = feat.shape
    
    
    coff_attns = feat[0]
    im_ = im_[0]
    im_mean = [0.486, 0.459, 0.408]
    im_std = [0.229, 0.224, 0.225]
    im_std = [0.229, 0.224, 0.225]
    im = im_.cpu().numpy()  # C,H,W
    im = im.transpose(1,2,0)
    h, w, _ = im.shape
    im = im * np.array(im_std).astype(float)
    im = im + np.array(im_mean)
    im = im * 255.
    im = np.clip(im,0,255)
    coff_attns = torch.abs(coff_attns)
    coff = coff_attns.data.cpu().numpy().reshape((feat.shape[1],feat.shape[2]))
    coff =  (coff - coff.min()) / (coff.max() - coff.min())
    coff = np.clip(coff, 0, 1)
    coff = np.uint8(255 * coff)
    heatmap = cv2.applyColorMap(cv2.resize(coff, (w, h)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + im * 0.5
    background = result
    cv2.imwrite('attn_maps_1/'+str(im_count%50)+'.jpg', background)
    im_count += 1
    return 
    
    
'''
def sim_loss(trans_feat, x):
    N, C, H, W = x.shape
    trans_pooled = trans_feat.view(N,C,H*W).mean(2)
    x_pooled = x.view(N,C,H*W).mean(2).float()
    trans_pooled = F.normalize(trans_pooled, dim=1)
    
    #print(trans_pooled.dtype, x_pooled.dtype)
    #x_pooled = F.normalize(x_pooled, dim=1)
    #dist = torch.einsum('nc,cm->nm', [trans_pooled, trans_pooled.permute(1,0)])
    dist = torch.einsum('nc,cm->nm', [trans_pooled, x_pooled.permute(1,0)])
    #dist /= 0.07
    #print(dist)
    #dist = F.softmax(dist, dim=1)
    dist = F.sigmoid(dist)
    print(dist[0,:])
    #print(dist)
    #label = torch.tensor([i for i in range(N)]).cuda().long()
    label = torch.zeros((N,N)).cuda().float()
    for i in range(N):
        label[i,int(i/4)*4:(int(i/4)+1)*4] = 1
    #sim = F.cross_entropy(dist, label)
    sim = F.binary_cross_entropy(dist, label)
    #print(sim)
    return sim
'''
def sim_loss(trans_feat, x):
    N, C, H, W = x.shape
    trans_pooled = trans_feat.view(N,C,H*W).mean(2)
    x_pooled = x.view(N,C,H*W).mean(2).float()
    #label = torch.zeros((N,N)).cuda().float()
    #for i in range(N):
    #    label[i,int(i/4)*4:(int(i/4)+1)*4] = int(i/4)
    label = torch.zeros(N).cuda().float()
    for i in range(N):
        label[int(i/4)*4:(int(i/4)+1)*4] = int(i/4)
    sim = triplet_loss_1(trans_pooled, x_pooled, label, 0, False, True)
    return sim



'''    
def sim_loss(trans_feat, x):
    N, C, H, W = x.shape
    trans_pooled = trans_feat.view(N,C,H*W).mean(2)
    x_pooled = x.view(N,C,H*W).mean(2).float()
    trans_pooled = F.normalize(trans_pooled, dim=1)
    
    #print(trans_pooled.dtype, x_pooled.dtype)
    #x_pooled = F.normalize(x_pooled, dim=1)
    #dist = torch.einsum('nc,cm->nm', [trans_pooled, trans_pooled.permute(1,0)])
    dist = torch.einsum('nc,cm->nm', [trans_pooled, x_pooled.permute(1,0)])
    dist /= 0.07
    #print(dist)
    dist = F.softmax(dist, dim=1)
    print(dist[0,:])
    #print(dist)
    label = torch.tensor([i for i in range(N)]).cuda().long()
    sim = F.cross_entropy(dist, label)
    #print(sim)
    return sim 
''' 
    
def tsne_drawer(content_feat, trans_x1, name):
    

    N = content_feat.shape[0]
    feat_1 = content_feat.cpu()
    feat_2 = trans_x1.cpu()
    fig = plt.figure()
    if len(feat_1.shape)==4:
        N, C, H, W = feat_1.shape
        feat_1 = feat_1.view(N, C, -1).mean(2)
        feat_2 = feat_2.view(N, C, -1).mean(2)
        
        #print(feat_1[0]-feat_1[1], feat_2[0]-feat_2[1])
        
    feats_ = torch.cat([feat_1, feat_2],dim=0).view(2*N,-1).data.numpy()
    length = N
    x_2d = TSNE(perplexity=10).fit(feats_)
    
    x_1 = x_2d[:length, 0]
    y_1 = x_2d[:length, 1]
    
    x_2 = x_2d[length:length*2, 0]
    y_2 = x_2d[length:length*2, 1]
    
    
    size = 1
    plt.scatter(x_1, y_1, marker='o', s=size, c='g')
    plt.scatter(x_2, y_2, marker='^', s=size, c='r')
    global counter
    counter = 0
    
   
    plt.savefig(name)
    counter+=1
    if counter>50:
        counter=0
    plt.close()
    return

    
    
    


def tsne_drawer_(feat_1, feat_2, feat_3, feat_4, feat_5):

    
    feat_1 = feat_1.cpu()
    feat_2 = feat_2.cpu()
    feat_3 = feat_3.cpu()
    feat_4 = feat_4.cpu()
    feat_5 = feat_5.cpu()
    
    if len(feat_1.shape)==4:
        N, C, H, W = feat_1.shape
        feat_1 = feat_1.view(N, C, -1).mean(2)
        feat_2 = feat_2.view(N, C, -1).mean(2)
        feat_3 = feat_3.view(N, C, -1).mean(2)
        feat_4 = feat_4.view(N, C, -1).mean(2)
        feat_5 = feat_5.view(N, C, -1).mean(2)
    
    N = feat_1.shape[0]
    feats_ = torch.cat([feat_1, feat_2, feat_3, feat_4, feat_5],dim=0).view(5*N,-1).data.numpy()
    
    length = N
    print(feat_1.shape, feat_2.shape, feat_3.shape, feat_4.shape, feat_5.shape)
    print('###TSNE Drawing###')
    print(feats_.shape)
    x_2d = TSNE(perplexity=10).fit(feats_)
    
    x_1 = x_2d[:length, 0]
    y_1 = x_2d[:length, 1]
    x_2 = x_2d[length:length*2, 0]
    y_2 = x_2d[length:length*2, 1]
    
    x_3 = x_2d[length*2:length*3, 0]
    y_3 = x_2d[length*2:length*3, 1]
    
    x_4 = x_2d[length*3:length*4, 0]
    y_4 = x_2d[length*3:length*4, 1]
    
    x_5 = x_2d[length*4:length*5, 0]
    y_5 = x_2d[length*4:length*5, 1]
    
    
    
    print('done1')
    size = 1
    plt.scatter(x_1, y_1, marker='o', s=size, c='r', label="market")
    plt.scatter(x_2, y_2, marker='^', s=size, c='g', label="duke")
    plt.scatter(x_3, y_3, marker='x', s=size, c='b', label="cuhk03")
    plt.scatter(x_4, y_4, marker='*', s=size, c='y', label="sysu")
    plt.scatter(x_5, y_5, marker='s', s=size, c='k', label="cuhk02")
    print('done2')
    #plt.scatter(x_3, y_3, marker='s', s=size, label="MSMT")
    #plt.scatter(x_4, y_4, marker='D', s=size, label="Cuhk03")
    plt.savefig('check.jpg')
    
    print('done3')
    plt.close(0) 
    
    
    
    
    
    