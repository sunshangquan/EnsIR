import os
import cv2
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

from compute_psnr import compute_metrics

from gmm_torch.gmm import GaussianMixture
def reorganize(imgs_cad, img_gt, bin_width=5):
    # indices = imgs_cad // bin_width
    indices = torch.div(imgs_cad, bin_width, rounding_mode='trunc')
    num_cad = imgs_cad.shape[0]
    num_bin_1d = math.ceil(256 / bin_width)
    data = {}
    if num_cad == 1:
        for i_bin in range(num_bin_1d):
            id_datum_r = (indices[0,0,] == i_bin)
            id_datum_g = (indices[0,1,] == i_bin)
            id_datum_b = (indices[0,2,] == i_bin)
            # print(imgs_cad.shape, id_datum_r.shape, (id_datum_r.sum()))
            datum_r = imgs_cad[:,0][:,id_datum_r]
            datum_g = imgs_cad[:,1][:,id_datum_g]
            datum_b = imgs_cad[:,2][:,id_datum_b]
            gt_r = img_gt[:,0][:,id_datum_r]
            gt_g = img_gt[:,1][:,id_datum_g]
            gt_b = img_gt[:,2][:,id_datum_b]
            data[(i_bin*bin_width)] = [
                torch.cat((gt_r, datum_r), 0), 
                torch.cat((gt_g, datum_g), 0), 
                torch.cat((gt_b, datum_b), 0)
            ] # K+1, N_
    elif num_cad == 2:
        for i_bin in range(num_bin_1d):
            for j_bin in range(num_bin_1d):
                id_datum_r = (indices[0,0,] == i_bin)*(indices[1,0,] == j_bin)
                id_datum_g = (indices[0,1,] == i_bin)*(indices[1,1,] == j_bin)
                id_datum_b = (indices[0,2,] == i_bin)*(indices[1,2,] == j_bin)
                # print(imgs_cad.shape, id_datum_r.shape, (id_datum_r.sum()))
                datum_r = imgs_cad[:,0][:,id_datum_r]
                datum_g = imgs_cad[:,1][:,id_datum_g]
                datum_b = imgs_cad[:,2][:,id_datum_b]
                gt_r = img_gt[:,0][:,id_datum_r]
                gt_g = img_gt[:,1][:,id_datum_g]
                gt_b = img_gt[:,2][:,id_datum_b]
                data[(i_bin*bin_width, j_bin*bin_width)] = [
                    torch.cat((gt_r, datum_r), 0), 
                    torch.cat((gt_g, datum_g), 0), 
                    torch.cat((gt_b, datum_b), 0)
                ] # K+1, N_
    elif num_cad == 3:
        for i_bin in range(num_bin_1d):
            for j_bin in range(num_bin_1d):
                for k_bin in range(num_bin_1d):
                    id_datum_r = (indices[0,0,] == i_bin)*(indices[1,0,] == j_bin)*(indices[2,0,] == k_bin)
                    id_datum_g = (indices[0,1,] == i_bin)*(indices[1,1,] == j_bin)*(indices[2,1,] == k_bin)
                    id_datum_b = (indices[0,2,] == i_bin)*(indices[1,2,] == j_bin)*(indices[2,2,] == k_bin)
                    # print(imgs_cad.shape, id_datum_r.shape, (id_datum_r.sum()))
                    datum_r = imgs_cad[:,0][:,id_datum_r]
                    datum_g = imgs_cad[:,1][:,id_datum_g]
                    datum_b = imgs_cad[:,2][:,id_datum_b]
                    gt_r = img_gt[:,0][:,id_datum_r]
                    gt_g = img_gt[:,1][:,id_datum_g]
                    gt_b = img_gt[:,2][:,id_datum_b]
                    data[(i_bin*bin_width, j_bin*bin_width, k_bin*bin_width)] = [
                        torch.cat((gt_r, datum_r), 0), 
                        torch.cat((gt_g, datum_g), 0), 
                        torch.cat((gt_b, datum_b), 0)
                    ] # K+1, N_
    elif num_cad == 4:
        for i_bin in range(num_bin_1d):
            for j_bin in range(num_bin_1d):
                for k_bin in range(num_bin_1d):
                    for ii_bin in range(num_bin_1d):
                        id_datum_r = (indices[0,0,] == i_bin)*(indices[1,0,] == j_bin)*(indices[2,0,] == k_bin)*(indices[3,0,] == ii_bin)
                        id_datum_g = (indices[0,1,] == i_bin)*(indices[1,1,] == j_bin)*(indices[2,1,] == k_bin)*(indices[3,1,] == ii_bin)
                        id_datum_b = (indices[0,2,] == i_bin)*(indices[1,2,] == j_bin)*(indices[2,2,] == k_bin)*(indices[3,2,] == ii_bin)
                        # print(imgs_cad.shape, id_datum_r.shape, (id_datum_r.sum()))
                        datum_r = imgs_cad[:,0][:,id_datum_r]
                        datum_g = imgs_cad[:,1][:,id_datum_g]
                        datum_b = imgs_cad[:,2][:,id_datum_b]
                        gt_r = img_gt[:,0][:,id_datum_r]
                        gt_g = img_gt[:,1][:,id_datum_g]
                        gt_b = img_gt[:,2][:,id_datum_b]
                        data[(i_bin*bin_width, j_bin*bin_width, k_bin*bin_width, ii_bin*bin_width)] = [
                            torch.cat((gt_r, datum_r), 0), 
                            torch.cat((gt_g, datum_g), 0), 
                            torch.cat((gt_b, datum_b), 0)
                        ] # K+1, N_
    # print(data)
    return data

def pad_zero_one(x, dim=0):
    return x
    # mean_row = x.mean(1, keepdim=True)
    # delta = mean_row.max(dim=dim, keepdim=True)[0] - mean_row.min(dim=dim, keepdim=True)[0]

    # left = (x.min(dim=dim, keepdim=True)[0] - delta).clamp(0,255)
    # right = (x.max(dim=dim, keepdim=True)[0] + delta).clamp(0,255)
    # return torch.cat((left, right, x), dim)

def GMM(pix_gt_r, pix_cand_r, pPi_r_init=None):
    K, N = pix_cand_r.shape
    Data_r = pix_gt_r.transpose(0,1) # N_, 1
    mu_r = pix_cand_r.mean(1, keepdim=True)[None,] # 1, K, 1
    std_r = (pix_cand_r - mu_r[0]).pow(2).mean(1, keepdim=True)[None,...,None]
    model_r = GaussianMixture(K, 1, mu_init=mu_r, var_init=std_r)
    if pPi_r_init is not None:
        model_r.pi.data = pPi_r_init[None,:,None]
    model_r.fit(Data_r, delta=1e-5, n_iter=1000, warm_start=True)
    pPi_r = model_r.pi.data.squeeze()
    return pPi_r

def assign_default_weights(imgs_cad, if_mean=True, mean_in=None):
    K = imgs_cad.shape[0]
    if if_mean:
        weights = torch.ones(K) / K
    else:
        mean = imgs_cad.mean(0, keepdim=True) if mean_in is None else mean_in
        MSE = (imgs_cad - mean).pow(2).mean(1).sqrt()
        # MSE = (imgs_cad - mean).abs().mean(1)
        weights = (1/(1e-10 + MSE)).softmax(0)
    return weights

def learn_gmm(pix_gt_r, pix_cand_r, pPi_prev=None, default_weight=None):
    K, N = pix_cand_r.shape
    
    pix_cand_ = pad_zero_one(pix_cand_r)
    if N == 0:
        pPi_r = assign_default_weights(pix_cand_, if_mean=True) if pPi_prev is None else pPi_prev
    else:
        pPi_r_init = assign_default_weights(pix_cand_, if_mean=False, ) if default_weight is None else default_weight
        if N < 1e2:
            pPi_r = pPi_r_init
        else:
            try:
                pPi_r = GMM(pix_gt_r, pix_cand_, pPi_r_init)
            except:
                # print("???")
                # pPi_r = torch.cat((torch.zeros(2),torch.ones(K) / K), 0)
                pPi_r = pPi_r_init
                    
    return pPi_r

def compute_ensemble_weight(imgs_cad, img_gt, weight_file, bin_width, verbose=True, default_weight=None):
    data = reorganize(imgs_cad, img_gt, bin_width)
    pPi_dict = {}
    pPi_prev = None
    for key in data:
        datum_r, datum_g, datum_b = data[key]
        pix_gt_r = datum_r[:1] # 1, N_
        pix_gt_g = datum_g[:1] # 1, N_
        pix_gt_b = datum_b[:1] # 1, N_
        pix_cand_r = datum_r[1:] # K, N_
        pix_cand_g = datum_g[1:] # K, N_
        pix_cand_b = datum_b[1:] # K, N_
        if verbose:
            print(key, pix_cand_r.shape[1], pix_cand_g.shape[1], pix_cand_b.shape[1])
        # pix_gt_all = torch.cat((pix_gt_r, pix_gt_g, pix_gt_b), 1)
        # pix_cand_all = torch.cat((pix_cand_r, pix_cand_g, pix_cand_b), 1)
        # pPi = learn_gmm(pix_gt_all, pix_cand_all)
        pPi_r = learn_gmm(pix_gt_r, pix_cand_r, pPi_prev, default_weight=default_weight)
        pPi_prev = pPi_r
        pPi_g = learn_gmm(pix_gt_g, pix_cand_g, pPi_prev, default_weight=default_weight)
        pPi_prev = pPi_g
        pPi_b = learn_gmm(pix_gt_b, pix_cand_b, pPi_prev, default_weight=default_weight)
        pPi_prev = pPi_b
        pPi_dict[key] = [pPi_r, pPi_g, pPi_b]
        
        if verbose:
            print(key, pPi_dict[key], pix_cand_r.shape[1], pix_cand_g.shape[1], pix_cand_b.shape[1])
    save_dict(pPi_dict, weight_file)
    if verbose:
        print(pPi_dict)
    return pPi_dict

def save_dict(pPi_dict, weight_file='weight.pth'):
    torch.save(pPi_dict, weight_file)

def read_dict(weight_file='weight.pth'):
    pPi_dict = torch.load(weight_file)
    return pPi_dict
def ensemble_single(pPi_dict, imgs_cad, bin_width=5, return_weight=False):
    K, C, W, H = imgs_cad.shape
    img_ens = torch.zeros((1,C,W,H))
    wei_ens = torch.zeros((3,C,W,H))
    # indices = imgs_cad // bin_width
    indices = torch.div(imgs_cad, bin_width, rounding_mode='trunc')
    num_cad = imgs_cad.shape[0]
    num_bin_1d = math.ceil(256 / bin_width)
    if num_cad == 1:
        for i_bin in range(num_bin_1d):
            id_datum_r = (indices[0,0] == i_bin)
            id_datum_g = (indices[0,1] == i_bin)
            id_datum_b = (indices[0,2] == i_bin)
            datum_r = imgs_cad[:,0][:,id_datum_r] # K, N_
            datum_g = imgs_cad[:,1][:,id_datum_g] # K, N_
            datum_b = imgs_cad[:,2][:,id_datum_b] # K, N_
            pPi_r, pPi_g, pPi_b = pPi_dict[(i_bin*bin_width)]
            # print(datum_r.shape, pPi)
            img_ens[:,0][:,id_datum_r] = (pad_zero_one(datum_r) * pPi_r[:,None]).sum(0, keepdim=True)
            img_ens[:,1][:,id_datum_g] = (pad_zero_one(datum_g) * pPi_g[:,None]).sum(0, keepdim=True)
            img_ens[:,2][:,id_datum_b] = (pad_zero_one(datum_b) * pPi_b[:,None]).sum(0, keepdim=True)
    elif num_cad == 2:
        for i_bin in range(num_bin_1d):
            for j_bin in range(num_bin_1d):
                id_datum_r = (indices[0,0] == i_bin)*(indices[1,0] == j_bin)
                id_datum_g = (indices[0,1] == i_bin)*(indices[1,1] == j_bin)
                id_datum_b = (indices[0,2] == i_bin)*(indices[1,2] == j_bin)
                datum_r = imgs_cad[:,0][:,id_datum_r] # K, N_
                datum_g = imgs_cad[:,1][:,id_datum_g] # K, N_
                datum_b = imgs_cad[:,2][:,id_datum_b] # K, N_
                pPi_r, pPi_g, pPi_b = pPi_dict[(i_bin*bin_width, j_bin*bin_width)]
                # print(datum_r.shape, pPi)
                img_ens[:,0][:,id_datum_r] = (pad_zero_one(datum_r) * pPi_r[:,None]).sum(0, keepdim=True)
                img_ens[:,1][:,id_datum_g] = (pad_zero_one(datum_g) * pPi_g[:,None]).sum(0, keepdim=True)
                img_ens[:,2][:,id_datum_b] = (pad_zero_one(datum_b) * pPi_b[:,None]).sum(0, keepdim=True)
    elif num_cad == 3:
        for i_bin in range(num_bin_1d):
            for j_bin in range(num_bin_1d):
                for k_bin in range(num_bin_1d):
                    id_datum_r = (indices[0,0] == i_bin)*(indices[1,0] == j_bin)*(indices[2,0] == k_bin)
                    id_datum_g = (indices[0,1] == i_bin)*(indices[1,1] == j_bin)*(indices[2,1] == k_bin)
                    id_datum_b = (indices[0,2] == i_bin)*(indices[1,2] == j_bin)*(indices[2,2] == k_bin)
                    datum_r = imgs_cad[:,0][:,id_datum_r] # K, N_
                    datum_g = imgs_cad[:,1][:,id_datum_g] # K, N_
                    datum_b = imgs_cad[:,2][:,id_datum_b] # K, N_
                    pPi_r, pPi_g, pPi_b = pPi_dict[(i_bin*bin_width, j_bin*bin_width, k_bin*bin_width)]
                    # print(datum_r.shape, pPi)
                    img_ens[:,0][:,id_datum_r] = (pad_zero_one(datum_r) * pPi_r[:,None]).sum(0, keepdim=True)
                    img_ens[:,1][:,id_datum_g] = (pad_zero_one(datum_g) * pPi_g[:,None]).sum(0, keepdim=True)
                    img_ens[:,2][:,id_datum_b] = (pad_zero_one(datum_b) * pPi_b[:,None]).sum(0, keepdim=True)
                    if return_weight:
                        wei_ens[:,0][:,id_datum_r] = pPi_r[:,None]
                        wei_ens[:,1][:,id_datum_g] = pPi_g[:,None]
                        wei_ens[:,2][:,id_datum_b] = pPi_b[:,None]
    elif num_cad == 4:
        for i_bin in range(num_bin_1d):
            for j_bin in range(num_bin_1d):
                for k_bin in range(num_bin_1d):
                    for ii_bin in range(num_bin_1d):
                        id_datum_r = (indices[0,0] == i_bin)*(indices[1,0] == j_bin)*(indices[2,0] == k_bin)*(indices[3,0] == ii_bin)
                        id_datum_g = (indices[0,1] == i_bin)*(indices[1,1] == j_bin)*(indices[2,1] == k_bin)*(indices[3,1] == ii_bin)
                        id_datum_b = (indices[0,2] == i_bin)*(indices[1,2] == j_bin)*(indices[2,2] == k_bin)*(indices[3,2] == ii_bin)
                        datum_r = imgs_cad[:,0][:,id_datum_r] # K, N_
                        datum_g = imgs_cad[:,1][:,id_datum_g] # K, N_
                        datum_b = imgs_cad[:,2][:,id_datum_b] # K, N_
                        pPi_r, pPi_g, pPi_b = pPi_dict[(i_bin*bin_width, j_bin*bin_width, k_bin*bin_width, ii_bin*bin_width)]
                        # print(datum_r.shape, pPi)
                        img_ens[:,0][:,id_datum_r] = (pad_zero_one(datum_r) * pPi_r[:,None]).sum(0, keepdim=True)
                        img_ens[:,1][:,id_datum_g] = (pad_zero_one(datum_g) * pPi_g[:,None]).sum(0, keepdim=True)
                        img_ens[:,2][:,id_datum_b] = (pad_zero_one(datum_b) * pPi_b[:,None]).sum(0, keepdim=True)
    if not return_weight:
        return img_ens
    else:
        return img_ens, wei_ens



def ensemble_avg(imgs_cad, pixel_wise=True, return_weight=False):
    num_cad = imgs_cad.shape[0]
    weights = torch.ones(num_cad)
    N = imgs_cad.shape[0]
    if weights.dim() == 1:
        weights = weights[...,None,None,None]
        weights = weights.softmax(0)
    # else:
    #     weights = weights.clamp(0.1,0.9)
    #     weights = torch.cat((weights, 1-weights), 0)
    if pixel_wise:
        result = imgs_cad.mean(0, keepdim=True)
    else:
        result = (imgs_cad * weights).sum(0, keepdim=True)
    if not return_weight:
        return result
    else:
        return result, torch.ones_like(imgs_cad)*(1/N)


def ensemble_zzpm(imgs_cad, return_weight=False):
    K = imgs_cad.shape[0]
    mean = imgs_cad.mean(0, keepdim=True)
    # mean = imgs_cad.prod(0, keepdim=True).pow(1/K)
    MSE = (imgs_cad - mean).pow(2).mean((1,2,3), keepdim=True)
    weights = (1/(1e-3 + MSE)).softmax(0)
    result = (imgs_cad * weights).sum(0, keepdim=True)
    if not return_weight:
        return result
    else:
        return result, torch.ones_like(imgs_cad)*weights

def compute_regressor_weight(imgs_cad, img_gt, ensemble_type):
    if ensemble_type == 'gradient_boosting':
        from sklearn.experimental import enable_hist_gradient_boosting  # noqa
        from sklearn.ensemble import GradientBoostingRegressor
        regressor = GradientBoostingRegressor()
    elif ensemble_type == 'adaboost':
        from sklearn.ensemble import AdaBoostRegressor
        regressor = AdaBoostRegressor()
    elif ensemble_type == 'bagging':
        from sklearn.ensemble import BaggingRegressor
        regressor = BaggingRegressor()
    elif ensemble_type == 'extra_trees':
        from sklearn.ensemble import ExtraTreesRegressor
        regressor = ExtraTreesRegressor()
    elif ensemble_type == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor()
    elif ensemble_type == 'hist_gradient_boosting':
        from sklearn.experimental import enable_hist_gradient_boosting  # noqa
        from sklearn.ensemble import HistGradientBoostingRegressor
        regressor = HistGradientBoostingRegressor()
    weight = None
    X = imgs_cad.flatten(1).transpose(0,1).numpy()
    y = img_gt.flatten(1).transpose(0,1).numpy()
    weight = regressor.fit(X, y)
    weight.score(X, y)
    return weight

def ensemble_regression(imgs_cad, weight):
    b,c,h,w = imgs_cad.shape
    X = imgs_cad.flatten(1).transpose(0,1).numpy()
    img_ens = weight.predict(X)
    img_ens = torch.from_numpy(img_ens).view(1,c,h,w)
    return img_ens
