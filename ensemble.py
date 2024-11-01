import os
import cv2
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import time
from natsort import natsorted
from compute_psnr import compute_metrics
from ensemble_util import *

EXTENSIONS = ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']
def read_img(path, filename):
    img_path = os.path.join(path, filename)
    img = None
    for extension in EXTENSIONS:
        file = img_path+"."+extension
        if os.path.isfile(file):
            img = cv2.imread(file, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            break
    assert img is not None, "image file {} with any extension of {} does not exist".format(img_path, EXTENSIONS)
    tran = transforms.ToTensor()
    img_tensor = tran(img)*255.
    # print(img_tensor)
    return img_tensor

def read_imgs(path, dataset, models, filenames, if_flatten=False, crop=True):
    imgs = []
    for filename in filenames:
        datum = []
        for method in models+['gt',]:
            img_path = os.path.join(path, method, "{}".format(dataset))
            img = read_img(img_path, filename)
            if len(datum) > 0  and img.shape != datum[0].shape:
                shape = datum[0].shape
                if crop:
                    img = img[...,:shape[-2],:shape[-1]]
                else:
                    img = torch.nn.functional.interpolate(img, shape[2:])
            datum.append(img.unsqueeze(0))
            
        datum = torch.cat(datum, 0)
        if if_flatten:
            datum = datum.flatten(2)
        imgs.append(datum)
    imgs = torch.cat(imgs, -1)
    
    imgs_cad = imgs[:-1]
    img_gt = imgs[-1:]
    return imgs_cad, img_gt

def ensemble(imgs_cad, img_gt, models, weight_file, bin_width=10, ensemble_type='ensir', weight=None, log_file=None, crop_border=0, test_y_channel=True, verbose=False, metric_types=['psnr', 'ssim']):
    num_cad = len(models)
    
    # print(pPi_dict)
    time1 = time.time()
    if ensemble_type == 'ensir':
        pPi_dict = read_dict(weight_file) if weight is None else weight
        img_ens = ensemble_single(pPi_dict, imgs_cad, bin_width)
    elif ensemble_type == 'avg':
        img_ens = ensemble_avg(imgs_cad, )
    elif ensemble_type == 'zzpm':
        img_ens = ensemble_zzpm(imgs_cad,)
    elif ensemble_type in ['hist_gradient_boosting', 'gradient_boosting', 'adaboost', 'bagging', 'extra_trees', 'random_forest', 'stacking', 'voting']:
        assert weight is not None
        img_ens = ensemble_regression(imgs_cad, weight) 
    time2 = time.time()
    # print(img_ens, img_gt, img_ens.shape, img_gt.shape)
    results = [[] for i in range(len(metric_types))]
    result = compute_metrics(img_ens, img_gt, crop_border, metric_types, test_y_channel, data_range=255.)
    for i in range(len(metric_types)):
        results[i].append(result[i].item())
    if verbose:
        print("Emsemble:", " ".join([str(i.item()) for i in result])) 
    if log_file is not None:
        with open(log_file, 'a+') as f_:
            f_.write("Emsemble: {}\n".format(" ".join([str(i.item()) for i in result])))
    for i in range(num_cad):
        result = compute_metrics(imgs_cad[i:i+1], img_gt, crop_border, metric_types, test_y_channel, data_range=255.)
        if verbose:
            print(models[i], " ".join([str(i.item()) for i in result]))
        for i in range(len(metric_types)):
            results[i].append(result[i].item())
        if log_file is not None:
            with open(log_file, 'a+') as f_:
                f_.write("{}: {}\n".format(models[i], " ".join([str(result_single.item()) for result_single in result])))

    if log_file is not None:
        with open(log_file, 'a+') as f_:
            f_.write("\n")
    return results, img_ens, time2 - time1

def save_image(path, img):
    torchvision.utils.save_image(img_ens[0]/255., path+'.jpg')

def get_files_names(path, method, dataset):
    img_path = os.path.join(path, method, "{}".format(dataset))
    files = [".".join(i.split('.')[:-1]) for i in os.listdir(img_path)]
    return files

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', default='opt/deraining/Rain100L.yaml', type=str, help='Path to config file')
    args = parser.parse_args()
    yaml_file = args.yaml_file
    import yaml

    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    yml = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

    metric_types = yml['ensemble'].pop('metric_types')
    verbose = yml['verbose']
    task = yml['task']
    if_save_fig = yml['ensemble'].pop('if_save_fig')
    weight_file_path = yml['ensemble'].pop('weight_file_path')
    os.makedirs(weight_file_path, exist_ok=True)
    test_y_channel = yml['ensemble'].pop('test_y_channel')
    crop_border = yml['ensemble'].pop('crop_border')
    path = os.path.join(yml['dataset'].pop('data_root_path'), task)
    path_refn = yml['dataset'].pop('data_refn_path')
    dataset = yml['dataset'].pop('name')
    idx = 2
    print(idx)
    ensemble_type = yml['ensemble'].pop('name')
    models = yml['models']
    assert ensemble_type in ['ensir', 'avg', 'zzpm', 
    'hist_gradient_boosting', 
    'gradient_boosting', 
    'adaboost', 
    'bagging', 
    'extra_trees', 
    'random_forest', ]

    save_path = os.path.join(yml['ensemble'].pop('save_root_path'), task, dataset, ensemble_type)
    os.makedirs(save_path, exist_ok=True)
    # filenames_train = get_files_names(os.path.join(path, 'train'), 'gt', '')
    log_path = os.path.join(yml['ensemble'].pop('log_root_path'), task, dataset)
    os.makedirs(log_path, exist_ok=True)
    filenames_test = get_files_names(os.path.join(path, 'test'), 'gt', dataset)
    bin_width = yml['ensemble'].pop('bin_width')
    weight_file = os.path.join(weight_file_path, "weight_{}_{}_b{}_rgb.pth".format(dataset, "_".join(models), bin_width))
    weight = yml['ensemble'].pop('precompute_weight')
    default_weight = yml['ensemble'].pop('default_weight')
    default_weight = torch.tensor(default_weight) if default_weight is not None else None
    print("Loading ensemble method:", ensemble_type)
    if ensemble_type == 'ensir':# and not os.path.exists(weight_file):
        filenames_train = get_files_names(path_refn, 'gt', '')
        imgs_cad, img_gt = read_imgs(path_refn, '', models, filenames_train, if_flatten=True)
        print(filenames_train[idx:idx+10])
        weight = compute_ensemble_weight(imgs_cad, img_gt, weight_file, bin_width, verbose, default_weight) 
    elif ensemble_type not in ['ensir', 'avg', 'zzpm']:
        filenames_train = get_files_names(os.path.join(path, 'train'), 'gt', '')
        imgs_cad, img_gt = read_imgs(os.path.join(path, 'train'), '', models, filenames_train, if_flatten=True)
        weight = compute_regressor_weight(imgs_cad, img_gt, ensemble_type) 
        print(weight)

    log_file = os.path.join(log_path, "{}_{}".format(ensemble_type, "+".join(models)))
    if os.path.isfile(log_file):
        os.remove(log_file)
    metrics = []
    for idx, filename in enumerate(filenames_test):
        if verbose:
            print(idx, filename)
        imgs_cad, img_gt = read_imgs(os.path.join(path, 'test'), dataset, models, [filename,])
        if log_file is not None:
            with open(log_file, 'a+') as f_:
                f_.write("File: {}\n".format(filename))
        result, img_ens, _ = ensemble(imgs_cad, img_gt, models, weight_file, bin_width, ensemble_type, weight=weight, log_file=log_file, crop_border=crop_border, test_y_channel=test_y_channel, verbose=verbose, metric_types=metric_types)
        metrics.append(result)
        if if_save_fig:
            save_image(os.path.join(save_path, filename), img_ens)
    metrics = np.array(metrics)
    general_result = np.mean(metrics, 0)
    print(general_result)
    if log_file is not None:
        with open(log_file, 'a+') as f_:
            for result_line in general_result:
                f_.write("{}\n".format(" ".join([str(i) for i in result_line])))
