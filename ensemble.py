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

def read_imgs(path, dataset, methods, filenames, if_flatten=False, crop=True):
    imgs = []
    for filename in filenames:
        datum = []
        for method in methods+['gt',]:
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

def ensemble(imgs_cad, img_gt, methods, weight_file, bin_width=10, ensemble_type='ours', weight=None, log_file=None, crop_border=0, test_y_channel=True, verbose=False, metric_types=['psnr', 'ssim']):
    num_cad = len(methods)
    
    # print(pPi_dict)
    time1 = time.time()
    if ensemble_type == 'ours':
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
            print(methods[i], " ".join([str(i.item()) for i in result]))
        for i in range(len(metric_types)):
            results[i].append(result[i].item())
        if log_file is not None:
            with open(log_file, 'a+') as f_:
                f_.write("{}: {}\n".format(methods[i], " ".join([str(result_single.item()) for result_single in result])))

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
    parser.add_argument('-task', type=str, default='deblurring', help='')
    parser.add_argument('-ensemble_type', type=str, default='ours', help='')
    parser.add_argument('-dataset_id', type=int, default=2, help='')


    parser.add_argument('-bin_width', type=int, default=32, help='')
    parser.add_argument('-if_save_fig', type=int, default=0, help='')
    parser.add_argument('-specified_file', type=str, default='GOPR0410_11_00-000230', help='')
    parser.add_argument('-weight_file_path', type=str, default='weight', help='')
    parser.add_argument('-log_root_path', type=str, default='logs', help='')
    parser.add_argument('-save_root_path', type=str, default='result', help='')
    parser.add_argument('-data_root_path', type=str, default='/home/ssq/Desktop/phd/proj_ensembleIR/datasets/', help='')
    args = parser.parse_args()

    metric_types = ['psnr', 'ssim']
    verbose = False
    task = args.task
    weight_file_path = args.weight_file_path
    os.makedirs(weight_file_path, exist_ok=True)
    test_y_channel = False
    crop_border = 0
    path = os.path.join(args.data_root_path, task)
    if task == 'SR':
        methods = ["MambaIR", "SRFormer", 'SwinIR',]
        datasets = ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']
        test_y_channel = True
        crop_border = 4
    elif task == 'deblurring':
        methods = ["MPRNet", "Restormer", 'DGUNet']
        datasets = ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']
    elif task == 'deraining':
        methods = ["MPRNet", "MAXIM", "Restormer"]
        datasets = ['Rain100H', 'Rain100L', 'Test100', 'Test1200', 'Test2800']
        test_y_channel = True
    elif task == 'LLIE':
        path = '/home/ssq/Desktop/phd/proj_ensembleIR/datasets/LLIE/'
        methods = ["RQ-LLIE", 'RetinexFormer','CIDNet']
        datasets = ['LOLv1', 'LOLv2-real', 'LOLv2-synthetic']
    dataset = datasets[args.dataset_id]
    idx = 2
    print(idx)
    ensemble_type = args.ensemble_type

    assert ensemble_type in ['ours', 'avg', 'zzpm', 
    'hist_gradient_boosting', 
    'gradient_boosting', 
    'adaboost', 
    'bagging', 
    'extra_trees', 
    'random_forest', ]

    save_path = os.path.join(args.save_root_path, task, dataset, ensemble_type)
    os.makedirs(save_path, exist_ok=True)
    # filenames_train = get_files_names(os.path.join(path, 'train'), 'gt', '')
    log_path = os.path.join(args.log_root_path, task, dataset)
    os.makedirs(log_path, exist_ok=True)
    filenames_test = get_files_names(os.path.join(path, 'test'), 'gt', dataset)
    bin_width = args.bin_width
    weight_file = os.path.join(weight_file_path, "weight_{}_{}_b{}_rgb.pth".format(dataset, "_".join(methods), bin_width))
    weight = None
    default_weight = None
    print("Loading ensemble method:", ensemble_type)
    if ensemble_type == 'ours':# and not os.path.exists(weight_file):
        filenames_train = get_files_names(os.path.join(path, 'train'), 'gt', '')
        imgs_cad, img_gt = read_imgs(os.path.join(path, 'train'), '', methods, filenames_train[idx:idx+10], if_flatten=True)
        weight = compute_ensemble_weight(imgs_cad, img_gt, weight_file, bin_width, verbose, default_weight) 
    elif ensemble_type not in ['ours', 'avg', 'zzpm']:
        filenames_train = get_files_names(os.path.join(path, 'train'), 'gt', '')
        imgs_cad, img_gt = read_imgs(os.path.join(path, 'train'), '', methods, filenames_train[:10], if_flatten=True)
        weight = compute_regressor_weight(imgs_cad, img_gt, ensemble_type) 
        print(weight)

    log_file = os.path.join(log_path, "{}_{}".format(ensemble_type, "+".join(methods)))
    if os.path.isfile(log_file):
        os.remove(log_file)
    time_used = 0
    metrics = []
    for idx, filename in enumerate(filenames_test):
        if verbose:
            print(idx, filename)
        imgs_cad, img_gt = read_imgs(os.path.join(path, 'test'), dataset, methods, [filename,])
        if log_file is not None:
            with open(log_file, 'a+') as f_:
                f_.write("File: {}\n".format(filename))
        result, img_ens, time_single = ensemble(imgs_cad, img_gt, methods, weight_file, bin_width, ensemble_type, weight=weight, log_file=log_file, crop_border=crop_border, test_y_channel=test_y_channel, verbose=verbose, metric_types=metric_types)
        time_used += time_single
        metrics.append(result)
        if args.if_save_fig == 1:
            save_image(os.path.join(save_path, filename), img_ens)
    metrics = np.array(metrics)
    general_result = np.mean(metrics, 0)
    print(general_result)
    if log_file is not None:
        with open(log_file, 'a+') as f_:
            f_.write("Total time: {}\n".format(time_used))
            for result_line in general_result:
                f_.write("{}\n".format(" ".join([str(i) for i in result_line])))
