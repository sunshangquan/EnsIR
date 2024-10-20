import argparse
import cv2
import os
import numpy as np
import torch

from pytorch_msssim import ssim, ms_ssim, SSIM
from torchmetrics.image import PeakSignalNoiseRatio
from util_calculate_psnr_ssim import calculate_psnr, calculate_ssim
def rgb2ycbcrTorch(im, only_y=True, ): 
    '''
    same as matlab rgb2ycbcr
    Input:
        im: float [0,1], N x 3 x H x W
        only_y: only return Y channel
    '''
    scale = 255.0 if im.max() <= 1 else 1.
    
    im_temp = im.permute([0,2,3,1]) * scale  # N x H x W x C --> N x H x W x C, [0,255]
    
    # convert
    if only_y:
        rlt = torch.matmul(im_temp, torch.tensor([65.481, 128.553, 24.966], device=im.device, dtype=im.dtype).view([3,1])/ 255.0) + 16.0
    else:
        rlt = torch.matmul(im_temp, torch.tensor([[65.481,  -37.797, 112.0  ],
                                                  [128.553, -74.203, -93.786],
                                                  [24.966,  112.0,   -18.214]],
                                                  device=im.device, dtype=im.dtype)/255.0) + \
                                                    torch.tensor([16, 128, 128]).view([-1, 1, 1, 3])
    rlt /= scale
    rlt.clamp_(0.0, 255.0/scale)
    return rlt.permute([0, 3, 1, 2])

def loadfiles(folder):
    files = os.listdir(folder)
    return natsorted(files)

def resize(im, size, crop=True):
    if crop:
        return im[:size[0], :size[1]]
    else:
        return cv2.resize(im, size)

from natsort import natsorted

def np2torch(img):
    im = img.astype(np.float32) / 255
    im = torch.tensor(im).permute((2,0,1)).unsqueeze(0)
    return im.cuda()

def mean_squared_error(img1, img2):
    return (img1 - img2).pow(2).sum((1,2,3)).mean()
def structural_similarity(img1, img2, data_range):
    
    return ssim( img1, img2,  data_range=data_range, size_average=False) 

def peak_signal_noise_ratio(img1, img2, data_range):
    psnr = PeakSignalNoiseRatio(data_range=data_range)
    return psnr(img1, img2)
    # mse = np.mean((img1.numpy() - img2.numpy()) ** 2)
    # if mse == 0:
    #     return float('inf')
    # return 20. * np.log10(255. / np.sqrt(mse))
    

def compute_metrics(img1, img2, crop_border=0, metrics='all', ycbcr=True, data_range=1.):
    img1 = img1.type(torch.float64)
    img2 = img2.type(torch.float64)
    # crop = True
    # if img1.shape != img2.shape:
    #     if not crop:
    #         img1 = resize(img1, img2.shape[:2][::-1], False)
    #     else:
    #         img1 = resize(img1, img2.shape, True)
    
    if crop_border != 0:
        img1 = img1[..., crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]
    
    if ycbcr:
        img1 = rgb2ycbcrTorch(img1, True)
        img2 = rgb2ycbcrTorch(img2, True)
    results = []
    for metric in metrics:
        if metric in ['mse']:
            MSE = mean_squared_error(img1, img2)
            results.append(MSE)
        if metric in ['psnr']:
            PSNR = peak_signal_noise_ratio(img1, img2, data_range=data_range)
            # PSNR = calculate_psnr(255*img1[0].permute(1,2,0).numpy(), 255*img2[0].permute(1,2,0).numpy(), 4, test_y_channel=ycbcr
            results.append(PSNR)
        if metric in ['ssim']:
            SSIM = structural_similarity(img1, img2, data_range=data_range)
            results.append(SSIM)
    return results
    

