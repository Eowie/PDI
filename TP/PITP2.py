# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 18:40:16 2021

@author: silam
"""

from imageio import imread, imwrite
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

#changing directory to where the image is located
os.chdir('C:/Users/silam/OneDrive/Desktop/Mestrado/PDI/TP')
Img=imread('marilyn.tif')

filtro = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]],
                     dtype='float')/9

conv0 = np.zeros(Img.shape)
for i in range (1, Img.shape[0]-1):
    for j in range (1,Img.shape[1]-1):
        conv0[i,j]= Img[_, _]* filtro[_, _] + Img[_, _]* filtro[_, _] + Img[_, _]* filtro[_, _] + \
                    Img[_, _]* filtro[_, _] + Img[_, _]* filtro[_, _] + Img[_, _]* filtro[_, _] + \
                    Img[_, _]* filtro[_, _] + Img[_, _]* filtro[_, _] + Img[_, _]* filtro[_, _]
                    
conv1 = ndimage.convolve(Img.astype(float), filtro, mode='constant', cval=0)

gauss= imread('noisy_gauss.tif')
sp = imread('noisy_sp.tif')

# #mean filter
# mean_denoised = ndimage.convolve(F, h, mode='constant', cval=0.0)

# #gauss filter
# gauss_denoised = ndimage.filters.gaussian_filter(F, 1)

# #median filter
# median_denoised = ndimage.filters.median_filter(F, 3)

# #prewitt filter
# sx = ndimage.prewitt(F, axis=0, mode='constant')
# sy = ndimage.prewitt(F, axis=1, mode='constant')
# prw = np.hypot(sx, sy)

# # Sobel Filter
# sx = ndimage.sobel(F, axis=0, mode='constant')
# sy = ndimage.sobel(F, axis=1, mode='constant')
# sob= np.hypot(sx, sy)

#laplace filter

sx = ndimage.

