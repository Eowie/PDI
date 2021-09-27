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
plt.close('all')

#changing directory to where the image is located
os.chdir('C:/Users/Eow/Desktop/Mestrado/PDI/TP')
nome = 'Marilyn'
ext = '.tif'
Img = imread('marilyn.tif').astype(float)

plt.figure()
plt.imshow(Img, 'gray'); plt.axis('off');

filtro = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]],
                  dtype='float')/9

filtro1= np.ones((5,5))
filtro2= np.ones((9,9))
            
#                     [1,1,1],
#                     [1,1,1]],
#                       dtype='float')/9

conv0= np.zeros(Img.shape)
for i in range (1, Img.shape[0]-1):
    for j in range (1,Img.shape[1]-1):
        conv0[i,j]= filtro[0, 0]*Img[i-1, j-1] + filtro[0, 1]*Img[i-1, j] + filtro[0, 2]*Img[i-1, j+1]  + \
                    filtro[1, 0]*Img[  i, j-1] + filtro[1, 1]*Img[  i, j] + filtro[1, 2]*Img[  i, j+1]  + \
                    filtro[2, 0]*Img[i+1, j-1] + filtro[2, 1]*Img[i+1, j] + filtro[2, 2]*Img[i+1, j+1]
                    
                    
                                     
conv1 = ndimage.convolve(Img.astype(float), filtro1, mode='constant', cval=0)
conv2 = ndimage.convolve(Img.astype(float), filtro2, mode='constant', cval=0)

plt.figure(figsize=(14,5));
plt.subplot(141); plt.imshow(Img, 'gray'); plt.axis('off');
plt.subplot(142); plt.imshow(conv0, 'gray'); plt.axis('off'),plt.title('Conv0')
plt.subplot(143); plt.imshow(conv1, 'gray'); plt.axis('off'),plt.title('Conv1')
plt.subplot(144); plt.imshow(conv1, 'gray'); plt.axis('off'),plt.title('Conv2')


gauss= imread('noisy_gauss.tif').astype(float)
sp = imread('noisy_sp.tif').astype(float)

# # mean filter
mean_denoisedga = ndimage.convolve(gauss,filtro,mode='constant', cval=0)
mean_denoisedsp = ndimage.convolve(sp,filtro,mode='constant', cval=0)
# mean_denoised = ndimage.convolve(filtro, gauss, mode='constant', cval=0.0)

# gauss filter
gauss_denoisedga = ndimage.filters.gaussian_filter(gauss, 1)
gauss_denoisedsp = ndimage.filters.gaussian_filter(sp, 1)

#median filter
median_denoisedga = ndimage.filters.median_filter(gauss, 3)
median_denoisedsp = ndimage.filters.median_filter(sp, 3)


plt.figure(figsize=(14,5));
plt.subplot(241); plt.imshow(gauss, 'gray'); plt.axis('off');plt.title('Noisy Gauss')
plt.subplot(242); plt.imshow(mean_denoisedga, 'gray'); plt.axis('off'),plt.title('Mean Denoised')
plt.subplot(243); plt.imshow(gauss_denoisedga, 'gray'); plt.axis('off'),plt.title('Gauss Denoised')
plt.subplot(244); plt.imshow(median_denoisedga, 'gray'); plt.axis('off'),plt.title('Median Denoised')
plt.subplot(245); plt.imshow(sp, 'gray'); plt.axis('off'); plt.title ('Noisy SP')
plt.subplot(246); plt.imshow(mean_denoisedsp, 'gray'); plt.axis('off'),plt.title('Mean Denoised')
plt.subplot(247); plt.imshow(gauss_denoisedsp, 'gray'); plt.axis('off'),plt.title('Gauss Denoised')
plt.subplot(248); plt.imshow(median_denoisedsp, 'gray'); plt.axis('off'),plt.title('Median Denoised')

#prewitt filter
px = ndimage.prewitt(Img, axis=0, mode='constant')
py = ndimage.prewitt(Img, axis=1, mode='constant')
prw = np.abs(px)+np.abs(py)

# Sobel Filter
sx = ndimage.sobel(Img, axis=0, mode='constant')
sy = ndimage.sobel(Img, axis=1, mode='constant')
sob= np.hypot(sx, sy)

plt.figure(figsize=(14,5))
plt.subplot(141); plt.imshow(Img,'gray')
plt.subplot(142); plt.imshow(prw,'gray', vmin=np.min(prw), vmax=np.max(prw))
plt.subplot(143); plt.imshow(sob,'gray', vmin=np.min(sob), vmax=np.max(sob))


#Unsharp gauss

Un1= ndimage.filters.gaussian_filter(Img, 1)
Un2= Img - Un1
k=0.2
k1=0.6
Un3 = Img+k*Un2
Un7 = Img+k1*Un2

#unsharp mean
Un4= ndimage.convolve(Img,filtro,mode='constant', cval=0)
Un5 = Img - Un4
k1=0.2
Un6 = Img+k*Un5


plt.figure(figsize=(14,5))
plt.subplot(141); plt.imshow(Un3, 'gray')
plt.subplot(142); plt.imshow(Un7, 'gray')

# #Gauss Laplace
#criar filtro
l=3
x, y = np.meshgrid( np.linspace(-l, l, 2*l+1), np.linspace(-l, l, 2*l+1))
sigma = 1
x0=0
y0=0

LoG = lambda x, y:-1/(np.pi*sigma**4)*(1-((x-x0)**2+(y-y0)**2)/(2*sigma**2))*np.e**(-((x-x0)**2+(y-y0)**2)/(2*sigma**2))
filtro_log =LoG(x, y)

conv3 = ndimage.convolve(Img.astype(float), filtro_log, mode='constant', cval=0)

plt.figure(figsize=(14,5))
plt.subplot(141); plt.imshow(conv3, 'gray')



noise=imread('Stripping_Noise.tif').astype(float)

mean_noise = ndimage.convolve(noise,filtro,mode='constant', cval=0)
sobel_noisex= ndimage.sobel(mean_noise, axis=0, mode='constant')
sobel_noisey= ndimage.sobel(mean_noise, axis=1, mode='constant')
sobel_noise = np.hypot(sobel_noisex, sobel_noisey)

plt.figure(figsize=(14,5))
plt.subplot(141); plt.imshow(noise, 'gray')
plt.subplot(142); plt.imshow(mean_noise, 'gray')
plt.subplot(143); plt.imshow(sobel_noise, 'gray')

                                         
             