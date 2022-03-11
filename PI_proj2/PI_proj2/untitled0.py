# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 05:11:17 2022

@author: silam
"""

from imageio import imread
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage.morphology import disk, rectangle, reconstruction, \
    binary_erosion, binary_dilation, binary_opening, binary_closing
    
    

# filtros passa baixa
filtro = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]],
                  dtype='float')/9

matriz = ([90,76,42,77,73,74,120,78,80,83],
          [89,74,70,73,72,71,73,200,77,20],
          [89,72,68,70,70,40,70,71,76,80],
          [87,71,70,00,68,69,68,71,73,70],
          [86,50,68,68,66,67,69,70,70,72],
          [85,102,66,65,66,65,67,69,70,72],
          [82,66,64,64,66,5,67,69,50,70],
          [83,66,63,63,00,65,68,70,72,72],
          [82,66,64,64,63,65,67,69,73,74],
          [85,85,96,34,57,85,86,98,72,102])
                  
conv1 = ndimage.convolve(matriz, filtro, mode='constant', cval=0)
gauss = ndimage.filters.gaussian_filter(matriz, 1)
median = ndimage.filters.median_filter(matriz,3)

filtro2 = np.array([[-1,-1,-1],
                   [-1,8,-1],
                   [-1,-1,-1]],
                  dtype='float')/9

conv2 = ndimage.convolve(matriz, filtro2, mode='constant', cval=0)
test = matriz - conv1

#filtros passa alta

rob_v = np.array( [[1, 0 ],
                   [0,-1 ]] )
  
rob_h = np.array( [[ 0, 1 ],
                   [ -1, 0 ]] )
  
roby=ndimage.convolve(matriz,rob_v)
robx=ndimage.convolve(matriz,rob_h)
rob=np.abs(roby)+np.abs(robx)


sx = ndimage.sobel(matriz, axis=0, mode='constant')
sy = ndimage.sobel(matriz, axis=1, mode='constant')
sob= np.hypot(sx, sy)

px = ndimage.prewitt(matriz, axis=0, mode='constant')
py = ndimage.prewitt(matriz, axis=1, mode='constant')
prw = np.abs(px)+np.abs(py)

#filtros passa banda

pb_media= ndimage.convolve(matriz,filtro,mode='constant', cval=0)
pb1 = matriz - pb_media
k3=0.6
pb = matriz+k3*pb1



ee = ([1,1,1],
     [1,1,1],
     [1,1,1])


be=binary_erosion(matriz2,ee)