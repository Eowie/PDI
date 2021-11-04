# -*- coding: utf-8 -*-
"""

@author: Guilherme Rodrigues Gomes Martinez
fc50387
"""

"""Projeto 2 ex1"""
from skimage.morphology import disk, rectangle, erosion, dilation, opening, closing, local_minima, local_maxima, watershed, binary_erosion, binary_dilation, binary_opening, binary_closing
from imageio import imread, imwrite
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
plt.close('all')

Img1 = imread('lsat01.tif')
#imagem com 3 bandas, converter para apenas uma com a media
Img1 = np.uint(Img1[:,:,0]/3 +Img1[:,:,1]/3 + Img1[:,:,2]/3)

d = 3
se = rectangle(d,d)

#Transformação da imagem em imagem binária
aux = 20
Img2 = np.zeros(Img1.shape)
Img2[Img1>aux] = 0
Img2[Img1<aux] = 1

#operaçóes morfologicas binarias para ter uma representação da linha do rio
ope = binary_opening(Img2, se)

clo = binary_closing(ope,se)


#funcçao da reconstruçao dual
def reconstrucao_dual(mask, marker):
    a = 1
    ee = rectangle(3,3)
    while a!=0:
        E = erosion(marker,ee)
        R = np.maximum(mask.astype(float), E.astype(float))
        a = np.count_nonzero(marker!=R)
        marker = deepcopy(R)
    return R

Dist = ndimage.distance_transform_edt(clo==1)

Dist1 = Dist + 1

r_d_dist = reconstrucao_dual(Dist, Dist1)

#minimo regionak que vai ser usado como marker
minimo_reg = r_d_dist - Dist  


#definir o marker
markers, n= ndimage.label(minimo_reg)

#watershed da imagem
Ws = watershed(clo, markers, mask = np.ones(clo.shape))
d = 1
ee = disk(d)
#linha que representa a linha media do rio/linha interior
Lint = Ws- erosion(Ws, ee)
#Lext = dilation(Ws,ee)- Ws
#bacia = np.logical_or(Lint,Lext)

#Imagem final do rio representado com a linha média 
Imgf = clo + Lint

plt.figure(figsize=(12,10))
a = plt.subplot(121); plt.imshow(clo, 'gray');plt.axis('off');
b = plt.subplot(122); plt.imshow(Imgf, 'gray');plt.axis('off');

a.title.set_text('Imagem binaria do rio')
b.title.set_text('Imagem do rio com a linha média')