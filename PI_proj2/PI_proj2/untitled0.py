# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 15:28:48 2021

@author: Sílvia Mourão
FC57541
"""
# PDI - Projeto 2

from skimage.morphology import disk, rectangle, erosion, dilation, opening, closing, local_minima, local_maxima, watershed, binary_erosion, binary_dilation, binary_opening, binary_closing
from imageio import imread
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
plt.close('all')

#Ex 1

LSat0 = imread('lsat01.tif')

#1.1

# A imagem LSat01.tif é uma imagem RGB pelo que é necessario torná-la primeiro numa imagem de cinzentos
LSat0_R=LSat0[:,:,0]
LSat0_G=LSat0[:,:,1]
LSat0_B=LSat0[:,:,2]

#Conversão numa imagem de cinzentos a partir da média dos valores de cinzento nas três bandas
LSat=(LSat0_R/3 + LSat0_G + LSat0_B/3).astype(int)

g,r = np.histogram(LSat, bins=256, range=(0,256))

plt.figure()
plt.subplot(111); plt.plot(g)

# por analise do histograma vemos que a zona de maior transicao se encontra entre os valores de cinzento 19 e 29
# podemos procurar o melhor valor para a limiarizacao dentro deste intervalo

#Seleção de um threshold para imagem para que se obtenha uma imagem binaria do leito do rio
lim=20 #valor medio do intervalo 19-29

#Threshold da imagem
LSat_bin=np.zeros(LSat.shape)
LSat_bin[LSat>lim] = 0
LSat_bin[LSat<lim] = 1

plt.figure()
plt.subplot(111); plt.imshow(LSat_bin)

